// pc_transform_node.cpp
//
// C++ port of pc_transform.py.
//
// Transform a PointCloud2 into world frame, re-center it at a robust
// centroid of near-ground points (or a ray-ground intersection), and
// yaw-align it using the drone's yaw so the cloud's orientation is fixed
// in world. Optionally writes each aligned cloud + pose to disk for
// dataset generation.
//
// Parameters (same names as the Python version):
//   input_cloud_topic, output_cloud_topic
//   world_frame, lidar_frame, drone_frame
//   ground_z, forward_axis
//   origin_mode ('centroid' | 'ray')
//   ground_band, mad_k, min_ground_points
//   tilt_axis ('y' | 'z'), roll_scale, roll_tilt_sign
//   min_range, max_range, min_intensity, max_intensity
//   save_clouds, save_poses, output_dir, filename_prefix
//   subscription_depth, writer_queue_size, drop_when_full

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <tf2/LinearMath/Matrix3x3.hpp>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2/LinearMath/Vector3.hpp>
#include <tf2/time.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <pc_transform_cpp/msg/aligned_cloud_with_pose.hpp>


namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
//  Constants matching sensor_msgs::PointField datatypes
// ---------------------------------------------------------------------------
static constexpr uint8_t PF_INT8    = sensor_msgs::msg::PointField::INT8;
static constexpr uint8_t PF_UINT8   = sensor_msgs::msg::PointField::UINT8;
static constexpr uint8_t PF_INT16   = sensor_msgs::msg::PointField::INT16;
static constexpr uint8_t PF_UINT16  = sensor_msgs::msg::PointField::UINT16;
static constexpr uint8_t PF_INT32   = sensor_msgs::msg::PointField::INT32;
static constexpr uint8_t PF_UINT32  = sensor_msgs::msg::PointField::UINT32;
static constexpr uint8_t PF_FLOAT32 = sensor_msgs::msg::PointField::FLOAT32;
static constexpr uint8_t PF_FLOAT64 = sensor_msgs::msg::PointField::FLOAT64;

// ---------------------------------------------------------------------------
//  Field helpers
// ---------------------------------------------------------------------------

struct FieldInfo {
    uint32_t offset = 0;
    uint8_t  datatype = 0;
    uint32_t count = 0;
    bool     present = false;
};

static FieldInfo find_field(const sensor_msgs::msg::PointCloud2 & msg,
                            const std::string & name) {
    FieldInfo fi;
    for (const auto & f : msg.fields) {
        if (f.name == name) {
            fi.offset = f.offset;
            fi.datatype = f.datatype;
            fi.count = f.count;
            fi.present = true;
            break;
        }
    }
    return fi;
}

static inline double read_field_as_double(const uint8_t * base,
                                          const FieldInfo & fi) {
    const uint8_t * p = base + fi.offset;
    switch (fi.datatype) {
        case PF_FLOAT32: return static_cast<double>(*reinterpret_cast<const float *>(p));
        case PF_FLOAT64: return *reinterpret_cast<const double *>(p);
        case PF_INT8:    return static_cast<double>(*reinterpret_cast<const int8_t *>(p));
        case PF_UINT8:   return static_cast<double>(*reinterpret_cast<const uint8_t *>(p));
        case PF_INT16:   return static_cast<double>(*reinterpret_cast<const int16_t *>(p));
        case PF_UINT16:  return static_cast<double>(*reinterpret_cast<const uint16_t *>(p));
        case PF_INT32:   return static_cast<double>(*reinterpret_cast<const int32_t *>(p));
        case PF_UINT32:  return static_cast<double>(*reinterpret_cast<const uint32_t *>(p));
        default:         return std::nan("");
    }
}

// ---------------------------------------------------------------------------
//  Robust centroid (MAD-based trimming)
// ---------------------------------------------------------------------------

static double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    double m = v[mid];
    if (v.size() % 2 == 0) {
        std::nth_element(v.begin(), v.begin() + (mid - 1), v.end());
        m = 0.5 * (m + v[mid - 1]);
    }
    return m;
}

//static std::pair<double, std::vector<bool>>
//robust_centroid_1d(const std::vector<double> & values,
//                   double mad_k,
//                   int max_iter = 5,
//                   size_t min_keep = 10)
//{
//    const size_t n = values.size();
//    if (n == 0) return {0.0, {}};
//
//    std::vector<bool> mask(n, true);
//    std::vector<double> kept;
//    kept.reserve(n);
//
//    for (int it = 0; it < max_iter; ++it) {
//        kept.clear();
//        size_t prev_count = 0;
//        for (size_t i = 0; i < n; ++i) {
//            if (mask[i]) { kept.push_back(values[i]); ++prev_count; }
//        }
//        if (kept.size() < min_keep) break;
//
//        const double med = median_of(kept);
//        std::vector<double> dev;
//        dev.reserve(kept.size());
//        for (double v : kept) dev.push_back(std::abs(v - med));
//        const double mad = median_of(dev);
//        const double sigma = 1.4826 * mad;
//        if (sigma < 1e-12) break;
//
//        const double thresh = mad_k * sigma;
//        std::vector<bool> new_mask(n, false);
//        size_t kept_count = 0;
//        for (size_t i = 0; i < n; ++i) {
//            if (mask[i] && std::abs(values[i] - med) <= thresh) {
//                new_mask[i] = true;
//                ++kept_count;
//            }
//        }
//        if (kept_count == 0) {
//            return {med, std::vector<bool>(n, true)};
//        }
//        if (kept_count == prev_count) {
//            mask = std::move(new_mask);
//            break;
//        }
//        mask = std::move(new_mask);
//    }
//
//    double sum = 0.0;
//    size_t count = 0;
//    for (size_t i = 0; i < n; ++i) {
//        if (mask[i]) { sum += values[i]; ++count; }
//    }
//    if (count == 0) {
//        return {median_of(values), std::vector<bool>(n, true)};
//    }
//    return {sum / static_cast<double>(count), mask};
//}

static std::tuple<double, double, std::vector<bool>>
robust_centroid_2d(const std::vector<double> & x,
                   const std::vector<double> & y,
                   double mad_k)
{
    const size_t n = x.size();
    if (n == 0) return {0.0, 0.0, {}};

    std::vector<bool> mask_x(n, true), mask_y(n, true);

    for (int it = 0; it < 5; ++it) {
        std::vector<double> xs, ys;
        xs.reserve(n); ys.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            if (mask_x[i] && mask_y[i]) { xs.push_back(x[i]); ys.push_back(y[i]); }
        }
        if (xs.size() < 10) break;

        // X pass
        {
            double med = median_of(xs);
            std::vector<double> dev;
            dev.reserve(xs.size());
            for (double v : xs) dev.push_back(std::abs(v - med));
            double mad = median_of(dev);
            double sigma = 1.4826 * mad;
            if (sigma >= 1e-12) {
                double thresh = mad_k * sigma;
                std::vector<bool> new_mask_x(n, false);
                size_t kept = 0;
                for (size_t i = 0; i < n; ++i) {
                    if (mask_x[i] && mask_y[i] && std::abs(x[i] - med) <= thresh) {
                        new_mask_x[i] = true; ++kept;
                    }
                }
                if (kept >= 10) mask_x = std::move(new_mask_x);
            }
        }
        // Y pass on the new joint mask
        {
            xs.clear(); ys.clear();
            for (size_t i = 0; i < n; ++i) {
                if (mask_x[i] && mask_y[i]) { xs.push_back(x[i]); ys.push_back(y[i]); }
            }
            if (xs.size() < 10) break;
            double med = median_of(ys);
            std::vector<double> dev;
            dev.reserve(ys.size());
            for (double v : ys) dev.push_back(std::abs(v - med));
            double mad = median_of(dev);
            double sigma = 1.4826 * mad;
            if (sigma >= 1e-12) {
                double thresh = mad_k * sigma;
                std::vector<bool> new_mask_y(n, false);
                size_t kept = 0;
                for (size_t i = 0; i < n; ++i) {
                    if (mask_x[i] && mask_y[i] && std::abs(y[i] - med) <= thresh) {
                        new_mask_y[i] = true; ++kept;
                    }
                }
                if (kept >= 10) mask_y = std::move(new_mask_y);
            }
        }
    }

    double sx = 0.0, sy = 0.0;
    size_t count = 0;
    std::vector<bool> joint(n, false);
    for (size_t i = 0; i < n; ++i) {
        if (mask_x[i] && mask_y[i]) {
            sx += x[i]; sy += y[i]; joint[i] = true; ++count;
        }
    }
    if (count == 0) {
        return {median_of(x), median_of(y), std::vector<bool>(n, true)};
    }
    return {sx / count, sy / count, std::move(joint)};
}

// ---------------------------------------------------------------------------
//  Node
// ---------------------------------------------------------------------------

class AlignCloudNode : public rclcpp::Node {
public:
    AlignCloudNode() : rclcpp::Node("align_cloud_node") {
        // Parameters
        declare_parameter<std::string>("input_cloud_topic",
            "/sim_lidar/pointcloud/downsampled");
        declare_parameter<std::string>("output_cloud_topic",
            "/sim_lidar/pointcloud/aligned");
        declare_parameter<std::string>("output_aligned_topic",
            "/sim_lidar/pointcloud/aligned_with_pose");
        declare_parameter<std::string>("world_frame", "scene_0021_1");
        declare_parameter<std::string>("lidar_frame",
            "iris_with_lidar/lidar_link/robosense_emx192");
        declare_parameter<std::string>("drone_frame", "iris_with_lidar");
        declare_parameter<double>("ground_z", 0.0);
        declare_parameter<std::vector<double>>("forward_axis",
            std::vector<double>{1.0, 0.0, 0.0});

        declare_parameter<std::string>("origin_mode", "centroid");
        declare_parameter<double>("ground_band", 0.5);
        declare_parameter<double>("mad_k", 3.0);
        declare_parameter<int>("min_ground_points", 20);

        declare_parameter<std::string>("tilt_axis", "y");
        declare_parameter<double>("roll_scale", 1.0 / 3.0);
        declare_parameter<double>("roll_tilt_sign", 1.0);

        declare_parameter<double>("min_range", 0.0);
        declare_parameter<double>("max_range", 500.0);
        declare_parameter<double>("min_intensity", -1.0e9);
        declare_parameter<double>("max_intensity",  1.0e9);

        declare_parameter<bool>("save_clouds", true);
        declare_parameter<bool>("save_poses", true);
        declare_parameter<std::string>("output_dir", "/tmp/dnn_dataset");
        declare_parameter<std::string>("filename_prefix", "cloud_");

        declare_parameter<int>("subscription_depth", 20);
        declare_parameter<int>("writer_queue_size", 200);
        declare_parameter<bool>("drop_when_full", false);

        // Read parameters
        world_frame_   = get_parameter("world_frame").as_string();
        lidar_frame_   = get_parameter("lidar_frame").as_string();
        drone_frame_   = get_parameter("drone_frame").as_string();
        ground_z_      = get_parameter("ground_z").as_double();

        auto axis = get_parameter("forward_axis").as_double_array();
        double n2 = 0.0;
        for (double v : axis) n2 += v * v;
        if (n2 < 1e-18) throw std::runtime_error("forward_axis must be non-zero");
        double inv = 1.0 / std::sqrt(n2);
        forward_axis_ = {axis[0] * inv, axis[1] * inv, axis[2] * inv};

        origin_mode_ = get_parameter("origin_mode").as_string();
        if (origin_mode_ != "centroid" && origin_mode_ != "ray") {
            throw std::runtime_error("origin_mode must be 'centroid' or 'ray'");
        }
        ground_band_       = get_parameter("ground_band").as_double();
        mad_k_             = get_parameter("mad_k").as_double();
        min_ground_points_ = get_parameter("min_ground_points").as_int();

        tilt_axis_       = get_parameter("tilt_axis").as_string();
        roll_scale_      = get_parameter("roll_scale").as_double();
        roll_tilt_sign_  = get_parameter("roll_tilt_sign").as_double();

        min_range_     = get_parameter("min_range").as_double();
        max_range_     = get_parameter("max_range").as_double();
        min_intensity_ = get_parameter("min_intensity").as_double();
        max_intensity_ = get_parameter("max_intensity").as_double();

        save_clouds_    = get_parameter("save_clouds").as_bool();
        save_poses_     = get_parameter("save_poses").as_bool();
        output_dir_     = get_parameter("output_dir").as_string();
        filename_prefix_ = get_parameter("filename_prefix").as_string();

        subscription_depth_ = get_parameter("subscription_depth").as_int();
        writer_queue_size_  = get_parameter("writer_queue_size").as_int();
        drop_when_full_     = get_parameter("drop_when_full").as_bool();

        if (save_clouds_ || save_poses_) {
            std::error_code ec;
            fs::create_directories(output_dir_, ec);
            if (ec) {
                RCLCPP_ERROR(get_logger(), "Could not create output_dir '%s': %s",
                             output_dir_.c_str(), ec.message().c_str());
            }
        }

        // TF
        tf_buffer_   = std::make_shared<tf2_ros::Buffer>(get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Publisher
        auto pub_qos = rclcpp::SensorDataQoS();
        cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
            get_parameter("output_cloud_topic").as_string(), pub_qos);

        // This one publishes the same cloud but with pose information
        // It is to be used by the DNN for inference and global occupancy mapping
        aligned_pub_ = create_publisher<pc_transform_cpp::msg::AlignedCloudWithPose>(
            get_parameter("output_aligned_topic").as_string(), pub_qos);

        // Subscriber
        auto sub_qos = rclcpp::QoS(rclcpp::KeepLast(subscription_depth_))
                           .reliability(rclcpp::ReliabilityPolicy::BestEffort)
                           .durability(rclcpp::DurabilityPolicy::Volatile);
        cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            get_parameter("input_cloud_topic").as_string(),
            sub_qos,
            std::bind(&AlignCloudNode::cloud_callback, this, std::placeholders::_1));

        // Writer thread
        writer_stop_.store(false);
        writer_thread_ = std::thread(&AlignCloudNode::writer_loop, this);

        RCLCPP_INFO(get_logger(),
            "Aligning clouds from '%s' to world '%s', yaw source '%s', "
            "ground_z=%.3f, origin_mode='%s', ground_band=%.3f, mad_k=%.2f",
            lidar_frame_.c_str(), world_frame_.c_str(), drone_frame_.c_str(),
            ground_z_, origin_mode_.c_str(), ground_band_, mad_k_);
    }

    ~AlignCloudNode() override {
        {
            std::unique_lock<std::mutex> lk(queue_mutex_);
            writer_stop_.store(true);
        }
        queue_cv_.notify_all();
        if (writer_thread_.joinable()) writer_thread_.join();
    }

private:
    // ---- subscriber callback: just enqueue ----
    void cloud_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
        {
            std::unique_lock<std::mutex> lk(queue_mutex_);
            if (drop_when_full_ && writer_queue_size_ > 0 &&
                queue_.size() >= static_cast<size_t>(writer_queue_size_)) {
                // Silently drop. Counter is kept for potential future use.
                ++dropped_count_;
                return;
            }
            queue_.push(std::move(msg));
        }
        queue_cv_.notify_one();
    }

    // ---- writer thread ----
    void writer_loop() {
        while (true) {
            sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg;
            {
                std::unique_lock<std::mutex> lk(queue_mutex_);
                queue_cv_.wait_for(lk, std::chrono::milliseconds(200), [&]{
                    return writer_stop_.load() || !queue_.empty();
                });
                if (writer_stop_.load() && queue_.empty()) return;
                if (queue_.empty()) continue;
                cloud_msg = std::move(queue_.front());
                queue_.pop();
            }

            try {
                process_cloud(cloud_msg);
            } catch (const std::exception & e) {
                RCLCPP_ERROR(get_logger(), "[writer] Failed to process cloud: %s",
                             e.what());
            }
        }
    }

    // ---- main pipeline ----
    void process_cloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud_msg) {
        //const auto start = std::chrono::steady_clock::now(); // DEBUG
        const rclcpp::Time stamp(cloud_msg->header.stamp);

        // 1. Look up world <- cloud frame.
        geometry_msgs::msg::TransformStamped tf_wl;
        try {
            tf_wl = tf_buffer_->lookupTransform(
                world_frame_, cloud_msg->header.frame_id,
                stamp, tf2::durationFromSec(0.05));
        } catch (const tf2::TransformException & e) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                "[writer] No TF %s <- %s at %.9f: %s",
                world_frame_.c_str(), cloud_msg->header.frame_id.c_str(),
                stamp.seconds(), e.what());
            return;
        }

        // 2. Build the world-frame transform.
        tf2::Quaternion q_wl(tf_wl.transform.rotation.x,
                             tf_wl.transform.rotation.y,
                             tf_wl.transform.rotation.z,
                             tf_wl.transform.rotation.w);
        tf2::Matrix3x3 R_wl(q_wl);
        const double tx = tf_wl.transform.translation.x;
        const double ty = tf_wl.transform.translation.y;
        const double tz = tf_wl.transform.translation.z;

        // 3. Extract fields from the original cloud.
        auto fx = find_field(*cloud_msg, "x");
        auto fy = find_field(*cloud_msg, "y");
        auto fz = find_field(*cloud_msg, "z");
        if (!fx.present || !fy.present || !fz.present) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                "Cloud has no x/y/z fields — skipping.");
            return;
        }
        auto fint = find_field(*cloud_msg, "intensity");

        const size_t n = static_cast<size_t>(cloud_msg->width) * cloud_msg->height;
        std::vector<double> x(n), y(n), z(n), intensity(n, 0.0);
        {
            const uint8_t * data = cloud_msg->data.data();
            const size_t step = cloud_msg->point_step;
            for (size_t i = 0; i < n; ++i) {
                const uint8_t * base = data + i * step;
                x[i] = read_field_as_double(base, fx);
                y[i] = read_field_as_double(base, fy);
                z[i] = read_field_as_double(base, fz);
                if (fint.present) intensity[i] = read_field_as_double(base, fint);
            }
        }

        // 4. Transform points into world frame (raw loop).
        std::vector<double> xw(n), yw(n), zw(n);
        for (size_t i = 0; i < n; ++i) {
            const tf2::Vector3 p(x[i], y[i], z[i]);
            const tf2::Vector3 pw = R_wl * p + tf2::Vector3(tx, ty, tz);
            xw[i] = pw.x(); yw[i] = pw.y(); zw[i] = pw.z();
        }

        // 5. Drop non-finite points.
        std::vector<size_t> keep;
        keep.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            if (std::isfinite(xw[i]) && std::isfinite(yw[i]) &&
                std::isfinite(zw[i]) && std::isfinite(intensity[i])) {
                keep.push_back(i);
            }
        }
        if (keep.empty()) return;
        const size_t m = keep.size();
        std::vector<double> xf(m), yf(m), zf(m), inf(m);
        for (size_t k = 0; k < m; ++k) {
            xf[k] = xw[keep[k]];
            yf[k] = yw[keep[k]];
            zf[k] = zw[keep[k]];
            inf[k] = intensity[keep[k]];
        }

        // 6. Determine origin.
        double hx = 0.0, hy = 0.0, hz = ground_z_;
        if (origin_mode_ == "centroid") {
            std::vector<double> gx, gy;
            gx.reserve(m); gy.reserve(m);
            for (size_t k = 0; k < m; ++k) {
                if (std::abs(zf[k] - ground_z_) <= ground_band_) {
                    gx.push_back(xf[k]);
                    gy.push_back(yf[k]);
                }
            }
            if (static_cast<int>(gx.size()) < min_ground_points_) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                    "Only %zu points within ±%.3f m of ground_z; need >= %d. Skipping.",
                    gx.size(), ground_band_, min_ground_points_);
                return;
            }
            auto [cx, cy, mask] = robust_centroid_2d(gx, gy, mad_k_);
            hx = cx; hy = cy; hz = ground_z_;
        } else {
            auto hit = compute_ray_intersection(stamp);
            if (!hit.has_value()) return;
            hx = hit->x(); hy = hit->y(); hz = hit->z();
        }

        // 7. Yaw of the drone body in world.
        auto yaw_opt = get_drone_yaw(stamp);
        if (!yaw_opt.has_value()) return;
        const double yaw = *yaw_opt;

        // 8. Translate by -origin, rotate by Rz(-yaw).
        const double c = std::cos(-yaw);
        const double s = std::sin(-yaw);
        std::vector<double> xr(m), yr(m), zr(m);
        for (size_t k = 0; k < m; ++k) {
            const double x0 = xf[k] - hx;
            const double y0 = yf[k] - hy;
            const double z0 = zf[k] - hz;
            xr[k] = c * x0 - s * y0;
            yr[k] = s * x0 + c * y0;
            zr[k] = z0;
        }

        // 9. Publish aligned cloud with pose.
        auto out = std::make_unique<pc_transform_cpp::msg::AlignedCloudWithPose>();
        out->header = cloud_msg->header;
        out->header.frame_id = world_frame_;
        out->n_points = static_cast<uint32_t>(xr.size());
        out->origin_world.x = hx;
        out->origin_world.y = hy;
        out->origin_world.z = hz;
        out->yaw = yaw;

        // If you want the full drone orientation too:
        // (reuse tf_wd from get_drone_yaw if you cached it, or do another lookup)
        out->drone_orientation_world.x = 0.0;
        out->drone_orientation_world.y = 0.0;
        out->drone_orientation_world.z = std::sin(yaw * 0.5);
        out->drone_orientation_world.w = std::cos(yaw * 0.5);
        out->drone_position_world.x = hx;  // placeholder; use tf_wd if desired
        out->drone_position_world.y = hy;
        out->drone_position_world.z = hz;

        out->points.resize(3 * xr.size());
        float* dst = out->points.data();
        for (size_t k = 0; k < xr.size(); ++k) {
            dst[3 * k + 0] = static_cast<float>(xr[k]);
            dst[3 * k + 1] = static_cast<float>(yr[k]);
            dst[3 * k + 2] = static_cast<float>(zr[k]);
        }

        aligned_pub_->publish(std::move(out));
        // ########## DEBUG #############
        //const auto end = std::chrono::steady_clock::now();
        //const auto duration = 
        //std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        //RCLCPP_INFO(this->get_logger(),
        //        "Cloud processing time: %ld us (%.3f ms)",
        //        duration, duration / 1000.0);
        // ########## DEBUG END #############

        // 10. Publish in world frame for visualizing it in rviz2
        auto out_msg = build_cloud(cloud_msg, xr, yr, zr, inf, world_frame_);
        cloud_pub_->publish(*out_msg);

        // 11. Save (optional).
        if (save_poses_) save_pose(stamp, yaw, hx, hy, hz, m);
        if (save_clouds_) save_cloud(stamp, xr, yr, zr, inf);
    }

    // ---- ray-mode origin ----
    std::optional<tf2::Vector3> compute_ray_intersection(const rclcpp::Time & stamp) {
        geometry_msgs::msg::TransformStamped tf_wl, tf_wd;
        try {
            tf_wl = tf_buffer_->lookupTransform(world_frame_, lidar_frame_,
                                                stamp, tf2::durationFromSec(0.05));
        } catch (const tf2::TransformException & e) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                "No TF %s <- %s at %.9f: %s",
                world_frame_.c_str(), lidar_frame_.c_str(),
                stamp.seconds(), e.what());
            return std::nullopt;
        }
        try {
            tf_wd = tf_buffer_->lookupTransform(world_frame_, drone_frame_,
                                                stamp, tf2::durationFromSec(0.05));
        } catch (const tf2::TransformException & e) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                "No TF %s <- %s at %.9f: %s",
                world_frame_.c_str(), drone_frame_.c_str(),
                stamp.seconds(), e.what());
            return std::nullopt;
        }

        const double sx = tf_wl.transform.translation.x;
        const double sy = tf_wl.transform.translation.y;
        const double sz = tf_wl.transform.translation.z;

        tf2::Quaternion q_d(tf_wd.transform.rotation.x, tf_wd.transform.rotation.y,
                            tf_wd.transform.rotation.z, tf_wd.transform.rotation.w);

        tf2::Vector3 fwd = tf2::Matrix3x3(q_d) * tf2::Vector3(1.0, 0.0, 0.0);
        fwd.normalize();

        double roll, pitch, yaw;
        tf2::Matrix3x3(q_d).getRPY(roll, pitch, yaw);
        const double comp = roll_scale_ * roll_tilt_sign_ * roll;

        tf2::Vector3 lat = tf2::Matrix3x3(q_d) * tf2::Vector3(0.0, 1.0, 0.0);
        lat.normalize();

        const double ct = std::cos(comp);
        const double st = std::sin(comp);
        const tf2::Vector3 kxv = lat.cross(fwd);
        const double kdv = lat.dot(fwd);
        tf2::Vector3 d = fwd * ct + kxv * st + lat * (kdv * (1.0 - ct));

        if (std::abs(d.z()) < 1e-9) return std::nullopt;
        const double t = (ground_z_ - sz) / d.z();
        if (t < 0.0) return std::nullopt;
        return tf2::Vector3(sx + t * d.x(), sy + t * d.y(), ground_z_);
    }

    // ---- yaw ----
    std::optional<double> get_drone_yaw(const rclcpp::Time & stamp) {
        geometry_msgs::msg::TransformStamped tf_wd;
        try {
            tf_wd = tf_buffer_->lookupTransform(world_frame_, drone_frame_,
                                                stamp, tf2::durationFromSec(0.05));
        } catch (const tf2::TransformException & e) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                "No TF %s <- %s at %.9f: %s",
                world_frame_.c_str(), drone_frame_.c_str(),
                stamp.seconds(), e.what());
            return std::nullopt;
        }
        tf2::Quaternion q(tf_wd.transform.rotation.x, tf_wd.transform.rotation.y,
                          tf_wd.transform.rotation.z, tf_wd.transform.rotation.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
        return yaw;
    }

    // ---- build output cloud ----
    sensor_msgs::msg::PointCloud2::UniquePtr build_cloud(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr & src,
        const std::vector<double> & x,
        const std::vector<double> & y,
        const std::vector<double> & z,
        const std::vector<double> & intensity,
        const std::string & frame_id)
    {
        auto out = std::make_unique<sensor_msgs::msg::PointCloud2>();
        out->header = src->header;
        out->header.frame_id = frame_id;
        out->height = 1;
        out->width = static_cast<uint32_t>(x.size());
        out->is_bigendian = false;
        out->is_dense = true;

        sensor_msgs::PointCloud2Modifier mod(*out);
        mod.setPointCloud2Fields(
            4,
            "x",         1, sensor_msgs::msg::PointField::FLOAT32,
            "y",         1, sensor_msgs::msg::PointField::FLOAT32,
            "z",         1, sensor_msgs::msg::PointField::FLOAT32,
            "intensity", 1, sensor_msgs::msg::PointField::FLOAT32);
        mod.resize(x.size());

        sensor_msgs::PointCloud2Iterator<float> ix(*out, "x");
        sensor_msgs::PointCloud2Iterator<float> iy(*out, "y");
        sensor_msgs::PointCloud2Iterator<float> iz(*out, "z");
        sensor_msgs::PointCloud2Iterator<float> ii(*out, "intensity");
        for (size_t k = 0; k < x.size(); ++k, ++ix, ++iy, ++iz, ++ii) {
            *ix = static_cast<float>(x[k]);
            *iy = static_cast<float>(y[k]);
            *iz = static_cast<float>(z[k]);
            *ii = static_cast<float>(intensity[k]);
        }
        return out;
    }

    // ---- write PLY ----
    int write_ply_binary(const fs::path & path,
                         const std::vector<double> & x,
                         const std::vector<double> & y,
                         const std::vector<double> & z,
                         const std::vector<double> & intensity)
    {
        const size_t n = x.size();
        if (n == 0) return 0;

        std::ofstream fh(path, std::ios::binary);
        if (!fh) return -1;

        const std::string header =
            "ply\n"
            "format binary_little_endian 1.0\n"
            "comment generated by pc_transform_node\n"
            "element vertex " + std::to_string(n) + "\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float intensity\n"
            "end_header\n";
        fh.write(header.data(), static_cast<std::streamsize>(header.size()));

        std::vector<float> buf(4 * n);
        for (size_t i = 0; i < n; ++i) {
            buf[4 * i + 0] = static_cast<float>(x[i]);
            buf[4 * i + 1] = static_cast<float>(y[i]);
            buf[4 * i + 2] = static_cast<float>(z[i]);
            buf[4 * i + 3] = static_cast<float>(intensity[i]);
        }
        fh.write(reinterpret_cast<const char *>(buf.data()),
                 static_cast<std::streamsize>(buf.size() * sizeof(float)));
        return static_cast<int>(n);
    }

    // ---- save cloud ----
    void save_cloud(const rclcpp::Time & stamp,
                    const std::vector<double> & x,
                    const std::vector<double> & y,
                    const std::vector<double> & z,
                    const std::vector<double> & intensity)
    {
        char fname[160];
        std::snprintf(fname, sizeof(fname), "%s%010ld_%09ld.ply",
                      filename_prefix_.c_str(),
                      static_cast<long>(stamp.seconds()),
                      static_cast<long>(stamp.nanoseconds() % 1000000000LL));
        fs::path fpath = fs::path(output_dir_) / fname;

        std::vector<double> xf, yf, zf, inf;
        xf.reserve(x.size()); yf.reserve(x.size());
        zf.reserve(x.size()); inf.reserve(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            const double r = std::sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
            if (max_range_ > 0.0 && r > max_range_) continue;
            if (min_range_ > 0.0 && r < min_range_) continue;
            if (intensity[i] < min_intensity_ || intensity[i] > max_intensity_) continue;
            xf.push_back(x[i]); yf.push_back(y[i]);
            zf.push_back(z[i]); inf.push_back(intensity[i]);
        }
        if (xf.empty()) return;

        int n_written = write_ply_binary(fpath, xf, yf, zf, inf);
        if (n_written <= 0) {
            std::error_code ec;
            fs::remove(fpath, ec);
            return;
        }
        const size_t count = ++saved_count_;
        if (count % 50 == 0) {
            RCLCPP_INFO(get_logger(), "Saved %zu clouds (last: %s, %d points).",
                        count, fname, n_written);
        }
    }

    // ---- save pose ----
    void save_pose(const rclcpp::Time & stamp, double yaw,
                   double hx, double hy, double hz, size_t n_pts)
    {
        char base[160];
        std::snprintf(base, sizeof(base), "%s%010ld_%09ld",
                      filename_prefix_.c_str(),
                      static_cast<long>(stamp.seconds()),
                      static_cast<long>(stamp.nanoseconds() % 1000000000LL));

        fs::path pose_path = fs::path(output_dir_) / (std::string(base) + ".pose.txt");
        {
            std::ofstream fh(pose_path);
            if (fh) {
                char line[256];
                std::snprintf(line, sizeof(line), "# yaw_rad %.9e\n", yaw);
                fh << line;
                std::snprintf(line, sizeof(line), "# origin %.9e %.9e %.9e\n", hx, hy, hz);
                fh << line;
                const double c = std::cos(-yaw);
                const double s = std::sin(-yaw);
                std::snprintf(line, sizeof(line),
                    "%.9e %.9e 0.000000000e+00 0.000000000e+00\n", c, -s);
                fh << line;
                std::snprintf(line, sizeof(line),
                    "%.9e %.9e 0.000000000e+00 0.000000000e+00\n", s, c);
                fh << line;
                fh << "0.000000000e+00 0.000000000e+00 1.000000000e+00 0.000000000e+00\n";
            }
        }

        fs::path csv_path = fs::path(output_dir_) / "poses.csv";
        const bool need_header = !fs::exists(csv_path);
        std::ofstream fh(csv_path, std::ios::app);
        if (fh) {
            if (need_header) fh << "sec,nsec,yaw_rad,hx,hy,hz,n_points\n";
            char line[256];
            std::snprintf(line, sizeof(line),
                "%ld,%09ld,%.9e,%.9e,%.9e,%.9e,%zu\n",
                static_cast<long>(stamp.seconds()),
                static_cast<long>(stamp.nanoseconds() % 1000000000LL),
                yaw, hx, hy, hz, n_pts);
            fh << line;
        }

        const size_t count = ++pose_count_;
        if (count % 50 == 0) {
            RCLCPP_INFO(get_logger(), "Saved %zu poses (last: %s, yaw=%.2f deg).",
                        count, base, yaw * 180.0 / M_PI);
        }
    }

    // ---- members ----
    std::string world_frame_, lidar_frame_, drone_frame_;
    double ground_z_ = 0.0;
    std::array<double, 3> forward_axis_{1.0, 0.0, 0.0};

    std::string origin_mode_ = "centroid";
    double ground_band_ = 0.5;
    double mad_k_ = 3.0;
    int min_ground_points_ = 20;

    std::string tilt_axis_ = "y";
    double roll_scale_ = 1.0 / 3.0;
    double roll_tilt_sign_ = 1.0;

    double min_range_ = 0.0, max_range_ = 500.0;
    double min_intensity_ = -1e9, max_intensity_ = 1e9;

    bool save_clouds_ = true;
    bool save_poses_ = true;
    std::string output_dir_ = "/tmp/dnn_dataset";
    std::string filename_prefix_ = "cloud_";

    int subscription_depth_ = 1000;
    int writer_queue_size_ = 100000;
    bool drop_when_full_ = false;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
    rclcpp::Publisher<pc_transform_cpp::msg::AlignedCloudWithPose>::SharedPtr aligned_pub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;

    std::queue<sensor_msgs::msg::PointCloud2::ConstSharedPtr> queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> writer_stop_{false};
    std::thread writer_thread_;

    std::atomic<size_t> saved_count_{0};
    std::atomic<size_t> pose_count_{0};
    std::atomic<size_t> dropped_count_{0};
};

// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<AlignCloudNode>();
        rclcpp::spin(node);
    } catch (const std::exception & e) {
        RCLCPP_FATAL(rclcpp::get_logger("pc_transform_node"),
                     "Fatal: %s", e.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
