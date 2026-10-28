// Multi-threaded pillar voxelizer exposed to Python with pybind11.
//
// Matches PillarVFE.voxelize in model/static_pillars.py: pillars are returned
// sorted by key (batch, col, row) and each keeps the first `max_points`
// points in input order; later points are dropped.
//
// The result is deterministic for any thread count. Points are split into
// contiguous chunks, per-chunk histograms give every point its exact rank in
// its pillar (a stable counting sort), and no two threads write the same slot.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

template <typename F>
void parallel_for(int num_threads, F&& fn) {
  if (num_threads <= 1) {
    fn(0);
    return;
  }
  std::vector<std::thread> workers;
  workers.reserve(num_threads - 1);
  for (int t = 1; t < num_threads; ++t) workers.emplace_back(fn, t);
  fn(0);
  for (auto& w : workers) w.join();
}

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<int32_t>;

py::tuple voxelize(FloatArray points, std::array<float, 2> range_min,
                   std::array<float, 2> voxel_size, std::array<int, 2> grid_size,
                   int max_points, int num_threads) {
  if (points.ndim() != 2 || points.shape(1) != 4)
    throw std::invalid_argument("points must have shape (N, 4): batch, x, y, z");
  if (max_points <= 0) throw std::invalid_argument("max_points must be > 0");
  if (grid_size[0] <= 0 || grid_size[1] <= 0)
    throw std::invalid_argument("grid_size must be positive");
  if (voxel_size[0] <= 0.f || voxel_size[1] <= 0.f)
    throw std::invalid_argument("voxel_size must be positive");

  const int64_t n = points.shape(0);
  const float* pts = points.data();
  const int64_t nx = grid_size[0], ny = grid_size[1];
  const int64_t plane = nx * ny;

  // Do not spawn threads that would only get a handful of points.
  const int64_t min_chunk = 2048;
  const int threads = static_cast<int>(std::max<int64_t>(
      1, std::min<int64_t>(std::max(num_threads, 1), n / min_chunk)));
  std::vector<int64_t> bounds(threads + 1);
  for (int t = 0; t <= threads; ++t) bounds[t] = n * t / threads;

  std::vector<int64_t> keys(n);
  std::vector<int32_t> hist;
  std::vector<int32_t> pillar_of_key;
  std::vector<int64_t> pillar_keys;
  std::vector<int32_t> pillar_counts;
  int64_t num_keys = 0;

  {
    py::gil_scoped_release release;

    // Pass 1: pillar key per point (-1 if outside the grid / non-finite).
    std::vector<int64_t> max_batch(threads, -1);
    parallel_for(threads, [&](int t) {
      int64_t mb = -1;
      for (int64_t i = bounds[t]; i < bounds[t + 1]; ++i) {
        const float* p = pts + 4 * i;
        keys[i] = -1;
        if (!std::isfinite(p[0]) || !std::isfinite(p[1]) ||
            !std::isfinite(p[2]) || !std::isfinite(p[3]))
          continue;
        // Same float32 arithmetic as the torch implementation.
        const float fx = std::floor((p[1] - range_min[0]) / voxel_size[0]);
        const float fy = std::floor((p[2] - range_min[1]) / voxel_size[1]);
        if (!(fx >= 0.f && fx < static_cast<float>(nx) && fy >= 0.f &&
              fy < static_cast<float>(ny)))
          continue;
        const int64_t b = static_cast<int64_t>(p[0]);
        if (b < 0) continue;
        keys[i] = b * plane + static_cast<int64_t>(fx) * ny + static_cast<int64_t>(fy);
        mb = std::max(mb, b);
      }
      max_batch[t] = mb;
    });
    const int64_t batch = *std::max_element(max_batch.begin(), max_batch.end()) + 1;
    num_keys = batch * plane;

    if (batch > 0) {
      // Per-thread histograms over the dense key space.
      hist.assign(static_cast<size_t>(threads) * num_keys, 0);
      parallel_for(threads, [&](int t) {
        int32_t* h = hist.data() + static_cast<size_t>(t) * num_keys;
        for (int64_t i = bounds[t]; i < bounds[t + 1]; ++i)
          if (keys[i] >= 0) ++h[keys[i]];
      });

      // Turn counts into per-thread start ranks and number the occupied
      // pillars in key order.
      pillar_of_key.assign(num_keys, -1);
      for (int64_t k = 0; k < num_keys; ++k) {
        int32_t total = 0;
        for (int t = 0; t < threads; ++t) {
          int32_t& c = hist[static_cast<size_t>(t) * num_keys + k];
          const int32_t here = c;
          c = total;
          total += here;
        }
        if (total > 0) {
          pillar_of_key[k] = static_cast<int32_t>(pillar_keys.size());
          pillar_keys.push_back(k);
          pillar_counts.push_back(total);
        }
      }
    }
  }

  const int64_t v = static_cast<int64_t>(pillar_keys.size());
  py::array_t<float> voxels(std::vector<py::ssize_t>{v, max_points, 3});
  IntArray counts(std::vector<py::ssize_t>{v});
  IntArray coords(std::vector<py::ssize_t>{v, 4});
  float* vox = voxels.mutable_data();
  int32_t* cnt = counts.mutable_data();
  int32_t* crd = coords.mutable_data();

  {
    py::gil_scoped_release release;

    // Zero the padding and fill counts/coords, split by pillar.
    parallel_for(threads, [&](int t) {
      const int64_t lo = v * t / threads, hi = v * (t + 1) / threads;
      std::memset(vox + lo * max_points * 3, 0,
                  sizeof(float) * (hi - lo) * max_points * 3);
      for (int64_t p = lo; p < hi; ++p) {
        const int64_t k = pillar_keys[p];
        cnt[p] = std::min<int32_t>(pillar_counts[p], max_points);
        crd[4 * p + 0] = static_cast<int32_t>(k / plane);
        crd[4 * p + 1] = 0;
        crd[4 * p + 2] = static_cast<int32_t>(k % ny);
        crd[4 * p + 3] = static_cast<int32_t>((k % plane) / ny);
      }
    });

    // Pass 2 (after the zeroing above has finished): write the points.
    parallel_for(threads, [&](int t) {
      int32_t* cursor = hist.data() + static_cast<size_t>(t) * num_keys;
      for (int64_t i = bounds[t]; i < bounds[t + 1]; ++i) {
        const int64_t k = keys[i];
        if (k < 0) continue;
        const int32_t rank = cursor[k]++;
        if (rank >= max_points) continue;
        float* dst =
            vox + (static_cast<int64_t>(pillar_of_key[k]) * max_points + rank) * 3;
        const float* p = pts + 4 * i;
        dst[0] = p[1];
        dst[1] = p[2];
        dst[2] = p[3];
      }
    });
  }

  return py::make_tuple(voxels, counts, coords);
}

using IntArray64 = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

// Builds the PFN input from voxelizer output, laid out channel-first and
// padded to a fixed pillar count: (F, max_pillars, P), NCHW without the batch
// dim. Channels: [x y z] (if use_abs_xyz), cluster offset (3), centre offset
// (3), [distance] (if with_distance). Padded points/pillars are all zero, as
// in PillarVFE.forward. Returns (features, num_valid_pillars); pillars beyond
// max_pillars are dropped.
py::tuple decorate(FloatArray voxels, IntArray64 num_points, IntArray64 coords,
                   std::array<float, 3> range_min,
                   std::array<float, 3> voxel_size, int max_pillars,
                   bool use_abs_xyz, bool with_distance, int num_threads) {
  if (voxels.ndim() != 3 || voxels.shape(2) != 3)
    throw std::invalid_argument("voxels must have shape (V, P, 3)");
  if (num_points.ndim() != 1 || num_points.shape(0) != voxels.shape(0))
    throw std::invalid_argument("num_points must have shape (V,)");
  if (coords.ndim() != 2 || coords.shape(0) != voxels.shape(0) ||
      coords.shape(1) != 4)
    throw std::invalid_argument("coords must have shape (V, 4)");
  if (max_pillars <= 0) throw std::invalid_argument("max_pillars must be > 0");

  const int64_t v = voxels.shape(0);
  const int64_t P = voxels.shape(1);
  const int64_t n = std::min<int64_t>(v, max_pillars);
  const int channels = (use_abs_xyz ? 3 : 0) + 6 + (with_distance ? 1 : 0);

  py::array_t<float> out(std::vector<py::ssize_t>{channels, max_pillars, P});
  float* o = out.mutable_data();
  const float* vox = voxels.data();
  const int32_t* np_ = num_points.data();
  const int32_t* crd = coords.data();

  const float x_off = voxel_size[0] / 2 + range_min[0];
  const float y_off = voxel_size[1] / 2 + range_min[1];
  const float z_off = voxel_size[2] / 2 + range_min[2];
  const int64_t plane = static_cast<int64_t>(max_pillars) * P;

  const int threads = static_cast<int>(std::max<int64_t>(
      1, std::min<int64_t>(std::max(num_threads, 1), n / 256)));
  {
    py::gil_scoped_release release;
    parallel_for(threads, [&](int t) {
      const int64_t lo = n * t / threads, hi = n * (t + 1) / threads;
      for (int c = 0; c < channels; ++c)
        std::memset(o + c * plane + lo * P, 0, sizeof(float) * (hi - lo) * P);
      // Padding pillars (index >= n) are zeroed by the last thread.
      if (t == threads - 1 && n < max_pillars)
        for (int c = 0; c < channels; ++c)
          std::memset(o + c * plane + n * P, 0, sizeof(float) * (max_pillars - n) * P);

      for (int64_t p = lo; p < hi; ++p) {
        const int32_t cnt = std::min<int64_t>(np_[p], P);
        if (cnt <= 0) continue;
        const float* pv = vox + p * P * 3;
        float sx = 0, sy = 0, sz = 0;
        for (int s = 0; s < cnt; ++s) {
          sx += pv[3 * s]; sy += pv[3 * s + 1]; sz += pv[3 * s + 2];
        }
        const float inv = static_cast<float>(cnt);
        const float mx = sx / inv, my = sy / inv, mz = sz / inv;
        const float cx = static_cast<float>(crd[4 * p + 3]) * voxel_size[0] + x_off;
        const float cy = static_cast<float>(crd[4 * p + 2]) * voxel_size[1] + y_off;
        for (int s = 0; s < cnt; ++s) {
          const float x = pv[3 * s], y = pv[3 * s + 1], z = pv[3 * s + 2];
          float* dst = o + p * P + s;
          int c = 0;
          if (use_abs_xyz) {
            dst[(c++) * plane] = x; dst[(c++) * plane] = y; dst[(c++) * plane] = z;
          }
          dst[(c++) * plane] = x - mx;
          dst[(c++) * plane] = y - my;
          dst[(c++) * plane] = z - mz;
          dst[(c++) * plane] = x - cx;
          dst[(c++) * plane] = y - cy;
          dst[(c++) * plane] = z - z_off;
          if (with_distance) dst[c * plane] = std::sqrt(x * x + y * y + z * z);
        }
      }
    });
  }
  return py::make_tuple(out, static_cast<int>(n));
}

}  // namespace

PYBIND11_MODULE(_voxelizer, m) {
  m.doc() = "Multi-threaded pillar voxelizer";
  m.def("voxelize", &voxelize, py::arg("points"), py::arg("range_min"),
        py::arg("voxel_size"), py::arg("grid_size"), py::arg("max_points"),
        py::arg("num_threads") = 4,
        "points (N,4) = (batch, x, y, z) -> (voxels (V,P,3) float32, "
        "num_points (V,) int32, coords (V,4) int32 = (batch, 0, row, col))");
  m.def("decorate", &decorate, py::arg("voxels"), py::arg("num_points"),
        py::arg("coords"), py::arg("range_min"), py::arg("voxel_size"),
        py::arg("max_pillars"), py::arg("use_abs_xyz") = true,
        py::arg("with_distance") = false, py::arg("num_threads") = 4,
        "-> (features (F, max_pillars, P) float32, num_valid_pillars)");
}
