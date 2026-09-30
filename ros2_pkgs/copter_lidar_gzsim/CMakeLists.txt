cmake_minimum_required(VERSION 3.16)
project(your_pkg)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(pc_transform_node src/pc_transform_node.cpp)
target_link_libraries(pc_transform_node
  rclcpp::rclcpp
  ${sensor_msgs_TARGETS}
  tf2::tf2
  tf2_ros::tf2_ros
  Eigen3::Eigen
)
target_include_directories(pc_transform_node PRIVATE
  ${EIGEN3_INCLUDE_DIR}
)

install(TARGETS pc_transform_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
