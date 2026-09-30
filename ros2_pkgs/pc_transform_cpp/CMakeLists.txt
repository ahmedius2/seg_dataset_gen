cmake_minimum_required(VERSION 3.20)
project(pc_transform_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AlignedCloudWithPose.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

add_executable(pc_transform_node src/pc_transform_node.cpp)
target_link_libraries(pc_transform_node
  rclcpp::rclcpp
  ${sensor_msgs_TARGETS}
  tf2::tf2
  tf2_ros::tf2_ros
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(pc_transform_node "${cpp_typesupport_target}")

install(TARGETS pc_transform_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch/ and config/ if you create them. Comment out any line
# whose directory does not exist, otherwise CMake fails.
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
# install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_export_dependencies(rosidl_default_runtime)
ament_package()
