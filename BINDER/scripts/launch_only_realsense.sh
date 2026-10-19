ros2 launch stretch_ros2_bridge rs_launch.py \
  align_depth.enable:=true \
  camera_name:=camera \
  camera_namespace:="/" \
  device_type:=d435i \
  rgb_camera.color_profile:="640x480x30" \
  depth_module.depth_profile:="640x480x30" \
  depth_module.infra_profile:="640x480x30" \
  enable_gyro:=true enable_accel:=true gyro_fps:=200 accel_fps:=100
