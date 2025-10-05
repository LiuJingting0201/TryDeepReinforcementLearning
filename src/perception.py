# -*- coding: utf-8 -*-
import pybullet as p
import numpy as np
import cv2

# Constants from geom.py, to be centralized later if needed
TABLE_TOP_Z = 0.625  # 与environment_setup.py保持一致
CAMERA_TARGET = [0.60, 0, TABLE_TOP_Z]  # 对准物体生成中心（更靠前）- 与environment_setup.py一致
CAMERA_DISTANCE = 1.2
CAMERA_PARAMS = {
    'width': 224,
    'height': 224, 
    'fov': 60.0,
    'near': 0.1,
    'far': 2.0
}

def set_topdown_camera(target=CAMERA_TARGET, distance=CAMERA_DISTANCE, 
                       yaw=90.0, pitch=-89.0, **camera_params):
    """Sets up a top-down camera and returns its parameters."""
    params = {**CAMERA_PARAMS, **camera_params}
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=distance,
        yaw=yaw,
        pitch=pitch,
        roll=0,
        upAxisIndex=2
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        params['fov'], 
        params['width']/float(params['height']), 
        params['near'], 
        params['far']
    )
    return params['width'], params['height'], view_matrix, proj_matrix

def get_rgb_depth(width, height, view, proj):
    """Captures RGB and depth images from the simulation."""
    img_arr = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER)
    rgb = np.asarray(img_arr[2], dtype=np.uint8).reshape(height, width, 4)[..., :3]
    depth = np.asarray(img_arr[3], dtype=np.float32).reshape(height, width)
    return rgb, depth

def get_rgb_depth_segmentation(width, height, view, proj, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX):
    """Captures RGB, depth, and segmentation mask from the simulation.
    
    Args:
        width, height: Image dimensions
        view, proj: Camera matrices
        flags: Segmentation flags (default: object+link segmentation)
    
    Returns:
        rgb: RGB image (height, width, 3)
        depth: Depth buffer (height, width)
        seg: Segmentation mask (height, width) - each pixel contains object ID
    """
    img_arr = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER, flags=flags)
    rgb = np.asarray(img_arr[2], dtype=np.uint8).reshape(height, width, 4)[..., :3]
    depth = np.asarray(img_arr[3], dtype=np.float32).reshape(height, width)
    seg = np.asarray(img_arr[4], dtype=np.int32).reshape(height, width)
    return rgb, depth, seg

def find_best_grasp_pixel(rgb, depth):
    """Finds the best pixel to grasp based on depth and color information."""
    height, width = rgb.shape[:2]
    
    # Use depth to find foreground objects
    foreground_mask = (depth > 0) & (depth < depth.max() * 0.95)
    
    if np.sum(foreground_mask) < 100: # If not enough points, use color
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        foreground_mask = gray < 200

    kernel = np.ones((5,5), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return width // 2, height // 2 # Default to center

    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return width // 2, height // 2

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    return cx, cy

def pixel_to_world(u, v, depth_buffer, view_matrix, proj_matrix):
    """
    将像素坐标和深度缓冲区值转换为世界坐标
    使用俯视相机的正确投影公式
    
    关键理解：
    - pitch=-89°的俯视相机，深度值是沿相机光轴的距离（几乎是垂直距离）
    - 视野范围应该基于**该点的实际深度**，而不是相机到目标的距离
    
    Args:
        u, v: 像素坐标
        depth_buffer: PyBullet深度缓冲区值 [0, 1]
        view_matrix, proj_matrix: 相机矩阵
    
    Returns:
        世界坐标 [x, y, z]
    """
    width, height = CAMERA_PARAMS['width'], CAMERA_PARAMS['height']
    near, far = CAMERA_PARAMS['near'], CAMERA_PARAMS['far']
    fov = CAMERA_PARAMS['fov']
    
    # 1. PyBullet深度缓冲区 → 实际距离（沿相机光轴）
    depth_real = far * near / (far - (far - near) * depth_buffer)
    
    # 2. 计算相机在该深度处的视野范围
    # 关键：视野范围取决于深度，不是固定的CAMERA_DISTANCE！
    fov_rad = np.radians(fov)
    view_width_at_depth = 2.0 * depth_real * np.tan(fov_rad / 2.0)
    view_height_at_depth = view_width_at_depth  # 正方形视野
    
    # 3. 像素坐标归一化到 [-0.5, 0.5]
    u_norm = (u / width) - 0.5
    v_norm = (v / height) - 0.5
    
    # 4. 计算XY世界坐标
    # 相机俯视，yaw=90度（相机绕Z轴旋转90°）：
    # - 图像左右(u) 对应世界 X轴（u小=左=-X，u大=右=+X）
    # - 图像上下(v) 对应世界 Y轴（v小=上=-Y，v大=下=+Y）
    x_world = CAMERA_TARGET[0] + u_norm * view_width_at_depth   # u → X
    y_world = CAMERA_TARGET[1] + v_norm * view_height_at_depth  # v → Y
    
    # 5. 计算Z坐标
    # 相机高度 = 目标高度 + 相机距离
    camera_height = CAMERA_TARGET[2] + CAMERA_DISTANCE
    z_world = camera_height - depth_real
    
    return np.array([x_world, y_world, z_world])
