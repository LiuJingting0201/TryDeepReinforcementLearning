# -*- coding: utf-8 -*-
import numpy as np
import pybullet as p

# ===== 统一的场景几何配置 (修复版本) =====
TABLE_TOP_Z = 0.625  # PyBullet内置table桌面高度近似值
TABLE_POS = [0.5, 0, 0]         # Table position (table center) - 与environment_setup.py一致
ROBOT_BASE_POS = [0, 0, TABLE_TOP_Z]  # Robot base mounted on the table surface
OBJECT_SPAWN_CENTER = [0.60, 0, TABLE_TOP_Z]  # 物体生成中心 - 与environment_setup.py一致

# 工作区域定义 (桌面上的安全抓取区域)
WORKSPACE_X_RANGE = [0.45, 0.75]  # X方向范围 (调整到桌子周围)
WORKSPACE_Y_RANGE = [-0.15, 0.15] # Y方向范围

# 相机配置
CAMERA_TARGET = OBJECT_SPAWN_CENTER  # 相机目标点 - 对准物体生成中心，与environment_setup.py一致
CAMERA_DISTANCE = 1.2  # 相机距离 - 与perception.py一致
CAMERA_PARAMS = {
    'width': 224,
    'height': 224, 
    'fov': 60.0,
    'near': 0.1,
    'far': 2.0
}

def set_topdown_camera(target=CAMERA_TARGET, distance=CAMERA_DISTANCE, 
                       yaw=0.0, pitch=-89.0, **camera_params):
    """设置近似顶视相机，返回 (W,H, view, proj)。"""
    params = {**CAMERA_PARAMS, **camera_params}  # 合并默认参数和自定义参数
    cx, cy, cz = target
    eye = [cx, cy, cz + distance]    # 相机在桌面正上方
    up = [0, 1, 0]                   # 顶视时 up 方向取 Y 更稳定
    view = p.computeViewMatrix(eye, target, up)
    proj = p.computeProjectionMatrixFOV(params['fov'], params['width']/float(params['height']), 
                                        params['near'], params['far'])
    return params['width'], params['height'], view, proj

def get_rgb_depth(width, height, view, proj):
    """获取 RGB & 深度（float32），RGB形状(H,W,3)。"""
    img = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER)
    rgb = np.asarray(img[2], dtype=np.uint8)[..., :3]
    depth = np.asarray(img[3], dtype=np.float32)
    return rgb, depth

def _mat_from_list(m): return np.array(m, dtype=np.float32).reshape(4,4)
def _invert_mat4(m):   return np.linalg.inv(m)

def pixel_to_world_on_plane(u, v, width, height, view, proj, plane_z=TABLE_TOP_Z):
    """像素(u,v)反投影到世界坐标，并与 z=plane_z 平面求交，返回 xyz。"""
    x_ndc = (2.0 * (u + 0.5) / width) - 1.0
    y_ndc = 1.0 - (2.0 * (v + 0.5) / height)
    pts_clip = np.array([[x_ndc, y_ndc, -1.0, 1.0],
                         [x_ndc, y_ndc,  1.0, 1.0]], dtype=np.float32)
    inv_vp = _invert_mat4(_mat_from_list(proj) @ _mat_from_list(view))
    pts_world = []
    for pc in pts_clip:
        pw = inv_vp @ pc
        pw = pw / pw[3]
        pts_world.append(pw[:3])
    p0, p1 = np.array(pts_world[0]), np.array(pts_world[1])
    dirv = p1 - p0
    if abs(dirv[2]) < 1e-6: return None
    t = (plane_z - p0[2]) / dirv[2]
    return p0 + t * dirv

def move_ee_via_ik(robot_id, ee_link, pos, orn=None, steps=240):
    """用IK移动末端到 pos, orn。"""
    if orn is None:
        orn = p.getQuaternionFromEuler([0, np.pi, 0])  # 工具Z朝下
    joints = p.calculateInverseKinematics(robot_id, ee_link, pos, orn, maxNumIterations=200)
    # 只控制前7个关节（机械臂关节）
    idxs = list(range(7))
    p.setJointMotorControlArray(robot_id, idxs, p.POSITION_CONTROL, targetPositions=joints[:7])
    for _ in range(steps): p.stepSimulation()

def control_gripper(robot_id, open_width=0.08, steps=120):
    """Franka手爪开合，关节9/10为手指关节。"""
    # 根据URDF，夹爪范围是0.000-0.040米，open_width需要调整
    max_width = 0.04  # 最大开口宽度
    target_width = min(open_width, max_width)  # 限制在有效范围内
    
    # 每个手指的位置是总宽度的一半
    finger_pos = target_width / 2.0
    
    print(f"设置夹爪宽度: {target_width:.3f}m (每个手指: {finger_pos:.3f}m)")
    
    p.setJointMotorControl2(robot_id, 9,  p.POSITION_CONTROL, 
                           targetPosition=finger_pos, force=20, maxVelocity=0.1)
    p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, 
                           targetPosition=finger_pos, force=20, maxVelocity=0.1)
    
    for _ in range(steps): 
        p.stepSimulation()
    
    # 返回实际手指位置用于验证
    finger1_pos = p.getJointState(robot_id, 9)[0]
    finger2_pos = p.getJointState(robot_id, 10)[0]
    actual_width = finger1_pos + finger2_pos
    print(f"实际夹爪宽度: {actual_width:.3f}m (手指1: {finger1_pos:.3f}m, 手指2: {finger2_pos:.3f}m)")
    
    return actual_width

def setup_scene(add_objects=True, n_objects=2):
    """统一的场景设置函数，确保所有位置一致。"""
    import pybullet_data
    
    # 设置重力
    p.setGravity(0, 0, -9.8)
    
    # 设置PyBullet数据路径
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 加载地面
    plane_id = p.loadURDF("plane.urdf")
    
    # 加载桌子 (使用统一位置)
    table_id = p.loadURDF("table/table.urdf", TABLE_POS, useFixedBase=True)
    
    # 加载机械臂 (使用统一位置)
    robot_id = p.loadURDF("franka_panda/panda.urdf", ROBOT_BASE_POS, useFixedBase=True)
    
    obj_ids = []
    if add_objects:
        # 使用与environment_setup.py相同的物体创建逻辑
        obj_ids = create_objects_like_environment_setup(n_objects)
    
    # 让物体稳定下来 - 增加仿真时间
    for _ in range(1000):  # 增加到1000步
        p.stepSimulation()
    
    return robot_id, table_id, obj_ids

def is_position_in_workspace(x, y):
    """检查位置是否在工作区域内。"""
    return (WORKSPACE_X_RANGE[0] <= x <= WORKSPACE_X_RANGE[1] and 
            WORKSPACE_Y_RANGE[0] <= y <= WORKSPACE_Y_RANGE[1])

def create_objects_like_environment_setup(num_objects=2):
    """Creates objects using the same logic as environment_setup.py's create_better_objects."""
    # Franka Panda gripper constraints
    MAX_GRIPPER_OPENING = 0.08  # 8cm maximum
    SAFE_OBJECT_WIDTH = 0.035   # 3.5cm - safe size for reliable grasping
    
    object_ids = []
    object_positions = []  # Track positions to maintain distance
    MIN_OBJECT_DISTANCE = 0.06  # Minimum distance between objects
    MAX_SPAWN_ATTEMPTS = 20
    
    # Limit the number of objects
    num_objects = min(num_objects, 5)  # Reasonable limit
    
    for i in range(num_objects):
        placed = False
        attempts = 0
        current_min_distance = MIN_OBJECT_DISTANCE
        
        while not placed and attempts < MAX_SPAWN_ATTEMPTS:
            attempts += 1
            
            # Generate random position in workspace
            x_pos = OBJECT_SPAWN_CENTER[0] + np.random.uniform(-0.15, 0.15)  # 0.45-0.75m
            y_pos = OBJECT_SPAWN_CENTER[1] + np.random.uniform(-0.25, 0.25)  # -0.25~0.25m
            candidate_pos = [x_pos, y_pos]
            
            # Check distance to existing objects
            too_close = False
            if len(object_positions) > 0:
                for existing_pos in object_positions:
                    distance = np.sqrt((candidate_pos[0] - existing_pos[0])**2 + 
                                     (candidate_pos[1] - existing_pos[1])**2)
                    if distance < current_min_distance:
                        too_close = True
                        break
            
            if not too_close:
                placed = True
                
            # Gradually reduce distance requirement if struggling to place
            elif attempts > MAX_SPAWN_ATTEMPTS // 2:
                current_min_distance = MIN_OBJECT_DISTANCE * 0.8
        
        if placed:
            object_positions.append(candidate_pos)
            
            shape_type = np.random.choice([p.GEOM_BOX, p.GEOM_CYLINDER])
            color = [np.random.random(), np.random.random(), np.random.random(), 1]
            
            if shape_type == p.GEOM_BOX:
                half_extents = [
                    np.random.uniform(0.02, SAFE_OBJECT_WIDTH/2),  # 1.5-1.75cm
                    np.random.uniform(0.02, SAFE_OBJECT_WIDTH/2),  # 1.5-1.75cm
                    np.random.uniform(0.02, 0.025)                 # 高度: 1.5-2.5cm
                ]
                shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
                visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
                z_pos = TABLE_TOP_Z + half_extents[2]
            elif shape_type == p.GEOM_CYLINDER:
                radius = np.random.uniform(0.008, SAFE_OBJECT_WIDTH/2)  # 0.8-1.75cm
                height = np.random.uniform(0.02, 0.04)                  # 高度: 2-4cm
                shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
                visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
                z_pos = TABLE_TOP_Z + height / 2
            else: # p.GEOM_SPHERE
                radius = np.random.uniform(0.008, SAFE_OBJECT_WIDTH/2)  # 0.8-1.75cm
                shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
                visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
                z_pos = TABLE_TOP_Z + radius

            body = p.createMultiBody(
                baseMass=np.random.uniform(0.05, 0.2),  # 较轻的物体更容易抓取
                baseCollisionShapeIndex=shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[x_pos, y_pos, z_pos + 0.005],  # Slightly above table
                baseOrientation=p.getQuaternionFromEuler([0, 0, np.random.uniform(0, 3.14)])
            )
            p.changeDynamics(body, -1, lateralFriction=1.5, restitution=0.1)
            object_ids.append(body)
            
        else:
            print(f"   ⚠️  Could not place object {i+1} after {MAX_SPAWN_ATTEMPTS} attempts")
    
    # Fallback: create at least one object if none were placed
    if len(object_ids) == 0:
        x_pos = OBJECT_SPAWN_CENTER[0]
        y_pos = OBJECT_SPAWN_CENTER[1]
        
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], 
                                         rgbaColor=[1, 0, 0, 1])  # Red fallback object
        body = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[x_pos, y_pos, TABLE_TOP_Z + 0.025]
        )
        p.changeDynamics(body, -1, lateralFriction=1.5, restitution=0.1)
        object_ids.append(body)
        print(f"   🆘 Created fallback object at center")
    
    return object_ids
