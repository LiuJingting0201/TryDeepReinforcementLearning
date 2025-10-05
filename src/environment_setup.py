# -*- coding: utf-8 -*-
import pybullet as p
import pybullet_data
import numpy as np

TABLE_TOP_Z = 0.625  # Table surface height (实际桌面高度)
TABLE_POS = [0.5, 0, 0]  # Table position (table center)
ROBOT_BASE_POS = [0, 0, TABLE_TOP_Z]  # Robot base mounted on the table surface

# 物体生成区域：在桌面上，机械臂前方
# 相机俯视目标也应该对准这个区域
# Franka Panda俯视抓取的理想范围：X在0.4-0.8米（距离基座更远，避免奇异点）
OBJECT_SPAWN_CENTER = [0.60, 0, TABLE_TOP_Z]  # 往前移动到0.6米！

# 物体间距参数 - 减小最小距离要求
MIN_OBJECT_DISTANCE = 0.06  # Reduced from 0.10 to 0.06 (6cm minimum distance)
MAX_SPAWN_ATTEMPTS = 20     # Reduced from 50 to 20 for faster placement

def setup_environment(num_objects=3):
    """Sets up the simulation environment with a robot, table, and objects."""
    print("🏗️  Setting up the environment...")
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./240.)
    
    # Load plane and table
    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", basePosition=TABLE_POS, useFixedBase=True)
    
    # Load robot
    robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=ROBOT_BASE_POS, useFixedBase=True)
    
    # Create objects
    object_ids = create_better_objects(num_objects)
    
    print("⏳ Waiting for objects to settle...")
    for _ in range(100):
        p.stepSimulation()
        
    print(f"✅ Environment setup complete. Robot ID: {robot_id}, Object IDs: {object_ids}")
    return robot_id, object_ids

def update_object_states(object_ids):
    """Check which objects are still on the table and remove IDs of fallen/moved objects."""
    active_objects = []
    removed_objects = []
    
    for obj_id in object_ids:
        try:
            # ✨ 额外检查：确保物体ID仍然存在
            if obj_id <= 2:  # 跳过环境物体ID
                print(f"   ⚠️  跳过环境物体ID {obj_id}")
                continue
                
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            
            # ✨ 更严格的位置检查
            in_workspace = (
                pos[2] > TABLE_TOP_Z - 0.1 and  # Not fallen below table
                pos[2] < TABLE_TOP_Z + 0.5 and  # Not too high (carried away)
                abs(pos[0] - OBJECT_SPAWN_CENTER[0]) < 0.4 and  # Still in X range
                abs(pos[1] - OBJECT_SPAWN_CENTER[1]) < 0.4      # Still in Y range
            )
            
            if in_workspace:
                active_objects.append(obj_id)
            else:
                print(f"   🗑️  Object {obj_id} outside workspace (pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]) - will be removed")
                removed_objects.append(obj_id)
                
        except:
            # Object might have been removed from simulation
            print(f"   ❌ Object {obj_id} no longer exists in simulation")
            removed_objects.append(obj_id)
            continue
    
    # ✨ 修复：物理移除超出工作空间的物体（但保护环境物体）
    if removed_objects:
        print(f"   🧹 清理 {len(removed_objects)} 个超出工作空间的物体...")
        for obj_id in removed_objects:
            if obj_id <= 2:  # 保护环境物体
                print(f"      🛡️  保护环境物体 {obj_id}，不移除")
                continue
                
            try:
                p.removeBody(obj_id)
                print(f"      ✅ 移除物体 {obj_id}")
            except:
                print(f"      ⚠️  无法移除物体 {obj_id} (可能已被移除)")
    
    return active_objects

def cleanup_workspace():
    """清理工作空间中的所有动态物体"""
    print("   🧹 清理工作空间...")
    
    # 获取所有物体ID
    all_bodies = []
    for i in range(p.getNumBodies()):
        body_id = p.getBodyUniqueId(i)
        all_bodies.append(body_id)
    
    removed_count = 0
    for body_id in all_bodies:
        try:
            # 检查是否是动态物体（非机器人、非桌子、非地面）
            body_info = p.getBodyInfo(body_id)
            body_name = body_info[0].decode('utf-8') if body_info[0] else ""
            
            # ✨ 修复：更严格的环境物体保护
            # 跳过所有固定的环境物体
            protected_names = ['plane', 'table', 'panda', 'franka']
            if any(name in body_name.lower() for name in protected_names):
                continue
            
            # ✨ 额外检查：通过body ID范围保护环境物体
            # 通常前几个ID是环境物体（plane=0, table=1, robot=2）
            if body_id <= 2:
                print(f"      🛡️  保护环境物体 {body_id} ({body_name})")
                continue
            
            # 检查物体位置
            pos, _ = p.getBasePositionAndOrientation(body_id)
            
            # ✨ 更保守的清理范围：只移除明显超出工作区域的物体
            should_remove = (
                pos[2] < TABLE_TOP_Z - 0.3 or  # 掉到桌面下方30cm
                pos[2] > TABLE_TOP_Z + 1.5 or  # 飞到桌面上方1.5m
                abs(pos[0]) > 3.0 or           # X方向超出3m
                abs(pos[1]) > 3.0              # Y方向超出3m
            )
            
            if should_remove:
                p.removeBody(body_id)
                removed_count += 1
                print(f"      🗑️  移除远程物体 {body_id} at [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            else:
                print(f"      ✅ 保留物体 {body_id} ({body_name}) at [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                
        except Exception as e:
            print(f"      ⚠️  检查物体 {body_id} 时出错: {e}")
            continue
    
    if removed_count > 0:
        print(f"   ✅ 清理完成，移除了 {removed_count} 个远程物体")
    else:
        print(f"   ✅ 工作空间无需清理")

def reset_objects_after_grasp(object_ids, min_objects=2):
    """Reset/respawn objects if too few remain for continued training."""
    active_objects = update_object_states(object_ids)
    
    if len(active_objects) < min_objects:
        print(f"⚠️  Only {len(active_objects)} objects remaining, respawning new ones...")
        
        # ✨ 新增：完全清理工作空间
        cleanup_workspace()
        
        # 等待物理世界稳定
        for _ in range(30):
            p.stepSimulation()
        
        # Create new batch of objects
        new_objects = create_better_objects(num_objects=min_objects)
        
        # Let them settle
        print("⏳ Waiting for new objects to settle...")
        for _ in range(50):
            p.stepSimulation()
            
        return new_objects
    
    return active_objects

def create_better_objects(num_objects=5):
    """Creates objects with more stable physical properties and ensures minimum spacing.
    
    Object size is constrained to fit within the Franka Panda gripper opening.
    Max gripper opening: ~0.08m (8cm), so objects should be < 0.06m (6cm) to be graspable.
    Objects are placed with minimum distance to avoid interference during grasping.
    """
    # Franka Panda gripper constraints
    MAX_GRIPPER_OPENING = 0.08  # 8cm maximum
    SAFE_OBJECT_WIDTH = 0.035   # 3.5cm - safe size for reliable grasping
    
    object_ids = []
    object_positions = []  # Track positions to maintain distance
    
    # Calculate feasible object count based on workspace
    workspace_area = 0.30 * 0.50  # 30cm x 50cm workspace
    object_area = np.pi * (MIN_OBJECT_DISTANCE/2)**2  # Exclusion zone per object
    max_objects = max(1, int(workspace_area / object_area * 0.5))  # 50% packing efficiency
    
    # Limit the number of objects to what's feasible
    num_objects = min(num_objects, max_objects)
    print(f"   🎯 Creating {num_objects} objects (max feasible: {max_objects})")
    
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
            if len(object_positions) > 0:  # Only check if there are existing objects
                for existing_pos in object_positions:
                    distance = np.sqrt((candidate_pos[0] - existing_pos[0])**2 + 
                                     (candidate_pos[1] - existing_pos[1])**2)
                    if distance < current_min_distance:
                        too_close = True
                        break
            
            if not too_close:
                # Position is valid, create object here
                placed = True
                
            # Gradually reduce distance requirement if struggling to place
            elif attempts > MAX_SPAWN_ATTEMPTS // 2:
                current_min_distance = MIN_OBJECT_DISTANCE * 0.8  # Reduce to 80%
                if attempts > MAX_SPAWN_ATTEMPTS * 0.75:
                    current_min_distance = MIN_OBJECT_DISTANCE * 0.6  # Further reduce to 60%
        
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
                radius = np.random.uniform(0.015, SAFE_OBJECT_WIDTH/2)  # 1.5-1.75cm
                height = np.random.uniform(0.02, 0.03)                  # 高度: 2-3cm
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
            p.changeDynamics(body, -1, lateralFriction=1.5, restitution=0.2)
            object_ids.append(body)
            
            actual_distance = current_min_distance if len(object_positions) > 1 else 0
            print(f"   📦 Object {i+1} placed at [{x_pos:.3f}, {y_pos:.3f}] (attempts: {attempts}, min_dist: {actual_distance*100:.1f}cm)")
        else:
            print(f"   ⚠️  Could not place object {i+1} after {MAX_SPAWN_ATTEMPTS} attempts - continuing with {len(object_ids)} objects")
    
    # Print final spacing statistics
    if len(object_positions) > 1:
        distances = []
        for i in range(len(object_positions)):
            for j in range(i+1, len(object_positions)):
                dist = np.sqrt((object_positions[i][0] - object_positions[j][0])**2 + 
                              (object_positions[i][1] - object_positions[j][1])**2)
                distances.append(dist)
        
        min_distance = min(distances)
        avg_distance = np.mean(distances)
        print(f"   📏 Object spacing - Min: {min_distance*100:.1f}cm, Avg: {avg_distance*100:.1f}cm")
    
    if len(object_ids) == 0:
        print("   ❌ No objects could be placed! Creating a single object without distance constraints...")
        # Fallback: create at least one object in the center
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
        
    print(f"   ✅ Successfully created {len(object_ids)} objects")
    return object_ids
