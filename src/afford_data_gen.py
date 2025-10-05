# # 调试模式 - 会非常慢，每步都暂停让你观察
# python afford_data_gen.py --debug --num_scenes 1

# # 或者结合可视化
# python afford_data_gen.py --debug --visualize_first --num_scenes 1

# -*- coding: utf-8 -*-
"""
自监督抓取可供性数据生成器 v3 - 清理版
Self-supervised Grasp Affordance Data Generator

简单策略：
1. 用RGB标准差找彩色物体
2. 使用这些像素的深度值
3. 生成抓取候选并测试
"""

import pybullet as p
import numpy as np
import time
import json
import os
from pathlib import Path
import cv2
import argparse

from environment_setup import setup_environment
from perception import set_topdown_camera, get_rgb_depth, get_rgb_depth_segmentation, pixel_to_world, CAMERA_PARAMS


# ==================== 配置参数 ====================

DATA_DIR = Path(__file__).parent.parent / "data" / "affordance_v4"
NUM_ANGLES = 16
ANGLE_BINS = np.linspace(0, np.pi, NUM_ANGLES, endpoint=False)

# 采样参数
FOREGROUND_STRIDE = 8
BACKGROUND_STRIDE = 64
MIN_DEPTH = 0.01
COLOR_DIFF_THRESHOLD = 30  # 颜色差异阈值：与桌子颜色的距离（越大越严格）
EDGE_MARGIN = 20  # 从图像边缘采样桌子颜色的边距（像素）

# 抓取参数
TABLE_TOP_Z = 0.625
PRE_GRASP_OFFSET = 0.12  # 预抓取高度（从物体顶部）
GRASP_OFFSET = -0.015    # 抓取高度：物体顶部下方2mm（进入物体以抓取）
POST_GRASP_OFFSET = 0.00
LIFT_HEIGHT = 0.30
GRIPPER_CLOSED = 0.00
FAST_STEPS = 120
SLOW_STEPS = 600


def create_data_dirs():
    """创建数据目录"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 数据目录: {DATA_DIR.absolute()}")


def move_fast(robot_id, ee_link, target_pos, target_ori, max_steps, slow=False, debug_mode=False):
    """数据收集优化的移动函数 - 添加调试模式"""
    print(f"            🎯 尝试移动到: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    
    # 检查当前位置
    current_pos = p.getLinkState(robot_id, ee_link)[0]
    print(f"            📍 当前位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
    
    # 计算移动距离
    move_distance = np.linalg.norm(np.array(target_pos) - np.array(current_pos))
    print(f"            📏 需要移动距离: {move_distance*100:.1f}cm")
    
    ll, ul, jr, rp = [], [], [], []
    for i in range(7):
        info = p.getJointInfo(robot_id, i)
        ll.append(info[8])
        ul.append(info[9])
        jr.append(info[9] - info[8])
        rp.append(p.getJointState(robot_id, i)[0])
    
    # ✨ 增强的IK求解
    joints = p.calculateInverseKinematics(
        robot_id, ee_link, target_pos, target_ori,
        lowerLimits=ll, upperLimits=ul, jointRanges=jr, restPoses=rp,
        maxNumIterations=100,
        residualThreshold=1e-4
    )
    
    if not joints or len(joints) < 7:
        print(f"            ❌ IK求解失败: joints={joints}")
        return False
    
    # ✨ 检查关节限制
    joint_violations = []
    for i in range(7):
        if joints[i] < ll[i] or joints[i] > ul[i]:
            joint_violations.append(f"Joint{i}: {joints[i]:.3f} 超出 [{ll[i]:.3f}, {ul[i]:.3f}]")
    
    if joint_violations:
        print(f"            ❌ 关节角度超限: {joint_violations}")
        return False
    
    print(f"            ✅ IK求解成功，关节角度在限制内")
    
    # ✨ 修复：根据距离动态调整运动参数
    if move_distance > 0.3:  # 如果距离超过30cm
        velocity = 2.0 if slow else 3.0    # 增加速度
        force = 1500 if slow else 2500     # 大幅增加力度  
        actual_steps = 300 if slow else 300  # 大幅增加步数确保到达
        print(f"            🚀 远距离移动模式: 速度={velocity}, 力度={force}, 步数={actual_steps}")
    else:
        velocity = 1.0 if slow else 2.0
        force = 600 if slow else 1000
        actual_steps = 80 if slow else 60
        print(f"            🎯 近距离移动模式: 速度={velocity}, 力度={force}, 步数={actual_steps}")
    
    # ✨ 添加调试模式 - 非常慢的运动用于观察
    if debug_mode:
        velocity = 1 # 非常慢的速度
        force = 100     # 较小的力
        actual_steps = 200  # 更多步数
        print(f"            🐌 调试模式: 速度={velocity}, 力度={force}, 步数={actual_steps}")
    
    for i in range(7):
        p.setJointMotorControl2(
            robot_id, i, p.POSITION_CONTROL,
            targetPosition=joints[i], force=force, maxVelocity=velocity
        )
    
    # ✨ 调试模式下更频繁的进度报告
    progress_interval = actual_steps // 8 if debug_mode else actual_steps // 4
    
    for step in range(actual_steps):
        p.stepSimulation()
        if debug_mode:
            time.sleep(1./60.)  # 调试模式下更慢的仿真
        else:
            time.sleep(1./240.)
        
        # 更频繁的进度检查
        if step % progress_interval == 0:
            current = p.getLinkState(robot_id, ee_link)[0]
            dist = np.linalg.norm(np.array(current) - np.array(target_pos))
            progress = max(0, (move_distance - dist) / move_distance * 100)
            print(f"            📊 步骤 {step}/{actual_steps}: 距离目标 {dist*100:.1f}cm, 进度 {progress:.1f}%")
            
            # 调试模式下暂停让用户观察
            if debug_mode and step > 0:
                input(f"            ⏸️  [调试] 按 Enter 继续... (当前进度: {progress:.1f}%)")
            
            # 早期成功检测
            if dist < 0.05:  # 如果已经很接近目标
                print(f"            ✅ 提前到达目标 (距离 {dist*100:.1f}cm)")
                break
    
    # ✨ 最终位置验证
    final_pos = p.getLinkState(robot_id, ee_link)[0]
    final_dist = np.linalg.norm(np.array(final_pos) - np.array(target_pos))
    print(f"            📍 最终位置: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
    print(f"            📏 最终误差: {final_dist*100:.1f}cm")
    
    # ✨ 根据移动距离动态调整容差
    if move_distance > 0.4:
        success_threshold = 0.15  # 15cm for very long moves
    elif move_distance > 0.2:
        success_threshold = 0.10  # 10cm for medium moves  
    else:
        success_threshold = 0.05  # 5cm for short moves
    
    success = final_dist < success_threshold
    
    if success:
        print(f"            ✅ 移动成功 (误差 {final_dist*100:.1f}cm < {success_threshold*100:.0f}cm)")
    else:
        print(f"            ❌ 移动失败 (误差 {final_dist*100:.1f}cm >= {success_threshold*100:.0f}cm)")
        print(f"            💡 建议: 原始距离 {move_distance*100:.1f}cm, 容差 {success_threshold*100:.0f}cm")
    
    return success

def close_gripper_slow(robot_id, steps):
    """慢速闭合夹爪"""
    pos = GRIPPER_CLOSED / 2.0
    p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, targetPosition=pos, force=50, maxVelocity=0.05)
    p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, targetPosition=pos, force=50, maxVelocity=0.05)
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1./240.)


def open_gripper_fast(robot_id):
    """打开夹爪 - 增强版"""
    pos = 0.04 / 2.0  # 完全打开
    
    # ✨ 使用最强的力和最快的速度强制打开
    p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, 
                          targetPosition=pos, force=300, maxVelocity=3.0)
    p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, 
                          targetPosition=pos, force=300, maxVelocity=3.0)
    
    # 确保夹爪完全打开
    for _ in range(40):  # 增加步数
        p.stepSimulation()
        time.sleep(1./240.)
    
    # 验证并报告夹爪状态
    finger_state = p.getJointState(robot_id, 9)[0]
    if finger_state > 0.015:
        print(f"            ✅ 夹爪已打开: {finger_state:.4f}")
    else:
        print(f"            ❌ 夹爪可能未完全打开: {finger_state:.4f}")

def reset_robot_home(robot_id):
    """重置机器人到初始位置"""
    home = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    
    # ✨ 确保在移动前夹爪是完全打开的
    print("   🔓 确保夹爪完全打开...")
    
    # 多次尝试确保夹爪打开
    for attempt in range(3):
        open_gripper_fast(robot_id)
        finger_state = p.getJointState(robot_id, 9)[0]
        print(f"      尝试 {attempt+1}: 夹爪状态 = {finger_state:.4f}")
        
        if finger_state > 0.015:
            print(f"      ✅ 夹爪已确认打开")
            break
        else:
            print(f"      ⚠️  夹爪未完全打开，重试...")
    
    # 使用位置控制而不是直接设置关节状态，更平滑
    for i in range(7):
        p.setJointMotorControl2(
            robot_id, i, p.POSITION_CONTROL,
            targetPosition=home[i], 
            force=500, 
            maxVelocity=2.0
        )
    
    # 等待到位
    for _ in range(120):
        p.stepSimulation()
        
        # 检查是否到位
        all_in_position = True
        for i in range(7):
            current = p.getJointState(robot_id, i)[0]
            if abs(current - home[i]) > 0.05:  # 容差3度
                all_in_position = False
                break
        
        if all_in_position:
            break
    
    # ✨ 最后再次强制确保夹爪打开
    print("   🔓 最终确保夹爪打开...")
    open_gripper_fast(robot_id)
    
    final_finger_state = p.getJointState(robot_id, 9)[0]
    print(f"   🏠 机器人已回到初始位置，夹爪状态: {final_finger_state:.4f}")


def estimate_object_height(depth, object_mask, percentile=10):
    """估计物体表面高度
    
    使用检测到的物体像素的深度值估计表面高度
    使用较小百分位数来避免噪声和边缘效应
    
    Args:
        depth: 深度图
        object_mask: 物体mask
        percentile: 使用的百分位数（默认10 = 最近的10%像素）
    
    Returns:
        物体表面高度（世界坐标Z值）
    """
    obj_depths = depth[object_mask]
    valid_depths = obj_depths[obj_depths > MIN_DEPTH]
    
    if len(valid_depths) == 0:
        return None
    
    # 使用较小百分位数的深度值（最接近相机 = 最高点）
    surface_depth = np.percentile(valid_depths, percentile)
    
    # 深度到世界Z的转换（简化版，假设俯视相机）
    # 相机高度 = TABLE_TOP_Z + camera_distance
    # 物体Z = 相机高度 - 深度
    camera_height = TABLE_TOP_Z + 1.2  # CAMERA_DISTANCE = 1.2
    object_z = camera_height - surface_depth
    
    return object_z

def fast_grasp_test(robot_id, world_pos, grasp_angle, object_ids, visualize=False, debug_mode=False):
    """数据收集优化的抓取测试 - 添加调试模式"""
    ee_link = 11
    
    print(f"         🎯 抓取测试: 位置=[{world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}], 角度={np.degrees(grasp_angle):.1f}°")
    
    if debug_mode:
        print(f"         🐌 调试模式启用 - 运动将很慢，每步都会暂停")
        input(f"         ⏸️  [调试] 按 Enter 开始抓取测试...")
    
    # 快速工作空间检查 - 与auto_collect_data.py一致
    dist = np.sqrt(world_pos[0]**2 + world_pos[1]**2)
    if dist < 0.25 or dist > 0.80 or abs(world_pos[1]) > 0.30:
        print(f"         ❌ 工作空间检查失败: 距离={dist:.3f}m, Y={world_pos[1]:.3f}m")
        return False
    
    # ✨ 修复高度检查 - 更宽松的范围
    if world_pos[2] < TABLE_TOP_Z - 0.02 or world_pos[2] > TABLE_TOP_Z + 0.15:
        print(f"         ❌ 高度检查失败: Z={world_pos[2]:.3f}m, 桌面={TABLE_TOP_Z:.3f}m, 范围=[{TABLE_TOP_Z-0.02:.3f}, {TABLE_TOP_Z+0.15:.3f}]")
        return False
    
    # 记录初始物体高度
    initial_z = {}
    for obj_id in object_ids:
        try:
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            initial_z[obj_id] = pos[2]
        except:
            continue
    
    try:
        ori = p.getQuaternionFromEuler([np.pi, 0, grasp_angle])
        
        # 1. 预抓取阶段
        max_reachable_z = TABLE_TOP_Z + 0.35
        pre_grasp_height = min(world_pos[2] + 0.08, max_reachable_z)
        pre_pos = [world_pos[0], world_pos[1], pre_grasp_height]
        print(f"         ↑ 预抓取: [{pre_pos[0]:.3f}, {pre_pos[1]:.3f}, {pre_pos[2]:.3f}]")
        
        if debug_mode:
            input(f"         ⏸️  [调试] 按 Enter 开始预抓取移动...")
        
        if not move_fast(robot_id, ee_link, pre_pos, ori, 30, debug_mode=debug_mode):
            print(f"         ❌ 预抓取失败")
            return False
        
        # 2. 下降阶段
        grasp_z = max(world_pos[2] - 0.01, TABLE_TOP_Z + 0.01)
        grasp_pos = [world_pos[0], world_pos[1], grasp_z]
        print(f"         ↓ 下降: [{grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f}]")
        
        if debug_mode:
            input(f"         ⏸️  [调试] 按 Enter 开始下降...")
        
        if not move_fast(robot_id, ee_link, grasp_pos, ori, 30, slow=True, debug_mode=debug_mode):
            print(f"         ❌ 下降失败")
            return False
        
        # 3. 闭合夹爪
        print(f"         🤏 闭合夹爪...")
        if debug_mode:
            input(f"         ⏸️  [调试] 按 Enter 开始闭合夹爪...")
        
        # 调试模式下慢速闭合夹爪
        gripper_steps = 40 if debug_mode else 20
        for step in range(gripper_steps):
            p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, targetPosition=0.0, force=50)
            p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, targetPosition=0.0, force=50)
            p.stepSimulation()
            if debug_mode:
                time.sleep(1./60.)
                if step % 10 == 0:
                    finger_state = p.getJointState(robot_id, 9)[0]
                    print(f"           🤏 夹爪进度: {finger_state:.4f}")
            else:
                time.sleep(1./240.)
        
        finger_state = p.getJointState(robot_id, 9)[0]
        print(f"         📏 夹爪闭合后: {finger_state:.4f}")
        
        # 4. 抬起阶段
        lift_z = min(grasp_z + 0.10, max_reachable_z)
        lift_pos = [grasp_pos[0], grasp_pos[1], lift_z]
        print(f"         ↑ 抬起: [{lift_pos[0]:.3f}, {lift_pos[1]:.3f}, {lift_pos[2]:.3f}]")
        
        if debug_mode:
            input(f"         ⏸️  [调试] 按 Enter 开始抬起...")
        
        if not move_fast(robot_id, ee_link, lift_pos, ori, 30, debug_mode=debug_mode):
            print(f"         ❌ 抬起失败")
            return False
        
        # 5. 成功检查
        success = False
        print(f"         🔍 检查物体状态...")
        if debug_mode:
            input(f"         ⏸️  [调试] 按 Enter 检查抓取结果...")
        
        for obj_id in object_ids:
            try:
                pos, _ = p.getBasePositionAndOrientation(obj_id)
                if obj_id in initial_z:
                    lift_height = pos[2] - initial_z[obj_id]
                    print(f"           物体{obj_id}: 初始Z={initial_z[obj_id]:.3f}, 当前Z={pos[2]:.3f}, 抬起={lift_height*100:.1f}cm")
                    if lift_height > 0.03:
                        success = True
                        print(f"         ✅ 成功抬起物体{obj_id}!")
                        break
            except:
                print(f"           物体{obj_id}: 已消失")
                continue
        
        if not success:
            print(f"         ❌ 没有物体被抬起")
        
        if debug_mode:
            result = "成功" if success else "失败"
            input(f"         ⏸️  [调试] 抓取{result}！按 Enter 释放夹爪...")
        
        # 6. 释放
        print(f"         🔓 释放夹爪...")
        p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, targetPosition=0.04/2.0, force=100)
        p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, targetPosition=0.04/2.0, force=100)
        for _ in range(15):
            p.stepSimulation()
            time.sleep(1./240.)
        
        return success
        
    except Exception as e:
        print(f"         ❌ 异常: {e}")
        # 异常时快速打开夹爪
        try:
            p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, targetPosition=0.04/2.0, force=100)
            p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, targetPosition=0.04/2.0, force=100)
            for _ in range(10):
                p.stepSimulation()
        except:
            pass
        return False

def sample_grasp_candidates(depth, num_angles=NUM_ANGLES, visualize=False, rgb=None, view_matrix=None, proj_matrix=None, seg_mask=None, object_ids=None):
    """数据收集优化的候选采样 - 随机化修复版"""
    height, width = depth.shape
    candidates = []
    
    if seg_mask is None or object_ids is None:
        return candidates
    
    if len(object_ids) == 0:
        return candidates
    
    # 工作空间检查
    valid_objects = []
    for obj_id in object_ids:
        try:
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            
            dist_from_base = np.sqrt(pos[0]**2 + pos[1]**2)
            workspace_valid = (
                pos[2] >= TABLE_TOP_Z and pos[2] <= TABLE_TOP_Z + 0.3 and
                dist_from_base >= 0.30 and dist_from_base <= 0.85 and
                abs(pos[1]) <= 0.5
            )
            
            print(f"   🔍 物体 {obj_id} 位置检查: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], 距离={dist_from_base:.3f}m, 有效={workspace_valid}")
            
            if workspace_valid:
                obj_pixels = (seg_mask == obj_id)
                pixel_count = obj_pixels.sum()
                print(f"   📊 物体 {obj_id} 分割像素: {pixel_count}")
                if pixel_count > 5:
                    valid_objects.append(obj_id)
                    print(f"   ✅ 物体 {obj_id} 有效")
                else:
                    print(f"   ❌ 物体 {obj_id} 像素不足 (只有 {pixel_count} 像素)")
            else:
                print(f"   ❌ 物体 {obj_id} 超出工作空间")
        except Exception as e:
            print(f"   ❌ 物体 {obj_id} 检查失败: {e}")
            continue
    
    if len(valid_objects) == 0:
        print(f"   ❌ 没有有效的物体")
        return []
    
    # ✨ 修复：为每个有效物体生成随机候选点
    for obj_id in valid_objects:
        obj_pixels = (seg_mask == obj_id)
        obj_pixels &= (depth > MIN_DEPTH)
        
        print(f"   🔍 物体 {obj_id} 像素统计: {obj_pixels.sum()} 像素")
        
        if obj_pixels.sum() == 0:
            print(f"   ⚠️  物体 {obj_id} 在分割掩码中没有像素，跳过")
            continue
        
        obj_coords = np.where(obj_pixels)
        if len(obj_coords[0]) == 0:
            print(f"   ⚠️  物体 {obj_id} 坐标为空，跳过")
            continue
        
        # ✨ 随机选择物体上的点，而不是总是选择中心
        num_samples = min(8, len(obj_coords[0]))  # 每个物体最多8个点
        if num_samples > 0:
            sample_indices = np.random.choice(len(obj_coords[0]), num_samples, replace=False)
            
            for idx in sample_indices:
                v, u = obj_coords[0][idx], obj_coords[1][idx]
                
                # ✨ 随机选择角度
                theta_idx = np.random.randint(0, min(8, num_angles))
                theta = ANGLE_BINS[theta_idx]
                
                candidates.append((u, v, theta_idx, theta))
    
    # ✨ 添加随机背景点作为负样本
    bg_count = 0
    for _ in range(15):  # 尝试15次
        u = np.random.randint(20, width-20)
        v = np.random.randint(20, height-20)
        
        # 确保不在任何物体上
        if seg_mask[v, u] <= 2 and depth[v, u] > MIN_DEPTH:
            candidates.append((u, v, 0, 0.0))
            bg_count += 1
            if bg_count >= 8:  # 最多8个背景样本
                break
    
    # ✨ 关键修复：随机打乱候选列表
    np.random.shuffle(candidates)
    
    print(f"   📍 采样了 {len(candidates)} 个候选点 (物体数: {len(valid_objects)}, 背景: {bg_count}) - 已随机打乱")
    
    return candidates

def generate_scene_data(scene_id, num_objects=3, visualize=False, debug_mode=False):
    """生成单个场景数据 - 添加调试模式"""
    print(f"\n🎬 场景 {scene_id:04d}")
    if debug_mode:
        print(f"🐌 调试模式启用 - 所有动作都会很慢并暂停让您观察")
    
    client = p.connect(p.GUI if (visualize or debug_mode) else p.DIRECT)
    if client < 0:
        return False
    
    try:
        robot_id, object_ids = setup_environment(num_objects=num_objects)
        if not object_ids:
            return False
        
        print("   ⏳ 等待物体稳定...")
        for _ in range(120):
            p.stepSimulation()
        
        # 确保机器人在初始位置
        reset_robot_home(robot_id)
        for _ in range(60):
            p.stepSimulation()
        
        if visualize or debug_mode:
            print("   ⏸️  按 Enter 继续...")
            input()
        
        # 主循环：持续抓取直到没有物体
        total_samples = 0
        total_success = 0
        grasp_attempt = 0
        consecutive_failures = 0
        
        # 用于保存最终数据的变量
        final_rgb = None
        final_depth = None
        final_label = None
        
        while grasp_attempt < 50:
            grasp_attempt += 1
            
            print(f"\n   📸 更新相机图像 (尝试 {grasp_attempt})")
            
            # 确保机器人回到初始位置
            print("   🏠 确保机器人回到初始位置...")
            reset_robot_home(robot_id)
            
            # 等待机器人完全稳定
            for _ in range(120):
                p.stepSimulation()
            
            # ✨ 关键修复：先更新物体状态再拍照
            print("   🔄 更新物体状态...")
            from environment_setup import update_object_states
            old_count = len(object_ids)
            object_ids = update_object_states(object_ids)
            new_count = len(object_ids)
            
            print(f"   📦 物体状态: {old_count} → {new_count}")
            
            # ✨ 修复2：如果连续多次没有有效物体，强制清理和重新生成
            if len(object_ids) == 0 or consecutive_failures >= 3:
                if consecutive_failures >= 3:
                    print(f"   ⚠️  连续 {consecutive_failures} 次失败，强制清理并重新生成物体...")
                else:
                    print("   ⚠️  桌面为空，立即重新生成物体...")
                
                from environment_setup import reset_objects_after_grasp
                object_ids = reset_objects_after_grasp([], min_objects=num_objects)
                consecutive_failures = 0
                
                if len(object_ids) == 0:
                    print("   ❌ 无法生成新物体，结束场景")
                    break
                else:
                    print(f"   ✅ 成功生成 {len(object_ids)} 个新物体")
                    # 等待物体稳定
                    for _ in range(120):
                        p.stepSimulation()
                    # 重新开始这个循环迭代，不计入尝试次数
                    grasp_attempt -= 1
                    continue
            
            # 拍摄新照片（确保有物体后才拍照）
            width, height, view_matrix, proj_matrix = set_topdown_camera()
            rgb, depth, seg_mask = get_rgb_depth_segmentation(width, height, view_matrix, proj_matrix)
            
            # 保存当前图像
            final_rgb = rgb.copy()
            final_depth = depth.copy()
            
            if visualize:
                for i, obj_id in enumerate(object_ids):
                    try:
                        pos, _ = p.getBasePositionAndOrientation(obj_id)
                        print(f"   📦 物体{i+1} (ID={obj_id}): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                    except:
                        print(f"   ❌ 物体{i+1} (ID={obj_id}): 已不存在")
            
            # 基于当前图像采样候选
            candidates = sample_grasp_candidates(depth, NUM_ANGLES, visualize, rgb, view_matrix, proj_matrix, seg_mask, object_ids)
            
            # ✨ 修复：如果候选点为空，立即触发重新生成
            if len(candidates) == 0:
                print("   ⚠️  未找到有效候选点")
                consecutive_failures += 1
                
                # 立即触发重新生成
                print("   🔄 立即重新生成物体...")
                from environment_setup import reset_objects_after_grasp
                object_ids = reset_objects_after_grasp([], min_objects=num_objects)
                consecutive_failures = 0
                
                if len(object_ids) == 0:
                    print("   ❌ 无法生成新物体，结束场景")
                    break
                else:
                    print(f"   ✅ 重新生成 {len(object_ids)} 个物体")
                    # 等待物体稳定
                    for _ in range(120):
                        p.stepSimulation()
                    # 重新开始循环，不计入尝试次数
                    grasp_attempt -= 1
                    continue
            
            # 重置失败计数器
            consecutive_failures = 0
            
            # ✨ 修复：测试多个候选，而不是只测试第一个
            max_candidates_to_test = min(5, len(candidates))  # 每次最多测试5个候选
            tested_candidates = 0
            scene_success = False
            
            for candidate_idx in range(max_candidates_to_test):
                u, v, theta_idx, theta = candidates[candidate_idx]
                
                if depth[v, u] < MIN_DEPTH:
                    print(f"   ⚠️  候选{candidate_idx+1}深度无效，跳过")
                    continue
                
                total_samples += 1
                tested_candidates += 1
                
                print(f"\n      === 测试候选 {candidate_idx+1}/{max_candidates_to_test} ===")
                print(f"         像素: ({u}, {v}), 角度: {np.degrees(theta):.1f}°")
                
                world_pos = pixel_to_world(u, v, depth[v, u], view_matrix, proj_matrix)
                print(f"         世界坐标: [{world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}]")
                
                # 执行抓取测试 - 传递调试模式
                success = fast_grasp_test(robot_id, world_pos, theta, object_ids, visualize, debug_mode=debug_mode)
                
                if success:
                    total_success += 1
                    scene_success = True
                    print(f"      ✅ 候选{candidate_idx+1}成功抓取！")
                    
                    # 创建/更新标签
                    if final_label is None:
                        final_label = np.zeros((height, width, NUM_ANGLES + 1), dtype=np.uint8)
                    
                    final_label[v, u, theta_idx] = 1
                    
                    # 更新物体列表
                    object_ids = update_object_states(object_ids)
                    print(f"      📦 剩余物体: {len(object_ids)}")
                    
                    break  # 找到成功的候选后停止测试更多候选
                else:
                    print(f"      ❌ 候选{candidate_idx+1}抓取失败")
            
            if not scene_success:
                consecutive_failures += 1
                print(f"      ❌ 所有候选都失败了 (测试了{tested_candidates}个)")
            
            if grasp_attempt % 10 == 0:
                print(f"   📊 进度: {grasp_attempt} 次尝试, 成功: {total_success}, 成功率: {100*total_success/total_samples if total_samples > 0 else 0:.1f}%")
        
        # 完成最终标签
        if final_label is not None:
            # 设置背景标签
            has_success = final_label[:, :, :-1].sum(axis=2) > 0
            final_label[:, :, -1] = (~has_success).astype(np.uint8)
            
            final_rate = total_success / total_samples if total_samples > 0 else 0
            print(f"\n   ✅ 总体成功率: {final_rate*100:.1f}% ({total_success}/{total_samples})")
            
            # 保存最终数据
            save_scene_data(scene_id, final_rgb, final_depth, final_label, {
                "num_objects": num_objects,
                "num_samples": total_samples,
                "success_count": int(total_success),
                "success_rate": final_rate,
                "grasp_attempts": grasp_attempt
            })
        else:
            print(f"   ⚠️  场景 {scene_id} 没有生成任何数据")
        
        return True
    
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print(f"   🔚 断开连接...")
        p.disconnect()

def save_scene_data(scene_id, rgb, depth, label, metadata):
    """保存数据"""
    prefix = DATA_DIR / f"scene_{scene_id:04d}"
    cv2.imwrite(str(prefix) + "_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(str(prefix) + "_depth.npy", depth)
    np.save(str(prefix) + "_label.npy", label)
    with open(str(prefix) + "_meta.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   💾 保存: {prefix}_*")


def generate_dataset(num_scenes, num_objects_range=(1, 3), visualize_first=False, debug_mode=False):
    """批量生成 - 添加调试模式"""
    print("=" * 60)
    print("🚀 生成数据集")
    if debug_mode:
        print("🐌 调试模式启用")
    print("=" * 60)
    
    create_data_dirs()
    success_scenes = 0
    start = time.time()
    
    for scene_id in range(num_scenes):
        num_objects = np.random.randint(num_objects_range[0], num_objects_range[1] + 1)
        vis = visualize_first and (scene_id == 0)
        
        if generate_scene_data(scene_id, num_objects, vis, debug_mode=debug_mode):
            success_scenes += 1
    
    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"✅ 完成！{success_scenes}/{num_scenes}")
    print(f"   耗时: {elapsed:.1f}s ({elapsed/num_scenes:.1f}s/场景)")
    print(f"   位置: {DATA_DIR.absolute()}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_scenes", type=int, default=10)
    parser.add_argument("--num_objects", type=int, nargs=2, default=[1, 3])
    parser.add_argument("--visualize_first", action="store_true")
    parser.add_argument("--debug", action="store_true", help="启用调试模式 - 慢速运动，每步暂停")
    args = parser.parse_args()
    
    generate_dataset(
        num_scenes=args.num_scenes,
        num_objects_range=tuple(args.num_objects),
        visualize_first=args.visualize_first,
        debug_mode=args.debug
    )


if __name__ == "__main__":
    main()