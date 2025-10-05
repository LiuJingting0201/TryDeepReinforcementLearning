#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化可供性训练数据收集器
Automatic Affordance Training Data Collector

基于已验证的工作系统，自动收集训练数据
"""

import numpy as np
import cv2
import os
import json
import time
import argparse
import pybullet as p
from PIL import Image
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# 导入工作的模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.environment_setup import setup_environment
from src.perception import set_topdown_camera, get_rgb_depth_segmentation, pixel_to_world
from src.afford_data_gen import sample_grasp_candidates, fast_grasp_test, reset_robot_home

class AutoAffordanceCollector:
    """自动化可供性数据收集器"""
    
    def __init__(self, data_dir="data/affordance_v5", num_angles=8, train_split=0.8):
        # Convert to absolute path relative to the script's parent directory (workspace root)
        if not Path(data_dir).is_absolute():
            script_dir = Path(__file__).parent
            workspace_root = script_dir.parent
            data_dir = workspace_root / data_dir
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建训练和测试子目录
        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test"
        self.train_dir.mkdir(exist_ok=True)
        self.test_dir.mkdir(exist_ok=True)
        
        self.num_angles = num_angles
        self.train_split = train_split  # 训练集比例 (0.8 = 80%)
        
        # 抓取角度设置
        self.grasp_angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
        
        print(f"🎯 自动化可供性数据收集器初始化")
        print(f"   数据目录: {self.data_dir.absolute()}")
        print(f"   训练目录: {self.train_dir}")
        print(f"   测试目录: {self.test_dir}")
        print(f"   训练集比例: {train_split:.1%}")
        print(f"   抓取角度数: {num_angles}")
    
    def collect_single_attempt_as_scene(self, scene_idx, robot_id, object_ids, target_dir):
        """收集单次抓取尝试作为独立场景，每次抓取前恢复环境到初始状态"""
        print(f"============================================================")
        
        # 1. 记录初始机器人和物体状态
        home_joints = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        initial_robot_joints = [p.getJointState(robot_id, i)[0] for i in range(7)]
        initial_gripper = [p.getJointState(robot_id, 9)[0], p.getJointState(robot_id, 10)[0]]
        object_states = {}
        for obj_id in object_ids:
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            object_states[obj_id] = (pos, orn)
        
        # 2. 确保机器人在初始位置
        print("   🏠 重置机器人...")
        for i in range(7):
            p.setJointMotorControl2(
                robot_id, i, p.POSITION_CONTROL,
                targetPosition=home_joints[i], 
                force=500, 
                maxVelocity=2.0
            )
        for _ in range(200):
            p.stepSimulation()
            all_in_position = True
            for i in range(7):
                current = p.getJointState(robot_id, i)[0]
                if abs(current - home_joints[i]) > 0.05:
                    all_in_position = False
                    break
            if all_in_position:
                break
        # 强制打开夹爪
        p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, targetPosition=0.02, force=300)
        p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, targetPosition=0.02, force=300)
        for _ in range(40):
            p.stepSimulation()
        # 验证机器人位置
        current_pos = p.getLinkState(robot_id, 8)[0]
        print(f"   📍 机器人末端位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
        # 拍摄照片
        print("   📷 拍摄场景照片...")
        width, height, view_matrix, proj_matrix = set_topdown_camera()
        rgb_image, depth_image, seg_mask = get_rgb_depth_segmentation(width, height, view_matrix, proj_matrix)
        print(f"   📷 相机数据: RGB {rgb_image.shape}, 深度 {depth_image.shape}")
        # 采样抓取候选点
        candidates = sample_grasp_candidates(
            depth=depth_image,
            num_angles=self.num_angles,
            visualize=False,
            rgb=rgb_image,
            view_matrix=view_matrix,
            proj_matrix=proj_matrix,
            seg_mask=seg_mask,
            object_ids=object_ids
        )
        
        print(f"   🔍 候选点采样结果: {len(candidates)} 个候选点")
        
        if not candidates:
            print("   ❌ 没有找到有效的抓取候选点")
            return False
        
        print(f"   📍 采样了 {len(candidates)} 个候选点 - 测试前 {min(20, len(candidates))} 个")
        
        # 测试多个候选点（每个场景最多20个，提供更密集的训练标签）
        test_count = min(20, len(candidates))
        successful_grasps = []
        failed_grasps = []
        
        for i, candidate in enumerate(candidates[:test_count]):
            if len(candidate) == 4:
                u, v, theta_idx, theta = candidate
            else:
                u, v, theta_idx = candidate
                theta = self.grasp_angles[theta_idx]
            world_pos = pixel_to_world(u, v, depth_image[v, u], view_matrix, proj_matrix)
            print(f"   🎯 测试 {i+1}/{test_count}: 像素({u},{v}), 角度{np.degrees(theta):.1f}°")
            # 每次抓取前恢复机器人和物体到初始状态
            if i > 0:
                print(f"      🏠 恢复机器人和物体到初始状态...")
                # 恢复机器人关节
                for j in range(7):
                    p.resetJointState(robot_id, j, home_joints[j])
                # 恢复夹爪
                p.resetJointState(robot_id, 9, 0.02)
                p.resetJointState(robot_id, 10, 0.02)
                # 恢复物体
                for obj_id in object_ids:
                    pos, orn = object_states[obj_id]
                    p.resetBasePositionAndOrientation(obj_id, pos, orn)
                # 让物理引擎稳定
                for _ in range(10):
                    p.stepSimulation()
            # 执行抓取测试
            success = fast_grasp_test(robot_id, world_pos, theta, object_ids, visualize=False)
            if success:
                print(f"      ✅ 成功!")
                successful_grasps.append({'pixel': (u, v), 'angle': theta, 'world_pos': world_pos})
            else:
                print(f"      ❌ 失败")
                failed_grasps.append({'pixel': (u, v), 'angle': theta, 'world_pos': world_pos})
        
        # 计算成功率
        success_rate = len(successful_grasps) / test_count if test_count > 0 else 0
        print(f"   📊 场景成功率: {len(successful_grasps)}/{test_count} ({success_rate:.1%})")
        
        # 生成可供性标签（标记所有测试的点）
        affordance_map = np.zeros((224, 224), dtype=np.float32)
        angle_map = np.zeros((224, 224), dtype=np.float32) 
        
        # 标记成功的抓取点
        for grasp in successful_grasps:
            u, v = grasp['pixel']
            theta = grasp['angle']
            affordance_map[v, u] = 1.0  # 成功点标记为1
            angle_map[v, u] = theta
        
        # 失败的点保持为0（已经初始化为0）
        
        # 合并所有抓取结果
        all_results = []
        for grasp in successful_grasps + failed_grasps:
            all_results.append({
                'success': grasp in successful_grasps,
                'pixel': grasp['pixel'],
                'world_pos': grasp['world_pos'],
                'angle': grasp['angle']
            })
        
        # 使用统一的保存方法
        self.save_scene_data(scene_idx, rgb_image, depth_image, affordance_map, angle_map, all_results, view_matrix, proj_matrix, target_dir, robot_id=robot_id)
        
        return True
        
    def is_position_reachable(self, world_pos):
        """检查位置是否在机器人工作空间内"""
        x, y, z = world_pos
        
        # 基本工作空间限制
        distance = np.sqrt(x**2 + y**2)
        
        # Franka Panda的工作空间约束
        if distance < 0.3 or distance > 0.85:  # 距离限制
            return False
        if abs(y) > 0.4:  # Y轴限制
            return False
        if z < 0.58 or z > 0.8:  # Z轴高度限制
            return False
        if x < 0.2 or x > 0.9:  # X轴前后限制
            return False
            
        return True
    
        """生成可供性热力图"""
    def generate_angle_map(self, results, image_shape):
        """生成最佳抓取角度地图"""
        angle_map = np.zeros(image_shape, dtype=np.float32)
        
        for result in results:
            if result['success'] and result['world_pos'][0] != 0:
                u, v = result['pixel']
                if 0 <= v < image_shape[0] and 0 <= u < image_shape[1]:
                    # 归一化角度到 [0, 1]
                    normalized_angle = result['angle'] / (2 * np.pi)
                    angle_map[v, u] = normalized_angle
        
        return angle_map
    
    def create_affordance_map(self, image_shape, results):
        """创建可供性地图"""
        height, width = image_shape
        affordance_map = np.zeros((height, width), dtype=np.float32)
        
        for result in results:
            if 'pixel' in result and len(result['pixel']) == 2:
                u, v = result['pixel']
                if 0 <= v < height and 0 <= u < width:
                    affordance_map[v, u] = 1.0 if result['success'] else 0.0
        
        return affordance_map
    
    def create_angle_map(self, image_shape, results):
        """创建角度地图"""
        height, width = image_shape
        angle_map = np.zeros((height, width), dtype=np.float32)
        
        for result in results:
            if result['success'] and result['world_pos'][0] != 0:
                u, v = result['pixel']
                if 0 <= v < height and 0 <= u < width:
                    angle_map[v, u] = result['angle']
        
        return angle_map
    
    def save_scene_data(self, scene_id, rgb_image, depth_image, affordance_map, angle_map, results, view_matrix, proj_matrix, target_dir, seg_mask=None, robot_id=None):
        """保存场景数据"""
        scene_prefix = f"scene_{scene_id:04d}"
        
        # 保存图像
        rgb_path = target_dir / f"{scene_prefix}_rgb.png"
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        
        # 转换深度图为相对深度
        # 使用精确的工作空间mask (基于世界坐标)
        height, width = depth_image.shape
        workspace_mask = np.zeros((height, width), dtype=bool)
        edge_mask = np.zeros((height, width), dtype=bool)
        
        # 为每个像素计算世界坐标并检查是否在工作空间内
        TABLE_TOP_Z = 0.625  # 从environment_setup.py
        for v in range(height):
            for u in range(width):
                # 将像素坐标转换为世界坐标
                world_pos = pixel_to_world(u, v, depth_image[v, u], view_matrix, proj_matrix)
                x, y, z = world_pos
                
                # 检查是否在工作空间内
                dist_from_base = np.sqrt(x**2 + y**2)
                in_workspace = (
                    z >= TABLE_TOP_Z and z <= TABLE_TOP_Z + 0.25 and  # Z范围更保守
                    dist_from_base >= 0.35 and dist_from_base <= 0.80 and  # 距离范围更保守
                    abs(y) <= 0.30  # Y轴范围更严格，排除机器人基座
                )
                
                workspace_mask[v, u] = in_workspace
                
                # 定义边缘区域：只在边缘采样桌面深度，避免物体干扰
                if in_workspace and (dist_from_base <= 0.45 or dist_from_base >= 0.70):
                    edge_mask[v, u] = True
        
        # 只使用工作空间边缘的像素计算桌面深度，避免物体干扰
        edge_depths = depth_image[edge_mask]
        
        if len(edge_depths) == 0:
            print("      ⚠️ 警告: 边缘区域没有有效像素，使用全工作空间")
            edge_depths = depth_image[workspace_mask]
            if len(edge_depths) == 0:
                edge_depths = depth_image.flatten()
        
        # 使用边缘区域内深度最低的像素作为桌面
        sorted_depths = np.sort(edge_depths)
        num_table_candidates = max(50, len(sorted_depths) // 5)
        table_pixels = sorted_depths[:num_table_candidates]
        table_depth = np.percentile(table_pixels, 10)
        
        print(f"      📏 save_scene_data: 使用工作空间边缘 {num_table_candidates} 个深度像素计算桌面深度: {table_depth:.6f}")
        
        # 创建相对深度图：工作空间内像素使用实际相对深度，工作空间外像素设为0（桌面深度）
        relative_depth = np.zeros_like(depth_image)
        relative_depth[workspace_mask] = depth_image[workspace_mask] - table_depth
        # 工作空间外像素保持为0（相当于桌面深度）
        
        # 保存深度图（相对深度）
        depth_path = target_dir / f"{scene_prefix}_depth.npy"
        np.save(depth_path, relative_depth)
        
        # 保存可供性地图
        affordance_path = target_dir / f"{scene_prefix}_affordance.npy"
        np.save(affordance_path, affordance_map)
        
        # 保存角度地图
        angle_path = target_dir / f"{scene_prefix}_angles.npy"
        np.save(angle_path, angle_map)
        
        # 保存元数据
        metadata = {
            'scene_id': int(scene_id),
            'image_shape': [int(x) for x in rgb_image.shape[:2]],
            'num_candidates': len(results),
            'num_successful': sum(1 for r in results if r['success']),
            'success_rate': float(sum(1 for r in results if r['success']) / len(results)) if results else 0.0,
            'candidates': [{
                'success': bool(r['success']),
                'pixel': [int(r['pixel'][0]), int(r['pixel'][1])],
                'world_pos': [float(r['world_pos'][0]), float(r['world_pos'][1]), float(r['world_pos'][2])],
                'angle': float(r['angle'])
            } for r in results]
        }
        
        meta_path = target_dir / f"{scene_prefix}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"      💾 数据已保存: {scene_prefix}_*")
    
    def visualize_affordance_map(self, affordance_map, scene_id):
        """使用jet colormap可视化可供性地图"""
        plt.figure(figsize=(8, 6))
        plt.imshow(affordance_map, cmap='jet', vmin=0, vmax=1)
        plt.title(f'Affordance Map - Scene {scene_id:04d}')
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout()
        
        # 保存可视化图像
        vis_path = target_dir / f"scene_{scene_id:04d}_affordance_vis.png"
        plt.savefig(str(vis_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      📊 可供性地图可视化已保存: {vis_path}")
    
    def collect_dataset(self, num_scenes, num_objects_range=(2, 4), max_attempts_per_scene=25):
        """收集完整数据集"""
        print(f"\n🚀 开始收集数据集")
        print(f"   场景数量: {num_scenes}")
        print(f"   物体数量范围: {num_objects_range[0]}-{num_objects_range[1]}")
        print(f"   每场景最大尝试数: {max_attempts_per_scene}")
        print("=" * 60)
        
        successful_scenes = 0
        total_grasps = 0
        total_successful_grasps = 0
        
        for scene_id in range(num_scenes):
            # ✨ 每个场景都创建新的环境配置，确保场景多样性
            # 清理之前的环境
            if scene_id > 0:
                for obj_id in object_ids:
                    try:
                        p.removeBody(obj_id)
                    except:
                        pass
            
            # 随机选择物体数量并创建新环境
            num_objects = np.random.randint(num_objects_range[0], num_objects_range[1] + 1)
            
            # ✨ 为每个场景设置不同的随机种子，确保物体位置不同
            np.random.seed(scene_id + int(time.time()) % 1000)
            
            print(f"🏗️  创建新环境配置 #{scene_id + 1}: {num_objects} 个物体")
            robot_id, object_ids = setup_environment(num_objects=num_objects)
            if not object_ids:
                print(f"   ❌ 环境设置失败，跳过场景 {scene_id}")
                continue
            
            # 调试: 打印物体位置和机器人姿态
            print(f"🔍 场景 {scene_id} 环境设置调试:")
            robot_pos = p.getLinkState(robot_id, 8)[0]
            print(f"  📍 机器人末端位置: {robot_pos}")
            print(f"  📍 物体位置:")
            for i, obj_id in enumerate(object_ids):
                pos, orn = p.getBasePositionAndOrientation(obj_id)
                print(f"    物体 {i}: 位置={pos}, 朝向={orn}")
            
            # 收集单次抓取尝试作为独立场景
            print(f"   🔍 场景 {scene_id} 开始 - 物体IDs: {object_ids}")
            
            # 随机决定该场景属于训练集还是测试集
            is_train = np.random.random() < self.train_split
            target_dir = self.train_dir if is_train else self.test_dir
            split_name = "训练集" if is_train else "测试集"
            print(f"   📊 场景分配: {split_name} ({target_dir.name})")
            
            success = self.collect_single_attempt_as_scene(scene_id, robot_id, object_ids, target_dir)
            
            # 读取场景统计信息
            import json
            meta_file = target_dir / f"scene_{scene_id:04d}_meta.json"
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                    scene_attempts = meta.get('num_candidates', 0)
                    scene_successes = meta.get('num_successful', 0)
                    total_grasps += scene_attempts
                    total_successful_grasps += scene_successes
            
            # 场景间短暂停顿，让物理引擎稳定
            if scene_id < num_scenes - 1:
                for _ in range(10):
                    p.stepSimulation()
                    time.sleep(1./240.)
            
            if success:
                successful_scenes += 1
                print(f"   ✅ 场景 {scene_id} 完成 (有可用数据)")
            else:
                print(f"   ❌ 场景 {scene_id} 完成 (无可用数据)")
        
        # 总结
        grasp_success_rate = (total_successful_grasps / total_grasps) if total_grasps > 0 else 0
        print("=" * 60)
        print(f"🎉 数据收集完成!")
        print(f"   有效场景: {successful_scenes}/{num_scenes}")
        print(f"   总抓取成功率: {grasp_success_rate:.1%} ({total_successful_grasps}/{total_grasps})")
        print(f"   数据保存位置:")
        print(f"     训练集: {self.train_dir} ({self.train_split:.1%})")
        print(f"     测试集: {self.test_dir} ({1-self.train_split:.1%})")
        print("✅ 数据收集成功!")
        
        return successful_scenes > 0

def main():
    parser = argparse.ArgumentParser(description='自动化可供性训练数据收集器')
    parser.add_argument('--num_scenes', type=int, default=10, help='收集的场景数量')
    parser.add_argument('--num_objects', type=int, nargs=2, default=[2, 4], help='物体数量范围 [min, max]')
    parser.add_argument('--max_attempts', type=int, default=25, help='每场景最大抓取尝试数')
    parser.add_argument('--data_dir', type=str, default='data/affordance_v5', help='数据保存目录')
    parser.add_argument('--angles', type=int, default=8, help='离散抓取角度数量')
    parser.add_argument('--train_split', type=float, default=0.8, help='训练集比例 (0.0-1.0)')
    parser.add_argument('--gui', action='store_true', help='显示GUI')
    
    args = parser.parse_args()
    
    # 连接PyBullet
    if args.gui:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)
    
    try:
        # 创建收集器
        collector = AutoAffordanceCollector(
            data_dir=args.data_dir,
            num_angles=args.angles,
            train_split=args.train_split
        )
        
        # 收集数据
        success = collector.collect_dataset(
            num_scenes=args.num_scenes,
            num_objects_range=tuple(args.num_objects),
            max_attempts_per_scene=args.max_attempts
        )
        
        if success:
            print("✅ 数据收集成功!")
        else:
            print("❌ 数据收集失败!")
    
    finally:
        p.disconnect()

if __name__ == "__main__":
    main()