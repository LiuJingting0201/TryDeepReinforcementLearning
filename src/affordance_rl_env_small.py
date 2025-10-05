# affordance_rl_env_small.py
import gymnasium as gym
import numpy as np
import torch
import pybullet as p
import time
import json
import os
from pathlib import Path
from PIL import Image

from perception import set_topdown_camera, get_rgb_depth_segmentation, pixel_to_world
from environment_setup import setup_environment, reset_objects_after_grasp
from train_affordance import UNetLarge  # 加载训练好的模型
from afford_data_gen import reset_robot_home, fast_grasp_test
from geom import TABLE_TOP_Z


class AffordanceRLSmallEnv(gym.Env):
    """基于深度学习可供性模型的强化学习抓取环境"""
    
    DATA_DIR = Path(__file__).parent.parent / "data" / "affordance_v5" / "train"
    def __init__(self, model_path, k_candidates=5, gui=False):
        super().__init__()
        self.k = k_candidates
        self.gui = gui
        p.connect(p.GUI if gui else p.DIRECT)
        p.setTimeStep(1. / 240.)

        # 创建环境（机器人和桌子）
        self.robot_id, self.object_ids = setup_environment(num_objects=2)
        for _ in range(240):
            p.stepSimulation()

        # 加载模型
        self.model = UNetLarge().cuda().eval()
        ckpt = torch.load(model_path)
        self.model.load_state_dict(ckpt["model_state_dict"])
        print(f"✅ Loaded affordance model from {model_path}")

        # 观测空间：每个候选点4维特征(u_norm, v_norm, affordance, angle_norm)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(self.k * 4,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.k)

    def convert_to_relative_depth(self, depth_image, view_matrix, proj_matrix):
        """Convert absolute depth to relative depth (same as training data)"""
        height, width = depth_image.shape
        workspace_mask = np.zeros((height, width), dtype=bool)
        edge_mask = np.zeros((height, width), dtype=bool)

        # For each pixel, check if it's in workspace
        for v in range(0, height, 4):  # Sample every 4th pixel for efficiency
            for u in range(0, width, 4):
                # Convert pixel to world coordinates
                depth_val = depth_image[v, u]
                world_pos = pixel_to_world(u, v, depth_val, view_matrix, proj_matrix)
                x, y, z = world_pos

                # Check if in workspace
                dist_from_base = np.sqrt(x**2 + y**2)
                in_workspace = (
                    z >= TABLE_TOP_Z and z <= TABLE_TOP_Z + 0.25 and
                    dist_from_base >= 0.35 and dist_from_base <= 0.80 and
                    abs(y) <= 0.30
                )

                workspace_mask[v, u] = in_workspace

                # Define edge region for table depth estimation
                if in_workspace and (dist_from_base <= 0.45 or dist_from_base >= 0.70):
                    edge_mask[v, u] = True

        # Calculate table depth from edge pixels
        edge_depths = depth_image[edge_mask]

        if len(edge_depths) == 0:
            edge_depths = depth_image[workspace_mask]
            if len(edge_depths) == 0:
                edge_depths = depth_image.flatten()

        # Use lowest depth pixels as table
        sorted_depths = np.sort(edge_depths)
        num_table_candidates = max(50, len(sorted_depths) // 5)
        table_pixels = sorted_depths[:num_table_candidates]
        table_depth = np.percentile(table_pixels, 10)

        # Create relative depth
        relative_depth = np.zeros_like(depth_image)
        relative_depth[workspace_mask] = depth_image[workspace_mask] - table_depth
        # Pixels outside workspace stay 0 (at table level)

        return relative_depth, table_depth

    def load_scene(self, scene_id):
        """Load a pre-generated scene from the training data"""
        print(f"Loading scene {scene_id}")
        rgb_path = self.DATA_DIR / f"scene_{scene_id:04d}_rgb.png"
        depth_path = self.DATA_DIR / f"scene_{scene_id:04d}_depth.npy"
        meta_path = self.DATA_DIR / f"scene_{scene_id:04d}_meta.json"
        
        # Load RGB image
        rgb = np.array(Image.open(rgb_path))
        
        # Load depth (absolute depth from data)
        depth = np.load(depth_path)
        print(f"Loaded depth max: {np.max(depth)}, min: {np.min(depth)}")
        
        # Load metadata
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # Get object positions from candidates (limit to 3 objects)
        num_objects = min(3, len(meta['candidates']))
        selected_candidates = np.random.choice(meta['candidates'], num_objects, replace=False)
        object_positions = [c['world_pos'] for c in selected_candidates]
        
        # Remove existing objects
        for obj_id in self.object_ids:
            try:
                p.removeBody(obj_id)
            except:
                pass
        self.object_ids = []
        
        # Create objects at the positions
        for pos in object_positions:
            # Create a simple box object
            half_extents = [0.02, 0.02, 0.02]
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[1, 0, 0, 1])
            body = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=pos
            )
            p.changeDynamics(body, -1, lateralFriction=1.5, restitution=0.2)
            self.object_ids.append(body)
        
        # Wait for objects to settle
        for _ in range(60):
            p.stepSimulation()
        
        return rgb, depth

    def _sample_candidates_and_get_obs(self, afford_prob, angle_degrees, depth, view, proj, H, W):
        """Sample candidates and return observation"""
        # ------------------ 基于模型输出采样候选 (like evaluation script) ------------------
        # Use full mask since scenes are clean (no background filtering needed)
        mask = np.ones((H, W), dtype=np.uint8)
        masked_afford = afford_prob * mask
        flat_afford = masked_afford.flatten()
        
        # Sample many candidates and pick the top k by affordance (like evaluation does for top 1)
        num_samples = 50  # Increased from 20
        candidates = []
        
        for _ in range(num_samples):
            if np.random.random() < 0.2:  # 20% from high affordance areas
                if flat_afford.sum() > 0:
                    # Sample proportionally to affordance values, but add noise to prevent domination
                    flat_afford_norm = flat_afford / flat_afford.sum()
                    # Add uniform noise to flatten the distribution significantly
                    noise_level = 0.4  # Reduced from 0.7 for more focused sampling
                    noise = np.full_like(flat_afford_norm, 1.0 / len(flat_afford_norm))
                    flat_afford_norm = (1 - noise_level) * flat_afford_norm + noise_level * noise
                    flat_afford_norm = flat_afford_norm / flat_afford_norm.sum()
                    flat_idx = np.random.choice(len(flat_afford), p=flat_afford_norm)
                    v, u = np.unravel_index(flat_idx, masked_afford.shape)
                else:
                    u = np.random.randint(10, W-10)
                    v = np.random.randint(10, H-10)
            else:  # 80% random
                u = np.random.randint(10, W-10)
                v = np.random.randint(10, H-10)

            # Check workspace validity (same as evaluation script)
            world_pos = pixel_to_world(u, v, depth[v, u], view, proj)
            x, y, z = world_pos
            dist = np.sqrt(x**2 + y**2)

            if (dist >= 0.25 and dist <= 0.80 and abs(y) <= 0.30 and
                z >= TABLE_TOP_Z - 0.05 and z <= TABLE_TOP_Z + 0.15 and
                depth[v, u] > 0.01):  # Valid depth

                affordance_val = afford_prob[v, u]
                angle_val = angle_degrees[v, u] if v < angle_degrees.shape[0] and u < angle_degrees.shape[1] else 0.0

                candidates.append({
                    'u': u, 'v': v,
                    'affordance': affordance_val,
                    'angle': angle_val,
                    'world_pos': world_pos
                })

        # Sort by affordance and take top k (with deduplication)
        if len(candidates) >= self.k:
            candidates.sort(key=lambda x: x['affordance'], reverse=True)
            
            # Deduplicate by pixel position (keep highest affordance for each pixel)
            seen_pixels = set()
            deduped_candidates = []
            for c in candidates:
                pixel_key = (c['u'], c['v'])
                if pixel_key not in seen_pixels:
                    seen_pixels.add(pixel_key)
                    deduped_candidates.append(c)
                    if len(deduped_candidates) >= self.k:
                        break
            
            candidates = deduped_candidates
        else:
            # Fallback: ensure we have exactly k candidates
            print(f"Warning: Only found {len(candidates)} valid candidates, padding to {self.k}")
            candidates.sort(key=lambda x: x['affordance'], reverse=True)
            # Add fallback candidates if needed
            while len(candidates) < self.k:
                u, v = W//2 + np.random.randint(-20, 20), H//2 + np.random.randint(-20, 20)
                world_pos = pixel_to_world(u, v, depth[v, u], view, proj)
                x, y, z = world_pos
                dist = np.sqrt(x**2 + y**2)

                if (dist >= 0.25 and dist <= 0.80 and abs(y) <= 0.30 and
                    z >= TABLE_TOP_Z - 0.05 and z <= TABLE_TOP_Z + 0.15 and
                    depth[v, u] > 0.01):  # Valid depth
                    angle_val = angle_degrees[v, u] if v < angle_degrees.shape[0] and u < angle_degrees.shape[1] else 0.0
                    candidates.append({
                        'u': u, 'v': v,
                        'affordance': afford_prob[v, u],
                        'angle': angle_val,
                        'world_pos': world_pos
                    })

        self.candidates = [(c['u'], c['v'], np.radians(c['angle'])) for c in candidates]

        # 观测：每个候选点的归一化坐标 + 置信度 + 角度
        obs = np.array([[c['u'] / W, c['v'] / H, c['affordance'], c['angle'] / 360.0] for c in candidates], dtype=np.float32).flatten()
        
        return obs


    def reset(self, *, seed=None, options=None):
        # Generate new scene
        for obj_id in self.object_ids:
            try:
                p.removeBody(obj_id)
            except:
                pass
        self.object_ids = []
        num_objects = np.random.randint(2, 5)
        self.object_ids = reset_objects_after_grasp(self.object_ids, min_objects=num_objects)

        reset_robot_home(self.robot_id)
        for _ in range(60):
            p.stepSimulation()

        # Get camera parameters (same as data generation)
        _, _, view, proj = set_topdown_camera()

        # 计算相对深度 (与训练数据一致)
        relative_depth = self.convert_to_relative_depth(np.zeros((224, 224)), view, proj)  # Placeholder, will capture real depth

        # 拍摄图像
        W, H, view, proj = set_topdown_camera()
        rgb, depth, seg = get_rgb_depth_segmentation(W, H, view, proj)

        # 计算相对深度 (与训练数据一致)
        relative_depth, table_depth = self.convert_to_relative_depth(depth, view, proj)
        self.table_depth = table_depth

        # 模型推理
        rgb_t = torch.tensor(rgb.transpose(2, 0, 1)).unsqueeze(0).float().cuda() / 255.0
        depth_t = torch.tensor(relative_depth).unsqueeze(0).unsqueeze(0).float().cuda()
        x = torch.cat([rgb_t, depth_t], dim=1)
        with torch.no_grad():
            out = self.model(x)
            afford_logits = out[0, 0]
            angle_logits = out[0, 1:]
            afford_prob = torch.sigmoid(afford_logits).cpu().numpy()
            angle_pred = torch.argmax(angle_logits, dim=0).cpu().numpy()
            angle_degrees = angle_pred * (360.0 / 36)  # 36 classes to degrees

        # Store for later use
        self.afford_prob = afford_prob
        self.angle_degrees = angle_degrees
        self.depth = depth
        self.view = view
        self.proj = proj
        self.H = H
        self.W = W
        self.relative_depth = relative_depth

        print(f"Model max afford_prob: {np.max(afford_prob)}")
        max_pos = np.unravel_index(np.argmax(afford_prob), afford_prob.shape)
        print(f"Max afford at pixel {max_pos}, relative_depth there: {relative_depth[max_pos[0], max_pos[1]]}")

        obs = self._sample_candidates_and_get_obs(afford_prob, angle_degrees, depth, view, proj, H, W)

        # Debug: print candidate info and check object proximity
        print(f"🔍 RL Debug:")
        print("  Object positions:")
        for i, obj_id in enumerate(self.object_ids):
            try:
                pos, _ = p.getBasePositionAndOrientation(obj_id)
                print(f"    Object {i}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            except:
                continue
        
        # Check model's max affordance prediction
        max_afford = np.max(afford_prob)
        max_pos = np.unravel_index(np.argmax(afford_prob), afford_prob.shape)
        max_world_pos = pixel_to_world(max_pos[1], max_pos[0], self.relative_depth[max_pos[0], max_pos[1]] + self.table_depth, view, proj)
        print(f"  Model max affordance: {max_afford:.3f} at pixel{max_pos} -> world({max_world_pos[0]:.3f},{max_world_pos[1]:.3f},{max_world_pos[2]:.3f})")
        
        for i, c in enumerate(self.candidates):
            world_pos = pixel_to_world(c[0], c[1], self.depth[c[1], c[0]], view, proj)
            print(f"  Candidate {i}: pixel({c[0]},{c[1]}) -> world({world_pos[0]:.3f},{world_pos[1]:.3f},{world_pos[2]:.3f})")
            
            # Check proximity to objects
            min_dist = float('inf')
            closest_obj_pos = None
            for obj_id in self.object_ids:
                try:
                    pos, _ = p.getBasePositionAndOrientation(obj_id)
                    dist = np.sqrt((pos[0] - world_pos[0])**2 + (pos[1] - world_pos[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_obj_pos = pos
                except:
                    continue
            print(f"    Closest object: ({closest_obj_pos[0]:.3f}, {closest_obj_pos[1]:.3f}, {closest_obj_pos[2]:.3f}), distance: {min_dist:.3f}m")
        print("🔍 End Debug")

        return obs, {}


    # -------------------- Step --------------------
    def step(self, action):
        # Reset robot to home position before recapturing scene
        reset_robot_home(self.robot_id)
        for _ in range(60):
            p.stepSimulation()

        # Recapture depth with arm out of view
        W, H, view, proj = set_topdown_camera()
        rgb, depth, seg = get_rgb_depth_segmentation(W, H, view, proj)
        relative_depth, table_depth = self.convert_to_relative_depth(depth, view, proj)
        self.depth = depth
        self.relative_depth = relative_depth
        self.table_depth = table_depth
        # Re-infer model with updated depth
        rgb_t = torch.tensor(rgb.transpose(2, 0, 1)).unsqueeze(0).float().cuda() / 255.0
        depth_t = torch.tensor(relative_depth).unsqueeze(0).unsqueeze(0).float().cuda()
        x = torch.cat([rgb_t, depth_t], dim=1)
        with torch.no_grad():
            out = self.model(x)
            afford_logits = out[0, 0]
            angle_logits = out[0, 1:]
            afford_prob = torch.sigmoid(afford_logits).cpu().numpy()
            angle_pred = torch.argmax(angle_logits, dim=0).cpu().numpy()
            angle_degrees = angle_pred * (360.0 / 36)
        self.afford_prob = afford_prob
        self.angle_degrees = angle_degrees

        u, v, theta = self.candidates[action]
        H, W = self.relative_depth.shape
        world_pos = pixel_to_world(u, v, self.relative_depth[v, u] + self.table_depth, self.view, self.proj)
        # Use the actual depth-based z-height instead of fixed table height
        # world_pos[2] = 0.625  # Remove this fixed height override

        print(f"🎯 [RL] trying candidate {action}: ({u},{v}), θ={np.degrees(theta):.1f}°")
        print(f"    World pos: ({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f})")
        
        time.sleep(0.3)
        success = fast_grasp_test(self.robot_id, world_pos, theta, self.object_ids, visualize=False)

        reward = 1.0 if success else 0.0
        if success:
            done = True
            obs = np.zeros(self.k * 4, dtype=np.float32)
        else:
            done = False
            # Update object states, remove fallen objects
            from environment_setup import update_object_states
            old_len = len(self.object_ids)
            self.object_ids = update_object_states(self.object_ids)
            if len(self.object_ids) == 0:
                # No valid objects left, create new scene
                num_objects = np.random.randint(2, 5)
                self.object_ids = reset_objects_after_grasp([], min_objects=num_objects)
                # Need to recapture scene and re-infer since objects changed
                reset_robot_home(self.robot_id)
                for _ in range(60):
                    p.stepSimulation()

                # Get camera parameters (same as data generation)
                _, _, view, proj = set_topdown_camera()

                # 计算相对深度 (与训练数据一致)
                relative_depth = self.convert_to_relative_depth(np.zeros((224, 224)), view, proj)  # Placeholder, will capture real depth

                # 拍摄图像
                W, H, view, proj = set_topdown_camera()
                rgb, depth, seg = get_rgb_depth_segmentation(W, H, view, proj)

                # 计算相对深度 (与训练数据一致)
                relative_depth, table_depth = self.convert_to_relative_depth(depth, view, proj)
                self.table_depth = table_depth

                # 模型推理
                rgb_t = torch.tensor(rgb.transpose(2, 0, 1)).unsqueeze(0).float().cuda() / 255.0
                depth_t = torch.tensor(relative_depth).unsqueeze(0).unsqueeze(0).float().cuda()
                x = torch.cat([rgb_t, depth_t], dim=1)
                with torch.no_grad():
                    out = self.model(x)
                    afford_logits = out[0, 0]
                    angle_logits = out[0, 1:]
                    afford_prob = torch.sigmoid(afford_logits).cpu().numpy()
                    angle_pred = torch.argmax(angle_logits, dim=0).cpu().numpy()
                    angle_degrees = angle_pred * (360.0 / 36)  # 36 classes to degrees

                # Store for later use
                self.afford_prob = afford_prob
                self.angle_degrees = angle_degrees
                self.depth = depth
                self.view = view
                self.proj = proj
                self.H = H
                self.W = W
                self.relative_depth = relative_depth
            obs = self._sample_candidates_and_get_obs(self.afford_prob, self.angle_degrees, self.depth, self.view, self.proj, self.H, self.W)

        info = {"success": success}
        return obs, reward, done, False, info
