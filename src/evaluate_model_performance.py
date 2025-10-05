#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script for Affordance-Based Grasping

This script evaluates the affordance model's performance by:
1. Loading the trained model (affordance_model_best (copy).pth)
2. Running multiple grasping trials in the simula            # C            x, y, z = world_pos
            dist = np.sqrt(x**2 + y**2)
            
            # Debug: check if dist is scalar
            if hasattr(dist, '__len__') and len(dist) > 1:
                print(f"Warning: dist is array with shape {dist.shape}, values: {dist}")
                continue

            # Same workspace check as data generation
            if (dist >= 0.25 and dist <= 0.85 and abs(y) <= 0.5 and
                z >= TABLE_TOP_Z - 0.05 and z <= TABLE_TOP_Z + 0.15 and
                depth_image[v, u] > 0.01):  # Valid depth workspace validity
            depth_val = depth_image[v, u]
            world_pos = self.pixel_to_world(u, v, depth_val)
            
            # Debug: check world_pos shape
            if not isinstance(world_pos, np.ndarray) or world_pos.shape != (3,):
                print(f"Warning: unexpected world_pos shape: {world_pos.shape if hasattr(world_pos, 'shape') else type(world_pos)}")
                continue
                
            x, y, z = world_pos
            dist = np.sqrt(x**2 + y**2) environment
3. Computing detailed performance metrics
4. Providing visualizations of model predictions
5. Comparing model predictions with ground truth when available

Usage:
    python evaluate_model_performance.py --model_path models/affordance_model_best\ \(copy\).pth --num_trials 20 --visualize
"""

import os
import sys

# Ensure src/ parent is in sys.path for module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import pybullet as p
import pybullet_data
import time
import cv2
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Import the same functions used in data generation
from src.afford_data_gen import reset_robot_home, open_gripper_fast, fast_grasp_test
from src.environment_setup import setup_environment, update_object_states, reset_objects_after_grasp
from perception import set_topdown_camera, get_rgb_depth, pixel_to_world, CAMERA_PARAMS
from geom import TABLE_TOP_Z


class UNetLarge(nn.Module):
    """The same UNetLarge model used in training"""
    def __init__(self, in_channels=4, out_channels=37):
        super(UNetLarge, self).__init__()
        # 更宽更深的通道 (1.5x深度)
        self.enc1 = self.conv_block(in_channels, 96)    # 4 → 96
        self.enc2 = self.conv_block(96, 192)            # 96 → 192
        self.enc3 = self.conv_block(192, 384)           # 192 → 384
        self.enc4 = self.conv_block(384, 768)           # 384 → 768
        self.enc5 = self.conv_block(768, 1536)          # 768 → 1536 (新增第5层)

        self.pool = nn.MaxPool2d(2)

        # 对应的解码器
        self.dec4 = self.conv_block(1536, 768)          # 1536 → 768
        self.dec3 = self.conv_block(768, 384)           # 768 → 384
        self.dec2 = self.conv_block(384, 192)           # 384 → 192
        self.dec1 = self.conv_block(192, 96)            # 192 → 96

        self.upconv4 = nn.ConvTranspose2d(1536, 768, 2, stride=2)  # 1536 → 768
        self.upconv3 = nn.ConvTranspose2d(768, 384, 2, stride=2)   # 768 → 384
        self.upconv2 = nn.ConvTranspose2d(384, 192, 2, stride=2)   # 384 → 192
        self.upconv1 = nn.ConvTranspose2d(192, 96, 2, stride=2)    # 192 → 96

        self.final = nn.Conv2d(96, out_channels, 1)

        # Dropout for regularization (更强的正则化)
        self.dropout = nn.Dropout2d(0.15)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder with 5 levels (更深)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))  # 新增第5层

        # Decoder with 5 levels (对应解码)
        d4 = self.upconv4(e5)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d4 = self.dropout(d4)  # 更强的dropout

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d3 = self.dropout(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d2 = self.dropout(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return out


class ModelEvaluator:
    def __init__(self, model_path, output_dir="./evaluation_results"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load model
        self.model = UNetLarge().to(self.device)
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded model checkpoint with keys: {list(checkpoint.keys())}")
            else:
                self.model.load_state_dict(checkpoint)
                print("✅ Loaded model state dict directly")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

        self.model.eval()
        print(f"✅ Model loaded on {self.device}")

        self.num_angle_classes = 36
        self.physics_client = None
        self.robot_id = None

        # Metrics tracking
        self.metrics = {
            'trials': 0,
            'successes': 0,
            'affordance_predictions': [],
            'angle_predictions': [],
            'execution_times': [],
            'trial_details': []
        }

    def initialize_simulation(self, num_objects=3, visualize=False):
        """Initialize PyBullet simulation"""
        client_id = p.connect(p.GUI if visualize else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Use the same environment setup as data generation
        self.robot_id, object_ids = setup_environment(num_objects=num_objects)

        # Set up camera
        width, height, view_matrix, proj_matrix = set_topdown_camera()
        self.view_matrix = view_matrix
        self.proj_matrix = proj_matrix
        self.camera_width = width
        self.camera_height = height

        return client_id, object_ids

    def capture_scene(self):
        """Capture RGB-D scene"""
        rgb, depth = get_rgb_depth(self.camera_width, self.camera_height,
                                  self.view_matrix, self.proj_matrix)
        return rgb, depth

    def preprocess_input(self, rgb, depth):
        """Preprocess for model input - convert to relative depth like training data"""
        transform = transforms.ToTensor()
        rgb_tensor = transform(Image.fromarray(rgb))

        # Convert depth to relative depth (same as training data)
        relative_depth = self.convert_to_relative_depth(depth, self.view_matrix, self.proj_matrix)
        depth_tensor = torch.tensor(relative_depth).unsqueeze(0).float()

        x = torch.cat([rgb_tensor, depth_tensor], dim=0).unsqueeze(0).to(self.device)
        return x

    def convert_to_relative_depth(self, depth_image, view_matrix, proj_matrix):
        """Convert absolute depth to relative depth (same as training data)"""
        from geom import TABLE_TOP_Z

        height, width = depth_image.shape
        workspace_mask = np.zeros((height, width), dtype=bool)
        edge_mask = np.zeros((height, width), dtype=bool)

        # For each pixel, check if it's in workspace
        for v in range(0, height, 4):  # Sample every 4th pixel for efficiency
            for u in range(0, width, 4):
                # Convert pixel to world coordinates
                depth_val = depth_image[v, u]
                world_pos = self.pixel_to_world(u, v, depth_val)
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

        return relative_depth

    def infer_affordance(self, rgb, depth):
        """Run model inference"""
        x = self.preprocess_input(rgb, depth)

        with torch.no_grad():
            pred = self.model(x)

        affordance_logits = pred[0, 0]
        angle_logits = pred[0, 1:]

        affordance_prob = torch.sigmoid(affordance_logits).cpu().numpy()
        angle_pred = torch.argmax(angle_logits, dim=0).cpu().numpy()
        angle_degrees = angle_pred * (360.0 / self.num_angle_classes)

        return affordance_prob, angle_degrees

    def find_best_grasp_point(self, affordance_prob, angle_degrees, depth, num_candidates=20, exploration_rate=0.3):
        """Find best grasp point using model predictions with exploration"""
        height, width = affordance_prob.shape

        candidates = []

        # Sample candidates based on affordance probabilities
        for _ in range(num_candidates):
            if np.random.random() < 0.8:  # 80% from high affordance areas
                flat_affordance = affordance_prob.flatten()
                if flat_affordance.sum() > 0:
                    flat_affordance = flat_affordance / flat_affordance.sum()
                    flat_idx = np.random.choice(len(flat_affordance), p=flat_affordance)
                    v, u = np.unravel_index(flat_idx, affordance_prob.shape)
                else:
                    u = np.random.randint(10, width-10)
                    v = np.random.randint(10, height-10)
            else:  # 20% random
                u = np.random.randint(10, width-10)
                v = np.random.randint(10, height-10)

            # Check workspace validity
            world_pos = self.pixel_to_world(u, v, depth[v, u])
            x, y, z = world_pos
            dist = np.sqrt(x**2 + y**2)

            # Same workspace check as data generation
            if (dist >= 0.25 and dist <= 0.85 and abs(y) <= 0.5 and
                z >= TABLE_TOP_Z - 0.05 and z <= TABLE_TOP_Z + 0.15 and
                depth[v, u] > 0.01):  # Valid depth

                affordance_val = affordance_prob[v, u]
                angle_val = angle_degrees[v, u] if v < angle_degrees.shape[0] and u < angle_degrees.shape[1] else 0.0

                candidates.append({
                    'u': u, 'v': v,
                    'world_pos': world_pos,
                    'affordance': affordance_val,
                    'angle': angle_val,
                    'distance': dist
                })

        # Sort by affordance value
        if candidates:
            candidates.sort(key=lambda x: x['affordance'], reverse=True)

            # Add exploration: sometimes pick from top candidates instead of just the best
            if len(candidates) > 1 and np.random.random() < exploration_rate:
                # Pick randomly from top 3 candidates (or all if less than 3)
                top_k = min(3, len(candidates))
                selected_idx = np.random.randint(0, top_k)
                best = candidates[selected_idx]
                print(f"🎲 Exploration: picked candidate {selected_idx+1}/{top_k} (affordance: {best['affordance']:.3f})")
            else:
                best = candidates[0]

            # Debug: show top candidates
            print(f"🏆 Top candidates:")
            for i, cand in enumerate(candidates[:5]):  # Show top 5
                marker = " ← SELECTED" if (cand['u'] == best['u'] and cand['v'] == best['v']) else ""
                print(f"  {i+1}. Affordance: {cand['affordance']:.3f}, Pos: ({cand['u']},{cand['v']}), Dist: {cand['distance']:.3f}{marker}")

            return best['u'], best['v'], best['angle'], best['affordance'], candidates

        # Fallback
        u, v = width // 2, height // 2
        world_pos = self.pixel_to_world(u, v, depth[v, u])
        angle = angle_degrees[v, u] if v < angle_degrees.shape[0] and u < angle_degrees.shape[1] else 0.0
        affordance_value = affordance_prob[v, u]

        return u, v, angle, affordance_value, []

    def pixel_to_world(self, u, v, depth_val):
        """Convert pixel to world coordinates"""
        return pixel_to_world(u, v, depth_val, self.view_matrix, self.proj_matrix)

    def analyze_object_selection(self, grasp_pos, object_ids, trial_num):
        """Analyze which object is being targeted and track patterns"""
        if not hasattr(self, 'object_selection_history'):
            self.object_selection_history = []

        # Find closest object to grasp position
        min_dist = float('inf')
        target_object_id = None
        target_pos = None

        for obj_id in object_ids:
            try:
                pos, _ = p.getBasePositionAndOrientation(obj_id)
                dist = np.sqrt((pos[0] - grasp_pos[0])**2 + (pos[1] - grasp_pos[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    target_object_id = obj_id
                    target_pos = pos
            except:
                continue

        if target_object_id is not None:
            selection_record = {
                'trial': trial_num,
                'object_id': target_object_id,
                'distance': min_dist,
                'object_pos': target_pos,
                'grasp_pos': grasp_pos
            }
            self.object_selection_history.append(selection_record)

            # Count frequency of object selection
            from collections import Counter
            recent_selections = [r['object_id'] for r in self.object_selection_history[-10:]]  # Last 10 trials
            selection_counts = Counter(recent_selections)

            most_common = selection_counts.most_common(1)[0] if selection_counts else (None, 0)

            print(f"🎯 Targeting object {target_object_id} (distance: {min_dist:.3f}m)")
            if most_common[1] > 1:
                print(f"📊 Object selection pattern: {most_common[0]} selected {most_common[1]}/10 recent trials")

    def create_visualization(self, rgb, depth, affordance_prob, angle_degrees, best_u=None, best_v=None,
                           trial_num=None, save_path=None):
        """Create visualization of model predictions"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original RGB
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title('RGB Image')
        axes[0, 0].axis('off')

        # Depth map
        depth_vis = axes[0, 1].imshow(depth, cmap='plasma')
        axes[0, 1].set_title('Depth Map')
        axes[0, 1].axis('off')
        plt.colorbar(depth_vis, ax=axes[0, 1], shrink=0.8)

        # Affordance probability
        afford_vis = axes[0, 2].imshow(affordance_prob, cmap='hot', vmin=0, vmax=1)
        axes[0, 2].set_title('Affordance Probability')
        axes[0, 2].axis('off')
        plt.colorbar(afford_vis, ax=axes[0, 2], shrink=0.8)

        # Angle predictions
        angle_vis = axes[1, 0].imshow(angle_degrees, cmap='hsv', vmin=0, vmax=360)
        axes[1, 0].set_title('Predicted Angles (degrees)')
        axes[1, 0].axis('off')
        plt.colorbar(angle_vis, ax=axes[1, 0], shrink=0.8)

        # Overlay affordance on RGB
        rgb_with_affordance = rgb.copy()
        affordance_overlay = (affordance_prob > 0.5).astype(np.uint8) * 255
        rgb_with_affordance[:, :, 0] = np.maximum(rgb_with_affordance[:, :, 0], affordance_overlay)  # Red channel
        axes[1, 1].imshow(rgb_with_affordance)
        axes[1, 1].set_title('RGB with Affordance Overlay (Red > 0.5)')
        axes[1, 1].axis('off')

        # Best grasp point
        rgb_with_point = rgb.copy()
        if best_u is not None and best_v is not None:
            # Draw point
            cv2.circle(rgb_with_point, (best_u, best_v), 5, (0, 255, 0), -1)
            # Draw angle indicator
            angle_rad = np.radians(angle_degrees[best_v, best_u])
            end_u = int(best_u + 20 * np.cos(angle_rad))
            end_v = int(best_v + 20 * np.sin(angle_rad))
            cv2.arrowedLine(rgb_with_point, (best_u, best_v), (end_u, end_v), (255, 0, 0), 2)

        axes[1, 2].imshow(rgb_with_point)
        axes[1, 2].set_title('Best Grasp Point (Green) + Angle (Blue Arrow)')
        axes[1, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved visualization: {save_path}")

        return fig

    def run_evaluation_trial(self, trial_num, object_ids, visualize=False):
        """Run a single evaluation trial"""
        print(f"\n{'='*60}")
        print(f"TRIAL {trial_num}")
        print(f"{'='*60}")

        start_time = time.time()

        # Reset robot to home
        print("🏠 Resetting robot to home...")
        reset_robot_home(self.robot_id)
        for _ in range(120):
            p.stepSimulation()

        # Update object states
        print("🔄 Updating object states...")
        object_ids = update_object_states(object_ids)

        # If no objects left, generate new ones
        if len(object_ids) == 0:
            print("⚠️ No objects remaining, generating new ones...")
            object_ids = reset_objects_after_grasp([], min_objects=3)
            for _ in range(120):
                p.stepSimulation()

        # Capture scene
        print("📸 Capturing scene...")
        rgb, depth = self.capture_scene()

        # Run inference
        print("🧠 Running model inference...")
        affordance_prob, angle_degrees = self.infer_affordance(rgb, depth)

        # Track prediction statistics
        self.metrics['affordance_predictions'].extend(affordance_prob.flatten())
        self.metrics['angle_predictions'].extend(angle_degrees.flatten())

        # Find best grasp point
        print("🎯 Finding best grasp point...")
        best_u, best_v, angle, affordance_value, candidates = self.find_best_grasp_point(
            affordance_prob, angle_degrees, depth)

        print(f"Best grasp: pixel({best_u},{best_v}), angle={angle:.1f}°, affordance={affordance_value:.3f}")
        print(f"Found {len(candidates)} valid candidates")

        # Convert to world coordinates
        world_pos = self.pixel_to_world(best_u, best_v, depth[best_v, best_u])
        print(f"World position: [{world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}]")

        # Analyze object selection patterns
        self.analyze_object_selection(world_pos, object_ids, trial_num)

        # Create visualization
        if visualize:
            vis_path = self.output_dir / f"trial_{trial_num:02d}_prediction.png"
            self.create_visualization(rgb, depth, affordance_prob, angle_degrees,
                                    best_u, best_v, trial_num, vis_path)

        # Execute grasp using the same logic as data generation
        print("🤏 Executing grasp...")
        success = fast_grasp_test(self.robot_id, world_pos, np.radians(angle), object_ids,
                                visualize=visualize, debug_mode=False)

        execution_time = time.time() - start_time
        self.metrics['execution_times'].append(execution_time)

        # Update metrics
        self.metrics['trials'] += 1
        if success:
            self.metrics['successes'] += 1

        # Store trial details
        trial_detail = {
            'trial_num': trial_num,
            'success': success,
            'best_affordance': float(affordance_value),
            'best_angle': float(angle),
            'world_pos': [float(x) for x in world_pos],
            'pixel_pos': [int(best_u), int(best_v)],
            'num_candidates': len(candidates),
            'execution_time': execution_time,
            'num_objects': len(object_ids)
        }
        self.metrics['trial_details'].append(trial_detail)

        print(f"Trial {trial_num} result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        print(f"Execution time: {execution_time:.2f}s")

        # Clean up objects after grasp
        print("🧹 Cleaning up objects...")
        object_ids = reset_objects_after_grasp(object_ids)

        return object_ids

    def compute_statistics(self):
        """Compute comprehensive statistics"""
        if self.metrics['trials'] == 0:
            return {}

        stats = {
            'total_trials': self.metrics['trials'],
            'successes': self.metrics['successes'],
            'success_rate': self.metrics['successes'] / self.metrics['trials'] * 100,
            'avg_execution_time': np.mean(self.metrics['execution_times']),
            'std_execution_time': np.std(self.metrics['execution_times']),
            'affordance_stats': {
                'mean': np.mean(self.metrics['affordance_predictions']),
                'std': np.std(self.metrics['affordance_predictions']),
                'min': np.min(self.metrics['affordance_predictions']),
                'max': np.max(self.metrics['affordance_predictions']),
                'percentile_95': np.percentile(self.metrics['affordance_predictions'], 95)
            },
            'angle_stats': {
                'mean': np.mean(self.metrics['angle_predictions']),
                'std': np.std(self.metrics['angle_predictions']),
                'min': np.min(self.metrics['angle_predictions']),
                'max': np.max(self.metrics['angle_predictions'])
            }
        }

        # Add object selection analysis if available
        if hasattr(self, 'object_selection_history') and self.object_selection_history:
            from collections import Counter
            all_selections = [r['object_id'] for r in self.object_selection_history]
            selection_counts = Counter(all_selections)

            stats['object_selection'] = {
                'total_unique_objects': len(selection_counts),
                'most_selected_object': selection_counts.most_common(1)[0][0] if selection_counts else None,
                'most_selected_count': selection_counts.most_common(1)[0][1] if selection_counts else 0,
                'selection_distribution': dict(selection_counts.most_common())
            }

        return stats

    def save_results(self):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"evaluation_results_{timestamp}.json"

        results = {
            'timestamp': timestamp,
            'metrics': self.metrics,
            'statistics': self.compute_statistics()
        }

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"💾 Results saved to: {results_file}")
        return results_file

    def print_summary(self):
        """Print evaluation summary"""
        stats = self.compute_statistics()

        if not stats:
            print("❌ No statistics available - evaluation may have failed")
            return

        print(f"\n{'='*80}")
        print("📊 EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total Trials: {stats['total_trials']}")
        print(f"Successes: {stats['successes']}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print(f"Average Execution Time: {stats['avg_execution_time']:.2f}s ± {stats['std_execution_time']:.2f}s")
        print()
        print("Affordance Predictions:")
        print(".3f")
        print(".3f")
        print(".3f")
        print(".3f")
        print(".3f")
        print()
        print("Angle Predictions:")
        print(".1f")
        print(".1f")
        print(".1f")
        print(".1f")

        # Print object selection analysis
        if 'object_selection' in stats:
            obj_sel = stats['object_selection']
            print()
            print("Object Selection Analysis:")
            print(f"  Unique objects targeted: {obj_sel['total_unique_objects']}")
            if obj_sel['most_selected_object'] is not None:
                print(f"  Most selected object: {obj_sel['most_selected_object']} ({obj_sel['most_selected_count']} times)")
                print(f"  Selection distribution: {obj_sel['selection_distribution']}")

        print(f"{'='*80}")

    def run_evaluation(self, num_trials=10, num_objects=3, visualize=False):
        """Run complete evaluation"""
        print("🚀 Starting comprehensive model evaluation...")
        print(f"Model: affordance_model_best (copy).pth")
        print(f"Trials: {num_trials}")
        print(f"Objects per trial: {num_objects}")
        print(f"Visualization: {'Enabled' if visualize else 'Disabled'}")
        print(f"Output directory: {self.output_dir}")

        # Initialize simulation
        client_id, object_ids = self.initialize_simulation(num_objects=num_objects, visualize=visualize)

        try:
            for trial in range(1, num_trials + 1):
                object_ids = self.run_evaluation_trial(trial, object_ids, visualize)

                # Progress update
                if trial % 5 == 0:
                    current_success_rate = self.metrics['successes'] / self.metrics['trials'] * 100
                    print(f"\n📈 Progress: {trial}/{num_trials} trials completed ({current_success_rate:.1f}% success rate)")

        finally:
            # Save results
            results_file = self.save_results()

            # Print summary
            self.print_summary()

            # Disconnect
            if client_id is not None:
                p.disconnect()

        return results_file


def main():
    parser = argparse.ArgumentParser(description="Evaluate affordance model performance")
    parser.add_argument("--model_path", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "affordance_model_best (copy).pth"),
                       help="Path to the trained model")
    parser.add_argument("--num_trials", type=int, default=20,
                       help="Number of evaluation trials")
    parser.add_argument("--num_objects", type=int, default=3,
                       help="Number of objects per trial")
    parser.add_argument("--visualize", action="store_true",
                       help="Enable visualization of predictions")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Output directory for results")

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return

    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(args.model_path, args.output_dir)
    results_file = evaluator.run_evaluation(
        num_trials=args.num_trials,
        num_objects=args.num_objects,
        visualize=args.visualize
    )

    print(f"\n✅ Evaluation complete! Results saved to: {results_file}")


if __name__ == '__main__':
    main()