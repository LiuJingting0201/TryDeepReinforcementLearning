import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Set up absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
scene_dir = os.path.join(project_root, "data", "affordance_v5")
save_dir = os.path.join(script_dir, "vis")
os.makedirs(save_dir, exist_ok=True)

# Recursively find all *_rgb.png in train and test subfolders
rgb_files = []
for subfolder in ["train", "test"]:
    subdir = os.path.join(scene_dir, subfolder)
    if os.path.exists(subdir):
        for f in os.listdir(subdir):
            if f.endswith("_rgb.png"):
                rgb_files.append(os.path.join(subdir, f))

prefixes = sorted([
    os.path.splitext(f)[0].replace("_rgb", "")
    for f in rgb_files
])

print(f"发现 {len(prefixes)} 个样本")


N = min(10, len(prefixes))
for i in range(N):
    prefix = prefixes[i]
    rgb = np.array(Image.open(f"{prefix}_rgb.png"))
    depth = np.load(f"{prefix}_depth.npy")
    afford = np.load(f"{prefix}_affordance.npy")
    angle = np.load(f"{prefix}_angles.npy")

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].imshow(rgb)
    axs[0, 0].set_title("RGB Image")
    
    axs[0, 1].imshow(depth, cmap="viridis")
    axs[0, 1].set_title("Depth")

    axs[1, 0].imshow(rgb)
    axs[1, 0].imshow(afford, cmap="jet", alpha=0.6)
    axs[1, 0].set_title("Affordance Overlay")

    axs[1, 1].imshow(angle, cmap="hsv")
    axs[1, 1].set_title("Grasp Angle")

    fig.suptitle(f"{prefix}", fontsize=12)
    plt.tight_layout()
    
    # 保存图像
    out_path = os.path.join(save_dir, os.path.basename(prefix) + "_vis.png")
    plt.savefig(out_path)
    plt.close()

print(f"✅ 图像已保存到 {save_dir}/")
