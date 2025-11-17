import os
import torch
from PIL import Image
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import pyiqa

# ===== Paths =====
mvsplat_root = "/data8/stilex/mvsplat_576_gap100"
diffusion_roots = {
    0: "/home/stilex/stable-virtual-camera/work_dirs/demo/gap100_21T",
    5: "/home/stilex/stable-virtual-camera/work_dirs/demo/gap100_addnoise/5",
    10: "/home/stilex/stable-virtual-camera/work_dirs/demo/gap100_addnoise/10",
    15: "/home/stilex/stable-virtual-camera/work_dirs/demo/gap100_addnoise/15",
    20: "/home/stilex/stable-virtual-camera/work_dirs/demo/gap100_addnoise/20",
    25: "/home/stilex/stable-virtual-camera/work_dirs/demo/gap100_addnoise/25",
    30: "/home/stilex/stable-virtual-camera/work_dirs/demo/gap100_addnoise/30",
    35: "/home/stilex/stable-virtual-camera/work_dirs/demo/gap100_addnoise/35",
    40: "/home/stilex/stable-virtual-camera/work_dirs/demo/gap100_addnoise/40",
    45: "/home/stilex/stable-virtual-camera/work_dirs/demo/gap100_addnoise/45",
}

# ===== Device & MUSIQ Metric =====
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
musiq_metric = pyiqa.create_metric('musiq', device=device)

# ===== Load helper =====
def load_image_tensor(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor

# ===== Storage =====
per_image_musiq = defaultdict(dict)
count = 0
# ===== Loop through scenes =====
for scene in tqdm(sorted(os.listdir(mvsplat_root))[1000:1050], desc="Processing scenes"):
    # if count ==1000:
    #     break
    # count += 1
    gt_dir = os.path.join(mvsplat_root, scene, "groundtruth")
    out_dir = os.path.join(mvsplat_root, scene, "output")
    context_dir = os.path.join(mvsplat_root, scene, "context")
    
    if not (os.path.isdir(gt_dir) and os.path.isdir(out_dir)):
        continue
    if not all(os.path.isdir(os.path.join(root, scene, "samples-rgb")) for root in diffusion_roots.values()):
        continue
    
    context_score = []
    for i in range(len(os.listdir(context_dir))):
        frame_name = f"{(i)*100:03d}.png" # 000, 100
        contxt_img_path = os.path.join(context_dir, frame_name)
        score = musiq_metric(contxt_img_path).item()
        context_score.append(score)
        
    gt_files = sorted(os.listdir(gt_dir))
    min_len_mvsplat = len(gt_files)

    # ==== MUSIQ for mvsplat ====
    for i in range(min_len_mvsplat):
        frame_name = f"{(i+1)*20:03d}.png"  # 020, 040, 060, 080...
        mvsplat_img_path = os.path.join(out_dir, frame_name)
        if not os.path.exists(mvsplat_img_path):
            continue

        row_key = f"{scene}/{i:03d}"
        score = musiq_metric(mvsplat_img_path).item()
        per_image_musiq[row_key]["mvsplat"] = score
        per_image_musiq[row_key]["context_000"] = context_score[0]
        per_image_musiq[row_key]["context_100"] = context_score[1]
        
    # ==== MUSIQ for each diffusion K ====
    for k, root in diffusion_roots.items():
        sample_dir = os.path.join(root, scene, "samples-rgb")
        pred_files = sorted(os.listdir(sample_dir))
        min_len = min(len(gt_files), len(pred_files))
        if min_len == 0:
            continue

        for i in range(min_len):
            pred_path = os.path.join(sample_dir, f"{i:03d}.png")
            if not os.path.exists(pred_path):
                continue

            row_key = f"{scene}/{i:03d}"
            score = musiq_metric(pred_path).item()
            per_image_musiq[row_key][k] = score

# ===== Save MUSIQ results =====
ks_sorted = ["mvsplat"] + ["context_000"] + ["context_100"] + sorted(diffusion_roots.keys())
df_musiq = pd.DataFrame.from_dict(per_image_musiq, orient="index")
df_musiq = df_musiq.reindex(columns=ks_sorted)
df_musiq.index.name = "scene/image"
df_musiq.sort_index().to_csv("musiq_per_image1_a1k1.csv")

print("Saved musiq_per_image1.csv")
