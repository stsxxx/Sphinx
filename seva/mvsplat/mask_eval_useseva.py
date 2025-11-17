import torch
import json
from pathlib import Path
from src.model.model_wrapper import ModelWrapper
from src.config import load_typed_root_config
from src.global_cfg import set_cfg
import hydra
from tqdm import tqdm
from omegaconf import OmegaConf
from jaxtyping import install_import_hook
import numpy as np
import statistics
from PIL import Image
import torchvision.transforms as transforms
from src.evaluation.metrics import compute_psnr, compute_ssim, compute_lpips
from src.misc.image_io import save_image
import open3d as o3d
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper

def save_gaussians_to_ply(gaussians, filename="gaussians.ply"):
    means = gaussians.means[0].cpu().numpy()           # [N, 3]
    harmonics = gaussians.harmonics[0].cpu().numpy()   # [N, 3, 25]
    opacities = gaussians.opacities[0].cpu().numpy()   # [N]

    # Use SH(0) (DC component) as base RGB
    colors = harmonics[:, :, 0]  # [N, 3]
    colors = np.clip(colors, 0.0, 1.0)
    colors *= opacities[:, None]

    # Build point cloud

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(means)
    pc.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pc)
    print(f"Saved Gaussian point cloud to {filename}")


def load_custom_example(json_path, image_folder, seva_dir, context_indices, target_indices):
    """Load a single example from your custom data format"""
    
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    frames = data['frames']
    
    # Prepare transform to convert PIL images to tensors
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    seva_json = seva_dir / "transforms.json"

    with open(seva_json, 'r') as f:
        seva_data = json.load(f)
    
    seva_frames = seva_data['frames']
    seva_target_images_c = []
    # Load context images and camera parameters
    context_images = []
    context_extrinsics = []
    context_intrinsics = []
    
    for i, idx in enumerate(context_indices):
        if i == 0:
            seva_frame_c = seva_frames[1]
        elif i == 1:
            seva_frame_c = seva_frames[-2]
        # Load image
        seva_img_path_c = Path(seva_dir) / seva_frame_c['file_path'].replace('./', '')
        seva_image_c = Image.open(seva_img_path_c).convert('RGB')

        seva_image_tensor = transform(seva_image_c)
        seva_target_images_c.append(seva_image_tensor)

        
        frame = frames[idx]
        # # Load image
        # img_path = Path(image_folder) / frame['file_path'].replace('./', '')
        # # print(img_path)
        # image = Image.open(img_path).convert('RGB')
        # image.save(f"input_{idx}.png")
        # image_tensor = transform(image)
        # context_images.append(image_tensor)
        
        transform_matrix = np.array(frame['transform_matrix'])
        if transform_matrix.shape == (3, 4):
            # Add the bottom row [0, 0, 0, 1]
            bottom_row = np.array([[0, 0, 0, 1]])
            transform_matrix = np.concatenate([transform_matrix, bottom_row], axis=0)
        
        # Convert to camera-to-world (if needed - check your convention)
        extrinsics = torch.tensor(transform_matrix, dtype=torch.float32)
        context_extrinsics.append(extrinsics.inverse())
        
        # Create intrinsics matrix from fl_x, fl_y, cx, cy
        intrinsics = torch.tensor([
            [frame['fl_x'], 0, frame['cx']],
            [0, frame['fl_y'], frame['cy']],
            [0, 0, 1]
        ], dtype=torch.float32)
        context_intrinsics.append(intrinsics)
    
    # Load target images and camera parameters
    target_images = []
    target_extrinsics = []
    target_intrinsics = []
    seva_target_images = []



    for i, idx in enumerate(target_indices):
        seva_frame = seva_frames[i+2]
        # Load image
        seva_img_path = Path(seva_dir) / seva_frame['file_path'].replace('./', '')
        seva_image = Image.open(seva_img_path).convert('RGB')

        seva_image_tensor = transform(seva_image)
        seva_target_images.append(seva_image_tensor)

        frame = frames[idx]
        # Load image
        img_path = Path(image_folder) / frame['file_path'].replace('./', '')
        image = Image.open(img_path).convert('RGB')

        image_tensor = transform(image)
        target_images.append(image_tensor)

        # Extract camera parameters
        transform_matrix = np.array(frame['transform_matrix'])
        if transform_matrix.shape == (3, 4):
            # Add the bottom row [0, 0, 0, 1]
            bottom_row = np.array([[0, 0, 0, 1]])
            transform_matrix = np.concatenate([transform_matrix, bottom_row], axis=0)
        extrinsics = torch.tensor(transform_matrix, dtype=torch.float32)
        target_extrinsics.append(extrinsics.inverse())
        
        # Create intrinsics matrix
        intrinsics = torch.tensor([
            [frame['fl_x'], 0, frame['cx']],
            [0, frame['fl_y'], frame['cy']],
            [0, 0, 1]
        ], dtype=torch.float32) 
        target_intrinsics.append(intrinsics)
    
    # Stack all tensors
    # context_images = torch.stack(context_images).unsqueeze(0)  # [1, num_context, 3, H, W]
    context_images = torch.stack(seva_target_images_c).unsqueeze(0)  # [1, num_context, 3, H, W]
    
    context_extrinsics = torch.stack(context_extrinsics).unsqueeze(0)  # [1, num_context, 4, 4]
    context_intrinsics = torch.stack(context_intrinsics).unsqueeze(0)  # [1, num_context, 3, 3]
    
    target_images = torch.stack(target_images).unsqueeze(0)  # [1, num_target, 3, H, W]
    target_extrinsics = torch.stack(target_extrinsics).unsqueeze(0)  # [1, num_target, 4, 4]
    target_intrinsics = torch.stack(target_intrinsics).unsqueeze(0)  # [1, num_target, 3, 3]
    seva_target_images = torch.stack(seva_target_images).unsqueeze(0)  # [1, num_target, 3, H, W]
    # Create batch in the expected format
    batch = {
        'context': {
            'extrinsics': context_extrinsics,
            'intrinsics': context_intrinsics,
            'image': context_images,
            'near': torch.ones(1, len(context_indices)) * 1,
            'far': torch.ones(1, len(context_indices)) * 100.0,
            'index': torch.tensor([context_indices], dtype=torch.long)
        },
        'target': {
            'extrinsics': target_extrinsics,
            'intrinsics': target_intrinsics,
            'image': target_images,
            'seva_image': seva_target_images,
            'near': torch.ones(1, len(target_indices)) * 1,
            'far': torch.ones(1, len(target_indices)) * 100.0,
            'index': torch.tensor([target_indices], dtype=torch.long)
        },
        'scene': ['custom_example']
    }
    
    return batch


@hydra.main(version_base=None, config_path="config", config_name="main")
def evaluate_all(cfg_dict):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    step_tracker = StepTracker()

    model_kwargs = {
        "optimizer_cfg": cfg.optimizer,
        "test_cfg": cfg.test,
        "train_cfg": cfg.train,
        "encoder": encoder,
        "encoder_visualizer": encoder_visualizer,
        "decoder": get_decoder(cfg.model.decoder, cfg.dataset),
        "losses": get_losses(cfg.loss),
        "step_tracker": step_tracker,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelWrapper.load_from_checkpoint(cfg.checkpointing.load, **model_kwargs, strict=False)
    model = model.to(device)
    model.eval()

    base_dir = Path("/data8/stilex/transcode_cropped_576")
    # base_dir = Path("/home/stilex/stable-virtual-camera/work_dirs/demo/img2img/gap50")
    output_base = Path("/data8/stilex/mask_eval_576_seva_50_1")
    seva_base = Path("/home/stilex/stable-virtual-camera/work_dirs/demo/img2img/gap50")


    seva_lpips = []
    seva_ssim = []
    seva_psnr = []

    alphas = [50,60,70,80,85,90,95]
    avg_area = []
    psnrs = []
    ssims = []
    lpipss = []
    for i in alphas:
        lpipss.append([])
        ssims.append([])
        psnrs.append([])
        avg_area.append([])
    count = 0
    for scene_dir in tqdm(sorted(base_dir.iterdir())):
        if not scene_dir.is_dir():
            continue
        
        count +=1
        
        if count > 1000:
            break

        if not (seva_base / scene_dir.name).is_dir():
            print(f"{scene_dir.name} does not exist under {seva_base}")
            continue
        
        seva_dir = seva_base / scene_dir.name
        scene_id = scene_dir.name
        json_path = scene_dir / "transforms.json"
        image_folder = scene_dir

        with open(json_path) as f:
            frames = json.load(f)["frames"]

        num_frames = len(frames)
        if num_frames < 50:
            print(f"Skipping {scene_id} due to insufficient frames")
            continue
        print(f"scence:", scene_dir.name)
        context_indices = [10, 40]
        # target_indices = []
        target_indices = [20, 30]



        batch = load_custom_example(json_path, image_folder,seva_dir, context_indices, target_indices)

        # Move batch to device
        batch = {k: {k2: v2.to(device) if isinstance(v2, torch.Tensor) else v2 
                    for k2, v2 in v.items()} if isinstance(v, dict) else v
                 for k, v in batch.items()}
        
        # print("batch context:", batch['context'])
        # print("batch target:", batch['target'])
        
        out_dir = output_base / scene_id
        out_dir_output = out_dir / "output"
        out_dir_output_mixed = []
        out_dir_masks = []

        for i in alphas:
            out_dir_groundtruth_mask = out_dir / f"groundtruth_mask_{i}"
            out_dir_groundtruth_mask.mkdir(parents=True, exist_ok=True)
            out_dir_masks.append(out_dir_groundtruth_mask)
            out_dir_output_mask = out_dir / f"output_mask_{i}"
            out_dir_output_mask.mkdir(parents=True, exist_ok=True)
            out_dir_output_mixed.append(out_dir_output_mask) 
                                    
        out_dir_gt = out_dir / "groundtruth"
        out_dir_output.mkdir(parents=True, exist_ok=True)
        out_dir_gt.mkdir(parents=True, exist_ok=True)

        seva_dir = out_dir / "seva"
        seva_dir.mkdir(parents=True, exist_ok=True)

        context_dir = out_dir / "context"
        context_dir.mkdir(parents=True, exist_ok=True)
               
        with torch.no_grad():
            if hasattr(model, 'data_shim'):
                batch = model.data_shim(batch)

            b, v, _, h, w = batch["target"]["image"].shape

            gaussians = model.encoder(batch["context"], 0, deterministic=False)
            output = model.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=None,
            )

            rendered_images = output.color[0]
            alpha = output.alpha[0]
            # print("alpha:", alpha.size())
            # print(output)
            ground_truth_images = batch["target"]["image"][0]
            seva_images = batch["target"]["seva_image"][0]
            context_images = batch["context"]["image"][0]
            save_image(context_images[0], context_dir / "000.png")
            save_image(context_images[1], context_dir / "049.png")
            
                
            
            for i, (rendered, gt, seva, a) in enumerate(zip(rendered_images, ground_truth_images,seva_images, alpha)):
                save_image(rendered, out_dir_output / f"{target_indices[i]:03d}.png")
                save_image(gt, out_dir_gt / f"{target_indices[i]:03d}.png")
                save_image(seva, seva_dir / f"{target_indices[i]:03d}.png")
                
                for j, threshold in enumerate(alphas):
                    binary_mask = (a >= threshold / 100.0).float()  # [H, W]
                    masked_ratio = (1-binary_mask).mean().item()
                    avg_area[j].append(masked_ratio)

                    print(f"threshold {threshold} avg area ratio:", sum(avg_area[j])/len(avg_area[j]))
                    if len(avg_area[j]) > 1:
                        std = statistics.stdev(avg_area[j])
                        # print(f"threshold {threshold} area ratio std:", std)
                    render_mixed = rendered * binary_mask + seva * (1 - binary_mask)
                    render_masked = rendered * binary_mask
                    save_image(render_mixed, out_dir_output_mixed[j] / f"{target_indices[i]:03d}.png")
                    save_image(render_masked, out_dir_masks[j] / f"{target_indices[i]:03d}.png")
                    psnr = compute_psnr(gt.unsqueeze(0) , render_mixed.unsqueeze(0) ).mean().item()
                    ssim = compute_ssim(gt.unsqueeze(0) , render_mixed.unsqueeze(0) ).mean().item()
                    lpips = compute_lpips(gt.unsqueeze(0) , render_mixed.unsqueeze(0) ).mean().item()
                    # print("psnr:", psnr)
                    # print("ssim:", ssim)
                    # print("lpips:", lpips)

                    lpipss[j].append(lpips)
                    ssims[j].append(ssim)
                    psnrs[j].append(psnr)
                    print(f"threshold {threshold}  psnr:", psnr)
                    print(f"threshold {threshold}  sim:", ssim)
                    print(f"threshold {threshold}  lpips:", lpips)
                print("\n")



            # Compute metrics for seva_images
            seva_psnr_value = compute_psnr(ground_truth_images, seva_images).mean().item()
            seva_ssim_value = compute_ssim(ground_truth_images, seva_images).mean().item()
            seva_lpips_value = compute_lpips(ground_truth_images, seva_images).mean().item()

            seva_psnr.append(seva_psnr_value)
            seva_ssim.append(seva_ssim_value)
            seva_lpips.append(seva_lpips_value)

            print("seva avg psnr:", seva_psnr_value)
            print("seva avg sim:", seva_ssim_value)
            print("seva avg lpips:", seva_lpips_value)

            # Compute metrics for rendered_images
            rendered_psnr_value = compute_psnr(ground_truth_images, rendered_images).mean().item()
            rendered_ssim_value = compute_ssim(ground_truth_images, rendered_images).mean().item()
            rendered_lpips_value = compute_lpips(ground_truth_images, rendered_images).mean().item()

            print("mvsplat avg psnr:", rendered_psnr_value)
            print("mvsplat avg sim:", rendered_ssim_value)
            print("mvsplat avg lpips:", rendered_lpips_value)

            # Save all results in one JSON file
            with open(out_dir / "metrics.json", "w") as f:
                json.dump({
                    "seva": {
                        "psnr": seva_psnr_value,
                        "ssim": seva_ssim_value,
                        "lpips": seva_lpips_value,
                    },
                    "rendered": {
                        "psnr": rendered_psnr_value,
                        "ssim": rendered_ssim_value,
                        "lpips": rendered_lpips_value,
                    },
                    "context_indices": context_indices,
                    "target_indices": target_indices,
                }, f, indent=2)

                
    for i, threshold in enumerate(alphas):
        ts = (threshold / 100.0)
        print("threshold:", ts)
        avg_l = sum(lpipss[i])/len(lpipss[i])
        avg_s = sum(ssims[i])/len(ssims[i])
        avg_p = sum(psnrs[i])/len(psnrs[i])
        avg_ratio = sum(avg_area[i])/len(avg_area[i])
        print("avg lpips:", avg_l)
        print("avg ssim:", avg_s)
        print("avg psnr:", avg_p)
        print("avg ratio:",avg_ratio)
        

if __name__ == "__main__":
    evaluate_all()