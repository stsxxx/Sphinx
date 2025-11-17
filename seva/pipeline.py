import glob
import os
import os.path as osp
import hydra
import fire
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from jaxtyping import install_import_hook
import torchvision.transforms as transforms
import json
from seva.data_io import get_parser
from seva.eval import (
    IS_TORCH_NIGHTLY,
    compute_relative_inds,
    create_transforms_simple,
    infer_prior_inds,
    infer_prior_stats,
    run_one_scene,
)
from seva.geometry import (
    generate_interpolated_path,
    generate_spiral_path,
    get_arc_horizontal_w2cs,
    get_default_intrinsics,
    get_lookat,
    get_preset_pose_fov,
)
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.sampling import DDPMDiscretization, DiscreteDenoiser
from seva.utils import load_model
from pathlib import Path
from typing import Tuple, Dict
# for mvsplat
from mvsplat.src.model.model_wrapper import ModelWrapper
from mvsplat.src.config import load_typed_root_config
from mvsplat.src.global_cfg import set_cfg
from mvsplat.src.evaluation.metrics import compute_psnr, compute_ssim, compute_lpips
from mvsplat.src.misc.image_io import save_image
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from mvsplat.src.config import load_typed_root_config
    from mvsplat.src.dataset.data_module import DataModule
    from mvsplat.src.global_cfg import set_cfg
    from mvsplat.src.loss import get_losses
    from mvsplat.src.misc.LocalLogger import LocalLogger
    from mvsplat.src.misc.step_tracker import StepTracker
    from mvsplat.src.misc.wandb_tools import update_checkpoint_path
    from mvsplat.src.model.decoder import get_decoder
    from mvsplat.src.model.encoder import get_encoder
    from mvsplat.src.model.model_wrapper import ModelWrapper
import pyiqa
from scipy.interpolate import PchipInterpolator
import time
from torchvision.io import read_image
from torchvision.utils import save_image, tensor_to_image
from BlurDetection2.process import blur_masks_from_numpy_batch
import open_clip



device = "cuda:0"

rules_98 = [
    [ 
        (0.50, 0.70, 5),
        (0.70, 0.80, 10),
        (0.80, 0.88, 15),
        (0.88, 0.95, 20),
        (0.95, 1.00, 25),
    ],
    [  

        (0.71, 0.77, 5),
        (0.77, 0.84, 10),
        (0.84, 0.90, 15),
        (0.90, 0.97, 20),
        (0.97, 1.00, 25),
    ],
    [   

        (0.63, 0.75, 5),
        (0.75, 0.90, 10),
        (0.90, 1.00, 15),
    ],
]


rules_95 = [
    [ 
        (0.43, 0.65, 10),
        (0.65, 0.78, 15),
        (0.78, 0.88, 20),
        (0.88, 0.94, 25),
        (0.94, 0.98, 30),
        (0.98, 1.00, 35),
    ],
    [   

        (0.54, 0.74, 15),
        (0.74, 0.84, 20),
        (0.84, 0.94, 30),
        (0.94, 1.00, 35),
    ],
    [  

        (0.58, 0.68, 5),
        (0.68, 0.76, 10),
        (0.76, 0.84, 15),
        (0.84, 0.90, 20),
        (0.90, 0.96, 25),
        (0.96, 1.00, 30),
    ]
]

rules_set = {"98": rules_98, "95": rules_95}

# Constants.
WORK_DIR = "work_dirs/demo"

if IS_TORCH_NIGHTLY:
    COMPILE = True
    os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
else:
    COMPILE = False

MODEL = SGMWrapper(load_model(device="cpu", verbose=True).eval()).to(device)
print(MODEL)
AE = AutoEncoder(chunk_size=1).to(device)
CONDITIONER = CLIPConditioner().to(device)
DISCRETIZATION = DDPMDiscretization()
DENOISER = DiscreteDenoiser(discretization=DISCRETIZATION, num_idx=1000, device=device)
VERSION_DICT = {
    "H": 576,
    "W": 576,
    "T": 21,
    "C": 4,
    "f": 8,
    "options": {},
}

if COMPILE:
    MODEL = torch.compile(MODEL, dynamic=False)
    CONDITIONER = torch.compile(CONDITIONER, dynamic=False)
    AE = torch.compile(AE, dynamic=False)


def dilate(m, k=5):
    return (F.max_pool2d(m, kernel_size=k, stride=1, padding=k//2) > 0).float()

def erode(m, k=5):
    return 1.0 - (F.max_pool2d(1.0 - m, kernel_size=k, stride=1, padding=k//2) > 0).float()

def morph_close(m, k=5):
    return erode(dilate(m, k), k)

def morph_open(m, k=3):
    return dilate(erode(m, k), k)

def remove_small(m, win=9, min_pixels=25):
    # Keep a pixel masked only if there are >= min_pixels masked neighbors in a win×win window.
    # This is a sliding-window density filter (fast, approximates area-opening).
    density = F.avg_pool2d(m, kernel_size=win, stride=1, padding=win//2) * (win * win)
    return (density >= min_pixels).float()

def smooth_mask(binary_mask):
    m = binary_mask
    m = morph_close(m, k=5)         # fill small gaps/holes
    m = morph_open(m,  k=3)         # remove isolated speckles
    m = remove_small(m, win=9, min_pixels=25)  # drop tiny islands
    return m

def mask_to_block_mask_exact(coherent_mask: torch.Tensor, block: int = 4) -> Tuple[torch.Tensor, Dict[str, int]]:
    # Normalize to (B,1,H,W)
    m = coherent_mask
    if m.dim() == 2:      # (H,W)
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.dim() == 3:    # (1,H,W) -> (B=1,C=1,H,W)
        m = m.unsqueeze(0)
    elif m.dim() != 4:
        raise ValueError("coherent_mask must have 2, 3, or 4 dims")

    B, _, H, W = m.shape
    assert H % block == 0 and W % block == 0, "H and W must be multiples of block"

    if not m.is_floating_point():
        m = m.float()

    Hb, Wb = H // block, W // block
    pooled = F.max_pool2d(m, kernel_size=block, stride=block)   # (B,1,Hb,Wb)
    block_mask = (pooled > 0).squeeze(1)                        # (B,Hb,Wb) bool

    meta = {"H": H, "W": W, "pad_h": 0, "pad_w": 0, "Hb": Hb, "Wb": Wb, "block": block}
    return block_mask, meta


def parse_task(
    task,
    scene,
    num_inputs,
    T,
    version_dict,
):
    options = version_dict["options"]

    anchor_indices = None
    anchor_c2ws = None
    anchor_Ks = None

    if task == "img2trajvid_s-prob":
        if num_inputs is not None:
            assert (
                num_inputs == 1
            ), "Task `img2trajvid_s-prob` only support 1-view conditioning..."
        else:
            num_inputs = 1
        num_targets = options.get("num_targets", T - 1)
        num_anchors = infer_prior_stats(
            T,
            num_inputs,
            num_total_frames=num_targets,
            version_dict=version_dict,
        )

        input_indices = [0]
        anchor_indices = np.linspace(1, num_targets, num_anchors).tolist()

        all_imgs_path = [scene] + [None] * num_targets

        c2ws, fovs = get_preset_pose_fov(
            option=options.get("traj_prior", "orbit"),
            num_frames=num_targets + 1,
            start_w2c=torch.eye(4),
            look_at=torch.Tensor([0, 0, 10]),
        )

        with Image.open(scene) as img:
            W, H = img.size
            aspect_ratio = W / H
        Ks = get_default_intrinsics(fovs, aspect_ratio=aspect_ratio)  # unormalized
        Ks[:, :2] *= (
            torch.tensor([W, H]).reshape(1, -1, 1).repeat(Ks.shape[0], 1, 1)
        )  # normalized
        Ks = Ks.numpy()

        anchor_c2ws = c2ws[[round(ind) for ind in anchor_indices]]
        anchor_Ks = Ks[[round(ind) for ind in anchor_indices]]

    else:
        parser = get_parser(
            parser_type="reconfusion",
            data_dir=scene,
            normalize=False,
        )
        all_imgs_path = parser.image_paths
        # print(all_imgs_path)
        c2ws = parser.camtoworlds
        # print("c2ws:", c2ws.shape)
        camera_ids = parser.camera_ids
        Ks = np.concatenate([parser.Ks_dict[cam_id][None] for cam_id in camera_ids], 0)

        if num_inputs is None:
            assert len(parser.splits_per_num_input_frames.keys()) == 1
            num_inputs = list(parser.splits_per_num_input_frames.keys())[0]
            split_dict = parser.splits_per_num_input_frames[num_inputs]  # type: ignore
        elif isinstance(num_inputs, str):
            split_dict = parser.splits_per_num_input_frames[num_inputs]  # type: ignore
            num_inputs = int(num_inputs.split("-")[0])  # for example 1_from32
        else:
            split_dict = parser.splits_per_num_input_frames[num_inputs]  # type: ignore

        num_targets = len(split_dict["test_ids"])

        if task == "img2img":
            # Note in this setting, we should refrain from using all the other camera
            # info except ones from sampled_indices, and most importantly, the order.
            num_anchors = infer_prior_stats(
                T,
                num_inputs,
                num_total_frames=num_targets,
                version_dict=version_dict,
            )
            print("num of anchors:", num_anchors)
            sampled_indices = np.sort(
                np.array(split_dict["train_ids"] + split_dict["test_ids"])
            )  # we always sort all indices first

            print("sampled_indices:", sampled_indices)

            traj_prior = options.get("traj_prior", None)
            print("traj prior:", traj_prior)
            if traj_prior == "spiral":
                assert parser.bounds is not None
                anchor_c2ws = generate_spiral_path(
                    c2ws[sampled_indices] @ np.diagflat([1, -1, -1, 1]),
                    parser.bounds[sampled_indices],
                    n_frames=num_anchors + 1,
                    n_rots=2,
                    zrate=0.5,
                    endpoint=False,
                )[1:] @ np.diagflat([1, -1, -1, 1])
            elif traj_prior == "interpolated":
                assert num_inputs > 1
                anchor_c2ws = generate_interpolated_path(
                    c2ws[split_dict["train_ids"], :3],
                    round((num_anchors + 1) / (num_inputs - 1)),
                    endpoint=False,
                )[1 : num_anchors + 1]
            elif traj_prior == "orbit":
                c2ws_th = torch.as_tensor(c2ws)
                lookat = get_lookat(
                    c2ws_th[sampled_indices, :3, 3],
                    c2ws_th[sampled_indices, :3, 2],
                )
                anchor_c2ws = torch.linalg.inv(
                    get_arc_horizontal_w2cs(
                        torch.linalg.inv(c2ws_th[split_dict["train_ids"][0]]),
                        lookat,
                        -F.normalize(
                            c2ws_th[split_dict["train_ids"]][:, :3, 1].mean(0),
                            dim=-1,
                        ),
                        num_frames=num_anchors + 1,
                        endpoint=False,
                    )
                ).numpy()[1:, :3]
            else:
                anchor_c2ws = None
            # anchor_Ks is default to be the first from target_Ks

            all_imgs_path = [all_imgs_path[i] for i in sampled_indices]
            c2ws = c2ws[sampled_indices]
            Ks = Ks[sampled_indices]
            # print("c2ws:", c2ws)
            # print("Ks:", Ks)
            # absolute to relative indices
            input_indices = compute_relative_inds(
                sampled_indices,
                np.array(split_dict["train_ids"]),
            )
            print('input idx:', input_indices)
            anchor_indices = np.arange(
                sampled_indices.shape[0],
                sampled_indices.shape[0] + num_anchors,
            ).tolist()  # the order has no meaning here
            print('anchor idx:', anchor_indices)

        elif task == "img2vid":
            num_targets = len(all_imgs_path) - num_inputs
            num_anchors = infer_prior_stats(
                T,
                num_inputs,
                num_total_frames=num_targets,
                version_dict=version_dict,
            )

            input_indices = split_dict["train_ids"]
            anchor_indices = infer_prior_inds(
                c2ws,
                num_prior_frames=num_anchors,
                input_frame_indices=input_indices,
                options=options,
            ).tolist()
            num_anchors = len(anchor_indices)
            anchor_c2ws = c2ws[anchor_indices, :3]
            anchor_Ks = Ks[anchor_indices]

        elif task == "img2trajvid":
            num_anchors = infer_prior_stats(
                T,
                num_inputs,
                num_total_frames=num_targets,
                version_dict=version_dict,
            )

            target_c2ws = c2ws[split_dict["test_ids"], :3]
            target_Ks = Ks[split_dict["test_ids"]]
            anchor_c2ws = target_c2ws[
                np.linspace(0, num_targets - 1, num_anchors).round().astype(np.int64)
            ]
            anchor_Ks = target_Ks[
                np.linspace(0, num_targets - 1, num_anchors).round().astype(np.int64)
            ]

            sampled_indices = split_dict["train_ids"] + split_dict["test_ids"]
            all_imgs_path = [all_imgs_path[i] for i in sampled_indices]
            c2ws = c2ws[sampled_indices]
            Ks = Ks[sampled_indices]

            input_indices = np.arange(num_inputs).tolist()
            anchor_indices = np.linspace(
                num_inputs, num_inputs + num_targets - 1, num_anchors
            ).tolist()

        else:
            raise ValueError(f"Unknown task: {task}")

    return (
        all_imgs_path,
        num_inputs,
        num_targets,
        input_indices,
        anchor_indices,
        torch.tensor(c2ws[:, :3]).float(),
        torch.tensor(Ks).float(),
        (torch.tensor(anchor_c2ws[:, :3]).float() if anchor_c2ws is not None else None),
        (torch.tensor(anchor_Ks).float() if anchor_Ks is not None else None),
    )
    
    
def load_custom_example(json_path, image_folder, context_indices, target_indices):
    """Load a single example from your custom data format"""
    
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    frames = data['frames']
    
    if applied_transform in data:
        # Top-level intrinsics (DL3DV often keeps these here)
        fl_x_top = data.get('fl_x', None)
        fl_y_top = data.get('fl_y', None)
        cx_top   = data.get('cx', None)
        cy_top   = data.get('cy', None)
        w = data.get('w', None)
        h = data.get('h', None)
        
        # Prepare transform to convert PIL images to tensors
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # --- Context ---
        context_images = []
        context_extrinsics = []
        context_intrinsics = []
        applied_transform = np.concatenate(
                        [data["applied_transform"], [[0, 0, 0, 1]]], axis=0
                    )
        for idx in context_indices:
            frame = frames[idx]
            # Load image
            img_path = Path(image_folder) / frame['file_path'].replace('./', '')
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image)
            context_images.append(image_tensor)

            # Transform matrix (use as-is; pad to 4x4 if needed), then inverse (exactly like your original)
            transform_matrix = np.array(frame['transform_matrix'], dtype=np.float32)
            transform_matrix = np.linalg.inv(applied_transform) @ transform_matrix
            
            transform_matrix[ :, [1, 2]] *= -1
            extrinsics = torch.tensor(transform_matrix, dtype=torch.float32)
            context_extrinsics.append(extrinsics)

            # Intrinsics (prefer per-frame; fallback to top-level)
            fx = frame.get('fl_x', fl_x_top)
            fy = frame.get('fl_y', fl_y_top)
            cx = frame.get('cx',   cx_top)
            cy = frame.get('cy',   cy_top)
            intrinsics = torch.tensor([
                [fx/w, 0, cx/w],
                [0, fy/h, cy/h],
                [0, 0, 1]
            ], dtype=torch.float32)
            context_intrinsics.append(intrinsics)

        # --- Target ---
        target_images = []
        target_extrinsics = []
        target_intrinsics = []

        for idx in target_indices:
            frame = frames[idx]
            # Load image
            img_path = Path(image_folder) / frame['file_path'].replace('./', '')
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image)
            target_images.append(image_tensor)

            # Transform matrix (as-is; pad if needed), then inverse
            transform_matrix = np.array(frame['transform_matrix'], dtype=np.float32)
            transform_matrix = np.linalg.inv(applied_transform) @ transform_matrix

            transform_matrix[ :, [1, 2]] *= -1
            extrinsics = torch.tensor(transform_matrix, dtype=torch.float32)
            target_extrinsics.append(extrinsics)

            # Intrinsics (per-frame -> top-level fallback)
            fx = frame.get('fl_x', fl_x_top)
            fy = frame.get('fl_y', fl_y_top)
            cx = frame.get('cx',   cx_top)
            cy = frame.get('cy',   cy_top)
            intrinsics = torch.tensor([
                [fx/w, 0, cx/w],
                [0, fy/h, cy/h],
                [0, 0, 1]
            ], dtype=torch.float32)
            target_intrinsics.append(intrinsics)

        # Stack all tensors (unchanged from your original)
        context_images = torch.stack(context_images).unsqueeze(0)                 # [1, Nc, 3, H, W]
        context_extrinsics = torch.stack(context_extrinsics).unsqueeze(0)         # [1, Nc, 4, 4]
        context_intrinsics = torch.stack(context_intrinsics).unsqueeze(0)         # [1, Nc, 3, 3]

        target_images = torch.stack(target_images).unsqueeze(0)                   # [1, Nt, 3, H, W]
        target_extrinsics = torch.stack(target_extrinsics).unsqueeze(0)           # [1, Nt, 4, 4]
        target_intrinsics = torch.stack(target_intrinsics).unsqueeze(0)           # [1, Nt, 3, 3]

        # Create batch in the expected format (unchanged)
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
                'near': torch.ones(1, len(target_indices)) * 1,
                'far': torch.ones(1, len(target_indices)) * 100.0,
                'index': torch.tensor([target_indices], dtype=torch.long)
            },
            'scene': ['custom_example']
        }
        
    else:
        # Prepare transform to convert PIL images to tensors
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Load context images and camera parameters
        context_images = []
        context_extrinsics = []
        context_intrinsics = []
        
        for idx in context_indices:
            frame = frames[idx]
            # Load image
            img_path = Path(image_folder) / frame['file_path'].replace('./', '')
            # print(img_path)
            image = Image.open(img_path).convert('RGB')
            # image.save(f"input_{idx}.png")
            image_tensor = transform(image)
            context_images.append(image_tensor)
            
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
        
        for idx in target_indices:
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
        context_images = torch.stack(context_images).unsqueeze(0)  # [1, num_context, 3, H, W]
        context_extrinsics = torch.stack(context_extrinsics).unsqueeze(0)  # [1, num_context, 4, 4]
        context_intrinsics = torch.stack(context_intrinsics).unsqueeze(0)  # [1, num_context, 3, 3]
        
        target_images = torch.stack(target_images).unsqueeze(0)  # [1, num_target, 3, H, W]
        target_extrinsics = torch.stack(target_extrinsics).unsqueeze(0)  # [1, num_target, 4, 4]
        target_intrinsics = torch.stack(target_intrinsics).unsqueeze(0)  # [1, num_target, 3, 3]
        
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
                'near': torch.ones(1, len(target_indices)) * 1,
                'far': torch.ones(1, len(target_indices)) * 100.0,
                'index': torch.tensor([target_indices], dtype=torch.long)
            },
            'scene': ['custom_example']
        }
    
    return batch

def power_ease_benchmark(c0, c1, t_targets, gamma=0.5):
    t = np.asarray(t_targets, dtype=np.float32)
    if c1 >= c0:
        f = t**gamma                # gamma<1 -> flatter near t=1
    else:
        f = 1.0 - (1.0 - t)**gamma  # gamma<1 -> flatter near t=0
    return c0 + (c1 - c0) * f  

def decide_k_curve(x: float, rules: list[tuple[float, float, int]], mvsplat_sentinel: int = 40) -> int:
    """Return k from x using piecewise (lo, hi, k) rules; x>=1.0 => serve MVSplat (sentinel)."""
    if x >= 1.0:
        return mvsplat_sentinel
    for lo, hi, k in rules:
        if lo <= x < hi:
            return k
    return 0

@hydra.main(version_base=None, config_path="mvsplat/config", config_name="main")
def main(
    cfg_dict
):  
    device = "cuda:0"
    print(cfg_dict)
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    base_dir = Path(cfg_dict.seva.data_path)
    baseline_dir = Path(cfg_dict.seva.baseline_path)
    task = cfg_dict.seva.task
    use_traj_prior=False
    quality_factor = cfg_dict.seva.quality_factor
    rules = rules_set[str(quality_factor)]
    
    


    
    if cfg_dict.seva.H is not None:
        VERSION_DICT["H"] = cfg_dict.seva.H
    if cfg_dict.seva.W is not None:
        VERSION_DICT["W"] = cfg_dict.seva.W
    if cfg_dict.seva.T is not None:
        VERSION_DICT["T"] = [int(t) for t in cfg_dict.seva.T.split(",")] if isinstance(cfg_dict.seva.T, str) else cfg_dict.seva.T

    options = VERSION_DICT["options"]
    options["chunk_strategy"] = "nearest-gt"
    options["video_save_fps"] = cfg_dict.seva.video_save_fps
    options["beta_linear_start"] = 5e-6
    options["log_snr_shift"] = 2.4
    options["guider_types"] = 1
    options["cfg"] = 2.0
    options["camera_scale"] = 2.0
    options["num_steps"] = 50
    options["cfg_min"] = 1.2
    options["encoding_t"] = 1
    options["decoding_t"] = 1
    options["num_inputs"] = cfg_dict.seva.num_inputs
    options["seed"] = 23
    options["replace_or_include_input"] = cfg_dict.seva.replace_or_include_input


    num_inputs = options["num_inputs"]
    seed = options["seed"]
    
    
    target_indices = list(range(5, 100, 5))              # 5..95 step 5 (19 targets)
    t_targets = np.array(target_indices, dtype=np.float32) / 100.0



    model_name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    clip_model.eval()
    
    path = "./cluster_centers.npy"
    centers_np = np.load(path)
    centers = torch.from_numpy(centers_np).float().to(device)

    
    # mvsplat part
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


    Output_mvsplat = Path(cfg_dict.seva.save_dir)
    musiq_metric = pyiqa.create_metric('musiq', device=device)        
    
    latencies_path = Output_mvsplat / "latency.json"
    latency_data = {}
    if latencies_path.exists():
        with open(latencies_path, "r") as f:
            try:
                latency_data = json.load(f)
            except json.JSONDecodeError:
                latency_data = {}
    
    for scene_dir in tqdm(sorted(baseline_dir.iterdir())):
        if not scene_dir.is_dir():
            continue
        scene_id = scene_dir.name
        scene_true_dir = base_dir / scene_id 
        json_path = scene_true_dir / "transforms.json"
        image_folder = scene_true_dir
    
        with open(json_path) as f:
            frames = json.load(f)["frames"]
        num_frames = len(frames)
        
        split_path = scene_true_dir / "train_test_split_2.json"
        with open(split_path, "r") as f:
            split_ids = json.load(f)
        context_indices = split_ids["train_ids"]
        target_indices = split_ids["test_ids"]
        
        start_time = time.time()
        if num_frames < context_indices[1]+1:
            print(f"Skipping {scene_id} due to insufficient frames")
            continue
        print(f"scence:", scene_id)
        batch = load_custom_example(json_path, image_folder, context_indices, target_indices)
        batch = {k: {k2: v2.to(device) if isinstance(v2, torch.Tensor) else v2 
                    for k2, v2 in v.items()} if isinstance(v, dict) else v
                 for k, v in batch.items()}
        out_mvsplat_dir = Output_mvsplat / scene_id
        if out_mvsplat_dir.exists():
            print("already done")
            continue
        mvsplat_output = out_mvsplat_dir / "output"
        groundtruth_output = out_mvsplat_dir / "groundtruth"
        context_dir = out_mvsplat_dir / "context"

        mvsplat_output.mkdir(parents=True, exist_ok=True)
        groundtruth_output.mkdir(parents=True, exist_ok=True)
        context_dir.mkdir(parents=True, exist_ok=True)

        
        coherent_masks = []
        block_mask = None
        meta = None
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
            rgb_images = tensor_to_image(rendered_images)

            blur_mask = blur_masks_from_numpy_batch(rgb_images)
            blur_mask = torch.from_numpy((1-blur_mask).astype(np.float32)).to(device)
            


            
            alpha = output.alpha[0]
            ground_truth_images = batch["target"]["image"][0]
            context_images = batch["context"]["image"][0]
            save_image(context_images[0], context_dir / "000.png")
            save_image(context_images[1], context_dir / "100.png")
            binary_mask = (alpha < 0.5).float()
            
            img = Image.open(context_dir / "000.png").convert("RGB")
            x = preprocess(img)                         
            x = x.unsqueeze(0).to(device)           
            with torch.no_grad():
                f = clip_model.encode_image(x)        
            f = torch.nn.functional.normalize(f, p=2, dim=-1)

            f_vec = f.squeeze(0)


            sims = centers @ f_vec    
            best_idx = sims.argmax().item()   
            binary_mask = ((binary_mask + blur_mask) > 0).float() if best_idx != 1 else (binary_mask > 0).float

            
            coherent_mask = binary_mask.any(dim=0).unsqueeze(0).float()
            coherent_mask = smooth_mask(coherent_mask)
            

            # Compute masked area ratio
            masked_area = coherent_mask.mean().item() * 100
            print(f"{masked_area:.2f}% of pixels masked")

            
            for i, (rendered, gt, a) in enumerate(
                zip(rendered_images, ground_truth_images, alpha)
            ):
                # Save original rendered and GT
                save_image(rendered, mvsplat_output / f"{target_indices[i]:03d}.png")
                save_image(gt, groundtruth_output / f"{target_indices[i]:03d}.png")



            
            coherent_mask_72 = (F.max_pool2d(coherent_mask, kernel_size=8,  stride=8,  ceil_mode=True) > 0).squeeze(0)
            coherent_mask_36 = (F.max_pool2d(coherent_mask, kernel_size=16, stride=16, ceil_mode=True) > 0).squeeze(0)
            coherent_mask_18 = (F.max_pool2d(coherent_mask, kernel_size=32, stride=32, ceil_mode=True) > 0).squeeze(0)
            coherent_mask_9  = (F.max_pool2d(coherent_mask, kernel_size=64, stride=64, ceil_mode=True) > 0).squeeze(0)
            # print(f"9 {coherent_mask_9.mean().item() * 100:.2f}% of pixels masked")
            
            coherent_masks.extend([coherent_mask_72, coherent_mask_36, coherent_mask_18, coherent_mask_9])

            
            if masked_area < 10.0 and masked_area > 0.0:
                block_mask, meta = mask_to_block_mask_exact(coherent_mask_72)
                print("block mask:", block_mask)
                print("meta:", meta)
            
                
            if masked_area > 85.0 or masked_area == 0.0:
                coherent_masks = None
                
 
            musiq_mvsplat = musiq_metric(rendered_images.clamp_(0.0, 1.0)).detach().cpu().numpy().reshape(-1)    
            musiq_context = musiq_metric(context_images.clamp_(0.0,1.0)).detach().cpu().numpy().reshape(-1)   
            c0 = float(musiq_context[0]) 
            c1 = float(musiq_context[1]) 

            # power interpolate
            benchmark_np = power_ease_benchmark(c0,c1,t_targets, gamma=0.5)   

            x_ratio = musiq_mvsplat / benchmark_np
            k_decision_curve = np.array([decide_k_curve(float(x), rules[best_idx]) for x in x_ratio], dtype=int).tolist()
            min_k = min(k_decision_curve)
            if masked_area == 0.0:
                k_decision_curve = [40] * VERSION_DICT["T"]
                print("k:", k_decision_curve)
            else:
                k_decision_curve = [min_k] + k_decision_curve + [min_k]
 
            mvsplat_latency = time.time() - start_time
            print("mvsplat latency:", mvsplat_latency)


        # SEVA
        start_time = time.time()
        scene = scene_true_dir
        # print("VERSION_DICT:",VERSION_DICT)
        image_files = [os.path.join(mvsplat_output,f) for f in sorted(os.listdir(mvsplat_output)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        save_path_scene = out_mvsplat_dir / "pipeline"
        (
                    all_imgs_path,
                    num_inputs,
                    num_targets,
                    input_indices,
                    anchor_indices,
                    c2ws,
                    Ks,
                    anchor_c2ws,
                    anchor_Ks,
        ) = parse_task(
            task,
            scene,
            num_inputs,
            VERSION_DICT["T"],
            VERSION_DICT,
        )
        assert num_inputs is not None
        all_imgs_path[1:-1] = image_files
        image_cond = {
                    "img": all_imgs_path,
                    "input_indices": input_indices,
                    "prior_indices": anchor_indices,
        }
        # print(all_imgs_path)
        camera_cond = {
                    "c2w": c2ws.clone(),
                    "K": Ks.clone(),
                    "input_indices": list(range(num_inputs + num_targets)),
        }
        video_path_generator = run_one_scene(
                    task,
                    VERSION_DICT,  # H, W maybe updated in run_one_scene
                    model=MODEL,
                    ae=AE,
                    conditioner=CONDITIONER,
                    denoiser=DENOISER,
                    image_cond=image_cond,
                    camera_cond=camera_cond,
                    save_path=save_path_scene,
                    use_traj_prior=use_traj_prior,
                    traj_prior_Ks=anchor_Ks,
                    traj_prior_c2ws=anchor_c2ws,
                    seed=seed,  
                    mvsplt_image_list=image_files,
                    # s = 0, 
                    s = k_decision_curve,
                    sparse_mask = coherent_masks,
                    block_mask = block_mask,
                    meta = meta
        )
        for _ in video_path_generator:
            pass
        
        seva_latency  = time.time() - start_time
        print("seva latency:", seva_latency)
        
        c2ws = c2ws @ torch.tensor(np.diag([1, -1, -1, 1])).float()
        img_paths = sorted(glob.glob(osp.join(save_path_scene, "samples-rgb", "*.png")))
        if len(img_paths) != len(c2ws):
            input_img_paths = sorted(
                glob.glob(osp.join(save_path_scene, "input", "*.png"))
            )
            assert len(img_paths) == num_targets
            assert len(input_img_paths) == num_inputs
            assert c2ws.shape[0] == num_inputs + num_targets
            target_indices = [i for i in range(c2ws.shape[0]) if i not in input_indices]
            img_paths = [
                input_img_paths[input_indices.index(i)]
                if i in input_indices
                else img_paths[target_indices.index(i)]
                for i in range(c2ws.shape[0])
            ]
        create_transforms_simple(
            save_path=save_path_scene,
            img_paths=img_paths,
            img_whs=np.array([VERSION_DICT["W"], VERSION_DICT["H"]])[None].repeat(
                num_inputs + num_targets, 0
            ),
            c2ws=c2ws,
            Ks=Ks,
        )
        latency_data[scene_id] = {
            "mvsplat_latency": float(mvsplat_latency),
            "seva_latency": float(seva_latency),
            "total_latency": float(mvsplat_latency + seva_latency),
            "s": k_decision_curve,
        }
    
        # Save after each scene (in case of interruption)
        with open(latencies_path, "w") as f:
            json.dump(latency_data, f, indent=2)


if __name__ == "__main__":
    main()
