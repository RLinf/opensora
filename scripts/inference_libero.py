import os
import json
import imageio
import hydra
from pprint import pformat
from pathlib import Path
from glob import glob
from omegaconf import DictConfig, OmegaConf

import numpy as np
import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm
from collections import deque
from PIL import Image
import tensorflow as tf
from mmengine.config import Config
from opensora.acceleration.parallel_states import set_sequence_parallel_group
# following 2 lines can not be deleted，which is tried to register SimpleVLAWebDataset
from opensora.registry import MODELS, SCHEDULERS, DATASETS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.inference_utils import prepare_multi_resolution_info
from opensora.utils.misc import create_logger, is_distributed, to_torch_dtype

'''
data_dir:

val_data/
├─ step_<step_num>_seed_<seed_num>_traj_<traj_num>/
│  ├─ images/
│  │  ├─ frame_0000.png
│  │  ├─ frame_0001.png
│  │  ├─ frame_{:4d}.png
│  │  ...
│  ├─ actions.npy
│  ├─ init_ee_pose.npy
│  ├─ rgb.npy
│  ├─ videos.mp4
│
├─ step_0_seed_0_traj_0/
│  └─ （同上结构）
│
└─ ... （更多
'''


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig):
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # Convert OmegaConf DictConfig to mmengine.Config for compatibility
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg = Config(cfg_dict)

    # ---------- 2. 读取归一化常量 ----------
    stats_path = cfg.stats
    stats = json.load(open(stats_path, "r"))
    q01 = np.asarray(stats["action"]["q01"], np.float32)
    q99 = np.asarray(stats["action"]["q99"], np.float32)
    
    print("begin inference")

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # == init distributed env ==
    if is_distributed():
        if not dist.is_initialized():
            colossalai.launch_from_torch({})
        coordinator = DistCoordinator()
        enable_sequence_parallelism = coordinator.world_size > 1
        if enable_sequence_parallelism:
            set_sequence_parallel_group(dist.group.WORLD)
    else:
        coordinator = None
        enable_sequence_parallelism = False
    set_random_seed(seed=cfg.get("seed", 1024))

    # == init logger ==
    logger = create_logger()
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 1)

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()

    # == prepare video size ==
    image_size = cfg.image_size
    num_frames = cfg.num_frames
    # num_frames = get_num_frames(cfg.num_frames)

    # == build diffusion model ==
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae.out_channels,
            enable_sequence_parallelism=False,
        )
        .to(device, dtype)
        .eval()
    )

    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # ======================================================
    # inference
    # ======================================================
    # == prepare arguments ==
    fps = cfg.fps
    multi_resolution = cfg.get("multi_resolution", None)
    condition_frame_length = cfg.get("condition_frame_length", 4)
    epoch_num = cfg.get("epoch_num", 0)
    
    # == data paths ==
    val_data_processed_dir = cfg.get("val_data_processed_dir", "/mnt/mnt/public/jzn/workspace/WMPO/data_files/val_data/rlinf_libero/val_data_processed")
    output_base_dir = cfg.get("output_dir", "./generate_opensora")
    output_dir = os.path.join(output_base_dir, f"epoch_{epoch_num}")
    os.makedirs(output_dir, exist_ok=True)

    model_args = prepare_multi_resolution_info(
        multi_resolution, 1, image_size, num_frames, fps, device, dtype
    )

    # == get all trajectories to process ==
    # Option 1: Process all trajectories in val_data_processed
    if cfg.get("process_all", True):
        traj_dirs = sorted(glob(os.path.join(val_data_processed_dir, "step_*_seed_*_traj_*")))
    else:
        # Option 2: Process specific trajectories from config
        # step_nums = cfg.get("step_nums", [0, 25, 50, 75, 100, 125, 150, 175])
        step_nums = cfg.get("step_nums", [0])
        seed_nums = cfg.get("seed_nums", [0, 1, 2, 3, 4, 5, 6, 7])
        traj_nums = cfg.get("traj_nums", [0, 1, 2])
        traj_dirs = []
        for step_num in step_nums:
            for seed_num in seed_nums:
                for traj_num in traj_nums:
                    traj_dir = os.path.join(
                        val_data_processed_dir,
                        f"step_{step_num}_seed_{seed_num}_traj_{traj_num}"
                    )
                    if os.path.exists(traj_dir):
                        traj_dirs.append(traj_dir)

    logger.info(f"Found {len(traj_dirs)} trajectories to process")

    # == process each trajectory ==
    for traj_dir in tqdm(traj_dirs, desc="Processing trajectories"):
        traj_path = Path(traj_dir)
        traj_name = traj_path.name  # e.g., "step_0_seed_0_traj_0"
        
        # Create output directory
        output_traj_dir = os.path.join(output_dir, traj_name)
        os.makedirs(output_traj_dir, exist_ok=True)
        output_images_dir = os.path.join(output_traj_dir, "images")
        os.makedirs(output_images_dir, exist_ok=True)

        # Load data
        delta_actions_path = traj_path / "delta_actions.npy" #(512, 7) -> (511, 7)
        abs_actions_path = traj_path / "abs_actions.npy"
        rgb_path = traj_path / "rgb.npy" #(3,256,256)

        if not all([delta_actions_path.exists(), abs_actions_path.exists(), rgb_path.exists()]):
            logger.warning(f"Missing files in {traj_dir}, skipping...")
            continue

        # Load actions (we'll use delta_actions or abs_actions based on config)
        actions = np.load(delta_actions_path)[1:]  # (512, 7)
        
        rgb_data = np.load(rgb_path)  # (512, 3, 256, 256)

        actions = 2 * ((actions - q01) / (q99 - q01)) - 1

        # Prepare action chunks
        action_chunk_length = num_frames - condition_frame_length
        disc_actions = torch.from_numpy(actions)
        z_mask_frame_num = int(action_chunk_length / 4 if cfg.vae['type'] == 'OpenSoraVAE_V1_2' else action_chunk_length)
        z_condition_frame_length = int(condition_frame_length / 4 if cfg.vae['type'] == 'OpenSoraVAE_V1_2' else condition_frame_length)
        
        # Prepare condition frames from RGB data
        image_queue = deque(maxlen=z_condition_frame_length)
        image_list = []

        # ======================================================
        # 修改1: 初始帧处理为 repeat 第一帧观测到 condition_frame_length
        # ======================================================
        # Process first frame only, then repeat it
        first_frame_rgb = rgb_data[0].transpose(1, 2, 0)  # (256, 256, 3)
        
        # Ensure uint8 format
        if first_frame_rgb.dtype != np.uint8:
            if first_frame_rgb.max() <= 1.0:
                first_frame_rgb = (first_frame_rgb * 255).astype(np.uint8)
            else:
                first_frame_rgb = np.clip(first_frame_rgb, 0, 255).astype(np.uint8)
        
        # Process image similar to inference_libero.py
        img = tf.image.encode_jpeg(first_frame_rgb)
        img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
        img = tf.image.resize(img, size=(256, 256), method="lanczos3", antialias=True)
        img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
        img = img.numpy()
        img = torch.tensor(img)
        img = img / 255.0
        img = img * 2 - 1
        img = img.permute(2, 0, 1)  # (3, H, W)
        
        processed_image = img.unsqueeze(1).unsqueeze(0).to(device).to(dtype)  # (1, C, 1, H, W)
        
        # Repeat first frame to condition_frame_length
        for i in range(condition_frame_length):
            image_list.append(processed_image)
        
        images = torch.concat(image_list, axis=2)  # (1, C, T, H, W)
        x = vae.encode(images)
        for i in range(x.shape[2]):
            image_queue.append(x[:, :, i:i+1])
        
        torch.manual_seed(cfg.get("seed", 1024))

        # Prepare actions for inference (skip condition frames) Ta
        disc_actions = disc_actions[condition_frame_length-1:]
        
        # Store generated video frames
        generated_video = []
        # Add condition frames to output (repeat first frame)
        first_frame_output = rgb_data[0].transpose(1, 2, 0)
        if first_frame_output.dtype != np.uint8:
            if first_frame_output.max() <= 1.0:
                first_frame_output = (first_frame_output * 255).astype(np.uint8)
            else:
                first_frame_output = np.clip(first_frame_output, 0, 255).astype(np.uint8)
        # Resize to match output size if needed
        if first_frame_output.shape[:2] != tuple(image_size):
            first_frame_output = np.array(Image.fromarray(first_frame_output).resize(image_size))
        # Repeat first frame for condition_frame_length times
        for i in range(condition_frame_length):
            generated_video.append(first_frame_output)

        # ======================================================
        # 修改2: 处理最后不足 action_chunk_length 的 actions
        # ======================================================
        # Generate video chunks
        total_actions = len(disc_actions)
        for i in tqdm(range(0, total_actions, action_chunk_length), 
                     desc=f"Generating {traj_name}", leave=False):
            action = disc_actions[i:i+action_chunk_length]
            is_last_chunk = (i + action_chunk_length >= total_actions)
            
            # If last chunk is shorter than action_chunk_length, pad with last action
            if len(action) < action_chunk_length:
                last_action = action[-1:]  # Get last action
                # Repeat last action to fill up to action_chunk_length
                padding_needed = action_chunk_length - len(action)
                padding_actions = last_action.repeat(padding_needed, 1)
                action = torch.cat([action, padding_actions], dim=0)
            
            action = action.reshape(-1, 7)
            y = action.unsqueeze(0).to(device).to(dtype)

            mask_images = torch.concat(list(image_queue), dim=2)
            z = torch.randn(1, vae.out_channels, z_mask_frame_num, *latent_size[1:], device=device, dtype=dtype)
            masks = torch.tensor([[0]*z_condition_frame_length+[1]*(z_mask_frame_num)], device=device, dtype=dtype)
            z = torch.concat([mask_images, z], dim=2)

            samples = scheduler.sample(
                model,
                z=z,
                y=y,
                device=device,
                additional_args=model_args,
                progress=verbose >= 2,
                mask=masks,
            )
            pred_images = samples[:, :, -z_mask_frame_num:, :, :].to(dtype)
            
            if cfg.vae['type'] == 'OpenSoraVAE_V1_2':
                image_queue.extend(pred_images.clone().chunk(z_mask_frame_num, dim=2))
                pred_images = vae.decode(pred_images, num_frames=12)
            else:
                image_queue.extend(pred_images.clone().chunk(action_chunk_length, dim=2))
                pred_images = vae.decode(pred_images)
            
            pred_images = pred_images.squeeze().cpu().to(torch.float32)
            pred_images = (
                (pred_images.permute(1, 2, 3, 0).numpy() * 0.5 + 0.5) * 255
            ).clip(0, 255).astype(np.uint8)  # (T, H, W, C)

            # If this is the last chunk and we padded actions, remove extra frames
            if is_last_chunk and len(disc_actions[i:]) < action_chunk_length:
                actual_actions = len(disc_actions[i:])
                # Calculate how many frames to keep based on actual actions
                if cfg.vae['type'] == 'OpenSoraVAE_V1_2':
                    # For VAE_V1_2, decode generates num_frames=12 frames
                    # We need to keep frames proportional to actual actions
                    # Since decode always generates 12 frames for action_chunk_length actions,
                    # we keep frames proportional to actual_actions / action_chunk_length
                    frames_to_keep = int(actual_actions / action_chunk_length * 12)
                else:
                    # For regular VAE, 1 action corresponds to 1 frame
                    frames_to_keep = actual_actions
                pred_images = pred_images[:frames_to_keep]

            generated_video.extend(pred_images)

        # Save generated images
        for frame_idx, frame in enumerate(generated_video):
            frame_filename = os.path.join(output_images_dir, f"frame_{frame_idx:04d}.png")
            Image.fromarray(frame).save(frame_filename)

        # Save video
        video_path = os.path.join(output_traj_dir, "video.mp4")
        imageio.mimwrite(video_path, generated_video, fps=30)
        
        logger.info(f"Saved generated video to {video_path}")

    logger.info("Inference finished.")
    logger.info(f"Saved all samples to {output_dir}")


if __name__ == "__main__":
    main()

