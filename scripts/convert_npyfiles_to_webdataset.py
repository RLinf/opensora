import os, argparse, json, warnings, random
from typing import Any, List, Tuple
import numpy as np
from webdataset import ShardWriter
from tqdm import tqdm


# ---------- 工具 ----------
def jsonable(v: Any):
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def npy_to_samples(ep_idx_start: int, video_path: str, action_path: str, step_num: int, seed_num: int):
    """
    从npy文件中提取所有轨迹，每个轨迹作为一个sample
    
    Args:
        ep_idx_start: 起始episode索引
        video_path: video.npy文件路径
        action_path: delta_actions.npy文件路径
        step_num: step编号
        seed_num: seed编号
    
    Returns:
        List[dict]: 每个轨迹的sample列表
    """
    video_data = np.load(video_path)  # (步数, 轨迹数, 3, 256, 256)
    action_data = np.load(action_path)  # (步数, 轨迹数, 7)
    
    num_steps, num_trajectories = video_data.shape[0], video_data.shape[1]
    
    samples = []
    for traj_idx in range(num_trajectories):
        # 提取单个轨迹: (步数, 3, 256, 256) -> (步数, 256, 256, 3) for uint8
        video_traj = video_data[:, traj_idx, :, :, :]  # (步数, 3, 256, 256)
        # 转换为 (步数, 256, 256, 3) 格式，符合常见的视频格式
        video_traj = np.transpose(video_traj, (0, 2, 3, 1))  # (步数, 256, 256, 3)
        
        # 提取单个轨迹的action: (步数, 7)
        action_traj = action_data[:, traj_idx, :]  # (步数, 7)
        
        # 构建metadata
        meta = {
            "finish_step": int(num_steps),
            "step_num": int(step_num),
            "seed_num": int(seed_num),
            "traj_idx": int(traj_idx),
            "video_path": video_path,
            "action_path": action_path,
        }
        
        key = f"{ep_idx_start + traj_idx:09d}"
        sample = {
            "__key__": key,
            "video.npy": video_traj.astype(np.uint8),
            "action.npy": action_traj.astype(np.float32),
            "meta.json": json.dumps(meta).encode("utf-8"),
        }
        samples.append(sample)
    
    return samples


def iter_npy_pairs(root: str, split: str = "train_data") -> List[Tuple[str, str, int, int]]:
    """
    遍历目录，找到所有(video.npy, delta_actions.npy)对
    
    目录格式: {split}/step_{step_num}/video/eval/seed_{seed_num}/{rgb.npy, delta_actions.npy}
    例如: train_data/step_0/video/eval/seed_0/rgb.npy
    
    step_num 范围: 0, 25, 50, 75, 100, 125, 150, 175
    seed_num 范围: 0, 1, 2, 3, 4, 5, 6, 7
    
    Returns:
        List[Tuple[video_path, action_path, step_num, seed_num]]
    """
    pairs = []
    split_dir = os.path.join(root, split)
    
    if not os.path.exists(split_dir):
        warnings.warn(f"目录不存在: {split_dir}")
        return pairs
    
    # 定义 step_num 和 seed_num 的范围
    step_nums = [0, 25, 50, 75, 100, 125, 150, 175]
    seed_nums = [0, 1, 2, 3, 4, 5, 6, 7]
    
    # 遍历所有 step_num 和 seed_num 的组合
    for step_num in step_nums:
        step_dir = f"step_{step_num}"
        # 构建路径: train_data/step_{step_num}/video/eval
        step_path = os.path.join(split_dir, step_dir, "video", "eval")
        
        if not os.path.exists(step_path):
            warnings.warn(f"路径不存在: {step_path}")
            continue
        
        for seed_num in seed_nums:
            seed_dir = f"seed_{seed_num}"
            # 构建完整路径: train_data/step_{step_num}/video/eval/seed_{seed_num}/
            seed_path = os.path.join(step_path, seed_dir)
            
            # 文件路径: train_data/step_{step_num}/video/eval/seed_{seed_num}/rgb.npy
            #          train_data/step_{step_num}/video/eval/seed_{seed_num}/delta_actions.npy
            video_path = os.path.join(seed_path, "rgb.npy")
            action_path = os.path.join(seed_path, "delta_actions.npy")
            
            if os.path.exists(video_path) and os.path.exists(action_path):
                pairs.append((video_path, action_path, step_num, seed_num))
            else:
                import pdb; pdb.set_trace()
                warnings.warn(f"文件缺失 [step={step_num}, seed={seed_num}]: video={os.path.exists(video_path)}, action={os.path.exists(action_path)}")
    
    return pairs


# ---------- 主函数 ----------
def convert(root, out_dir, split="train_data", episodes_per_shard=128):
    """
    将npy文件转换为webdataset格式
    
    Args:
        root: 数据根目录（包含train_data和val_data）
        out_dir: 输出目录
        split: 数据集分割名称（train_data或val_data）
        episodes_per_shard: 每个shard包含的episode数
    """
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "%06d.tar")
    
    # 获取所有npy文件对
    npy_pairs = iter_npy_pairs(root, split)
    random.shuffle(npy_pairs)  # 完全随机打乱
    
    print(f"找到 {len(npy_pairs)} 个 (rgb.npy, delta_actions.npy) 对")
    
    with ShardWriter(pattern, maxcount=episodes_per_shard) as sink:
        ep_idx = 0
        for video_path, action_path, step_num, seed_num in tqdm(
            npy_pairs, 
            desc=f"Processing {split} npy files"
        ):
            try:
                samples = npy_to_samples(ep_idx, video_path, action_path, step_num, seed_num)
                for sample in samples:
                    sink.write(sample)
                    ep_idx += 1
            except Exception as exc:
                print(f"[WARN] 跳过 ({step_num}, {seed_num}): {exc}")
    
    print(f"✅ 已写入 {ep_idx} 个 episode → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将npy格式的数据转换为webdataset格式"
    )
    parser.add_argument(
        "--data_path",
        default="/mnt/mnt/public/zefang/AgiBotWorldChallengeIROS2025-WorldModelBaseline/rlinf_dataset_1114_split",
        help="数据根目录（包含train_data和val_data）"
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/mnt/public/jzn/workspace/WMPO/webdataset_shards/rlinf_libero_dataset_1114_split",
        help="保存 .tar 的目录"
    )
    parser.add_argument(
        "--split",
        default="train_data",
        choices=["train_data", "val_data"],
        help="数据集分割名称"
    )
    parser.add_argument(
        "--episodes_per_shard",
        type=int,
        default=4,
        help="每个 .tar 包含的 episode 数"
    )
    args = parser.parse_args()
    
    convert(args.data_path, args.output_dir, args.split, args.episodes_per_shard)

