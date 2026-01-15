import os, argparse, json
import numpy as np
from tqdm import tqdm
from typing import List, Tuple


def iter_npy_pairs(root: str, split: str = "train_data") -> List[Tuple[str, str, int, int]]:
    """
    遍历目录，找到所有(rgb.npy, delta_actions.npy)对
    
    目录格式: {split}/step_{step_num}/video/eval/seed_{seed_num}/{rgb.npy, delta_actions.npy}
    
    step_num 范围: 0, 25, 50, 75, 100, 125, 150, 175
    seed_num 范围: 0, 1, 2, 3, 4, 5, 6, 7
    """
    pairs = []
    split_dir = os.path.join(root, split)
    
    if not os.path.exists(split_dir):
        print(f"目录不存在: {split_dir}")
        return pairs
    
    step_nums = [0, 25, 50, 75, 100, 125, 150, 175]
    seed_nums = [0, 1, 2, 3, 4, 5, 6, 7]
    
    for step_num in step_nums:
        step_dir = f"step_{step_num}"
        step_path = os.path.join(split_dir, step_dir, "video", "eval")
        
        if not os.path.exists(step_path):
            continue
        
        for seed_num in seed_nums:
            seed_dir = f"seed_{seed_num}"
            seed_path = os.path.join(step_path, seed_dir)
            
            video_path = os.path.join(seed_path, "rgb.npy")
            action_path = os.path.join(seed_path, "delta_actions.npy")
            
            if os.path.exists(video_path) and os.path.exists(action_path):
                pairs.append((video_path, action_path, step_num, seed_num))
    
    return pairs


def compute_statistics(data_path: str, split: str = "train_data"):
    """
    计算数据集的统计信息
    
    Returns:
        dict: 包含 action 统计信息的字典
    """
    print(f"正在扫描 {split} 数据...")
    npy_pairs = iter_npy_pairs(data_path, split)
    print(f"找到 {len(npy_pairs)} 个文件对")
    
    if len(npy_pairs) == 0:
        print("未找到数据文件，返回空统计信息")
        return {
            "action": {
                "mean": [0.0] * 7,
                "std": [1.0] * 7,
                "max": [0.0] * 7,
                "min": [0.0] * 7,
                "q01": [0.0] * 7,
                "q99": [0.0] * 7,
                "mask": [True] * 7
            },
            "proprio": {
                "mean": [0.0] * 7,
                "std": [0.0] * 7,
                "max": [0.0] * 7,
                "min": [0.0] * 7,
                "q01": [0.0] * 7,
                "q99": [0.0] * 7
            },
            "num_transitions": 0,
            "num_trajectories": 0
        }
    
    # 收集所有 action 数据
    all_actions = []
    num_trajectories = 0
    num_transitions = 0
    
    print("正在加载数据...")
    for video_path, action_path, step_num, seed_num in tqdm(npy_pairs, desc="Loading data"):
        try:
            action_data = np.load(action_path)  # (步数, 轨迹数, 7)
            num_steps, num_trajs = action_data.shape[0], action_data.shape[1]
            
            # 将所有轨迹的动作数据展平: (步数 * 轨迹数, 7)
            action_flat = action_data.reshape(-1, 7)
            all_actions.append(action_flat)
            
            num_trajectories += num_trajs
            num_transitions += num_steps * num_trajs
        except Exception as e:
            print(f"[WARN] 跳过 ({step_num}, {seed_num}): {e}")
    
    if len(all_actions) == 0:
        print("未成功加载任何数据")
        return None
    
    # 合并所有动作数据: (总步数, 7)
    all_actions = np.concatenate(all_actions, axis=0)
    print(f"总动作数: {all_actions.shape[0]}, 动作维度: {all_actions.shape[1]}")
    
    # 计算统计信息
    print("正在计算统计信息...")
    action_stats = {
        "mean": all_actions.mean(axis=0).tolist(),
        "std": all_actions.std(axis=0).tolist(),
        "max": all_actions.max(axis=0).tolist(),
        "min": all_actions.min(axis=0).tolist(),
        "q01": np.percentile(all_actions, 1, axis=0).tolist(),
        "q99": np.percentile(all_actions, 99, axis=0).tolist(),
        "mask": [True] * 7  # 默认所有维度都使用
    }
    
    # proprio 统计信息（当前数据集中没有，设为0）
    proprio_stats = {
        "mean": [0.0] * 7,
        "std": [0.0] * 7,
        "max": [0.0] * 7,
        "min": [0.0] * 7,
        "q01": [0.0] * 7,
        "q99": [0.0] * 7
    }
    
    return {
        "action": action_stats,
        "proprio": proprio_stats,
        "num_transitions": int(num_transitions),
        "num_trajectories": int(num_trajectories)
    }


def main():
    parser = argparse.ArgumentParser(
        description="计算数据集统计信息并生成 JSON 文件"
    )
    parser.add_argument(
        "--data_path",
        default="/mnt/mnt/public/zefang/AgiBotWorldChallengeIROS2025-WorldModelBaseline/rlinf_dataset_1114_split",
        help="数据根目录（包含train_data和val_data）"
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/mnt/public/jzn/workspace/WMPO/data_files/statistics/rlinf_libero_dataset_1114_split",
        help="输出目录"
    )
    parser.add_argument(
        "--dataset_name",
        default="rlinf_libero_dataset_1114_split",
        help="数据集名称"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train_data", "val_data"],
        help="要处理的数据分割"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    result = {}
    
    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"处理 {split}...")
        print(f"{'='*60}")
        
        stats = compute_statistics(args.data_path, split)
        if stats is not None:
            result[f"{args.dataset_name}_{split}"] = stats
        else:
            print(f"警告: {split} 统计信息计算失败")
    
    # 保存 JSON 文件
    output_path = os.path.join(args.output_dir, "dataset_statistics.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ 统计信息已保存到: {output_path}")
    print(f"包含 {len(result)} 个数据集分割的统计信息")


if __name__ == "__main__":
    main()

