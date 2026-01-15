#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VideoMAE success/failure classifier - 单GPU版本，详细进度打印
"""
from torch.utils.tensorboard import SummaryWriter
import os, glob, random
import traceback
from typing import Dict, Any
from collections import OrderedDict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import (
    VideoMAEConfig,
    VideoMAEForVideoClassification,
    AutoModel, AutoConfig,
)

# HF feature extractor / image processor
try:
    from transformers import VideoMAEImageProcessor as VideoMAEFeatureExtractor
except Exception:
    from transformers import VideoMAEFeatureExtractor

# =========================
# CONFIG
# =========================
CFG = dict(
    TRAIN_DATA_DIR="/mnt/mnt/public/jzn/workspace/RLinf-backup/reward_model/reward_data/embodiment_converted/train_data",
    VAL_DATA_DIR="/mnt/mnt/public/jzn/workspace/RLinf-backup/reward_model/reward_data/embodiment_converted/val_data",
    IMG_SIZE=224,
    WINDOW=8,
    BATCH_SIZE=8,                # 单GPU可以适当增大
    VAL_BATCH_SIZE=32,           # 验证批次可以更大
    NUM_WORKERS=8,               # 增加worker加速数据加载
    PERSISTENT_WORKERS=True,
    PREFETCH_FACTOR=4,
    LR=1e-4,
    WEIGHT_DECAY=1e-4,
    MAX_STEPS=200_000,
    EVAL_STEPS=1500,
    CKPT_DIR="ckpts_videomae_single2",
    SEED=42,
    MODEL_NAME="MCG-NJU/videomae-base",
    NUM_LABELS=2,
    THRESH_MIN=0.3,
    THRESH_MAX=1.0,
    THRESH_STEPS=20,
    NEG_TO_POS_RATIO=1.0,
    DROP_LAST=True,
)

# =========================
# Helpers
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    """获取设备"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"🎯 GPU型号: {torch.cuda.get_device_name()}")
        print(f"🎯 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    return device

def collate_fn(batch):
    vids = torch.stack([b[0] for b in batch])
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    meta_keys = batch[0][2].keys() if batch[0][2] else []
    meta: Dict[str, Any] = {k: [b[2][k] for b in batch] for k in meta_keys}
    return vids, ys, meta

# =========================
# Balanced Keyframe Dataset
# =========================
class BalancedKeyframeDataset(Dataset):
    def __init__(self, data_files, window=8, img_size=224, mode="train", 
                 neg_to_pos_ratio=1.0):
        self.data_files = data_files
        self.window = window
        self.mode = mode
        self.fe = VideoMAEFeatureExtractor(size=img_size)
        self.neg_to_pos_ratio = neg_to_pos_ratio
        
        print(f"📊 开始生成{mode}样本...")
        self.samples = self._balanced_keyframe_sampling()
        
        # 统计信息
        if len(self.samples) > 0:
            pos_count = sum(1 for _, _, label, _ in self.samples if label == 1)
            neg_count = sum(1 for _, _, label, _ in self.samples if label == 0)
            print(f"✅ {mode}样本统计 - 正样本: {pos_count:,}, 负样本: {neg_count:,}, 总计: {len(self.samples):,}")
            print(f"📈 正负样本比例: {pos_count/len(self.samples)*100:.1f}% : {neg_count/len(self.samples)*100:.1f}%")
        else:
            print(f"❌ 警告: {mode}数据集没有生成任何样本!")
    
    def _balanced_keyframe_sampling(self):
        samples = []
        
        print(f"🔍 处理 {len(self.data_files)} 个数据文件...")
        for file_idx, file_path in enumerate(self.data_files):
            if file_idx % 100 == 0:  # 每100个文件打印一次进度
                print(f"📁 处理文件 {file_idx}/{len(self.data_files)}: {os.path.basename(file_path)}")
                
            try:
                data = np.load(file_path, allow_pickle=True)
                T = len(data)
                rewards = np.array([item['reward'] for item in data])
                
                if self.mode == "train":
                    file_samples = self._sample_balanced_keyframes(file_idx, data, rewards, T)
                else:
                    file_samples = self._sample_validation(file_idx, data, rewards, T)
                
                samples.extend(file_samples)
            except Exception as e:
                print(f"❌ 加载文件 {file_path} 失败: {e}")
                continue
                
        return samples
    
    def _sample_balanced_keyframes(self, file_idx, data, rewards, T):
        W = self.window
        if T < W:
            return []
        
        # 找到所有关键帧（正reward的位置）
        keyframe_indices = np.where(rewards > 0)[0]
        
        # 策略2: 关键帧结尾的窗口
        positive_samples = []
        for keyframe in keyframe_indices:
            start = keyframe - W + 1
            if start >= 0:
                positive_samples.append((file_idx, start, 1, "keyframe_end"))
        
        # 负样本采样
        negative_samples = self._sample_balanced_negatives(file_idx, data, rewards, T, 
                                                          len(positive_samples))
        
        return positive_samples + negative_samples
    
    def _sample_balanced_negatives(self, file_idx, data, rewards, T, num_positives):
        W = self.window
        negative_samples = []
        
        if num_positives == 0:
            return self._sample_diverse_negatives(file_idx, data, T, 10)
        
        target_negatives = int(num_positives * self.neg_to_pos_ratio)
        
        # 候选负样本
        candidate_negatives = []
        for start in range(0, T - W + 1):
            window_rewards = rewards[start:start + W]
            if all(r == 0 for r in window_rewards):
                candidate_negatives.append(start)
        
        # 随机选择
        if len(candidate_negatives) > target_negatives:
            selected_negatives = random.sample(candidate_negatives, target_negatives)
        else:
            selected_negatives = candidate_negatives
        
        for start in selected_negatives:
            negative_samples.append((file_idx, start, 0, "balanced_negative"))
        
        return negative_samples
    
    def _sample_diverse_negatives(self, file_idx, data, T, num_samples):
        W = self.window
        negatives = []
        
        if T <= W:
            return negatives
            
        regions = [
            (0, T // 3),
            (T // 3, 2 * T // 3),  
            (2 * T // 3, T - W)
        ]
        
        for region_start, region_end in regions:
            if region_end > region_start:
                samples_in_region = max(1, num_samples // 3)
                for _ in range(samples_in_region):
                    start = random.randint(region_start, min(region_end, T - W))
                    negatives.append((file_idx, start, 0, "diverse_negative"))
        
        return negatives[:num_samples]
    
    def _sample_validation(self, file_idx, data, rewards, T):
        W = self.window
        samples = []
        stride = 4
        
        for start in range(0, T - W + 1, stride):
            window_rewards = rewards[start:start + W]
            label = 1 if any(r > 0 for r in window_rewards) else 0
            samples.append((file_idx, start, label, "validation"))
        
        return samples
    
    def _get_window_data(self, data, start_idx):
        frames = []
        for i in range(start_idx, start_idx + self.window):
            img_data = data[i]['image']
            if img_data.dtype != np.uint8:
                img_data = (np.clip(img_data, 0, 1) * 255).astype(np.uint8)
            frame = Image.fromarray(img_data)
            frames.append(frame)
        
        return self.fe(frames, return_tensors="pt")["pixel_values"][0]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_idx, start, label, sample_type = self.samples[idx]
        data = np.load(self.data_files[file_idx], allow_pickle=True)
        video_tensor = self._get_window_data(data, start)
        return video_tensor, torch.tensor(label, dtype=torch.long), {"type": sample_type}

# =========================
# Evaluation
# =========================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    logits_all, trues_all = [], []

    print("🔍 开始验证...")
    for batch_idx, (vids, ys, _) in enumerate(tqdm(loader, desc="验证")):
        vids = vids.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        logits = model(pixel_values=vids).logits
        logits_all.extend(logits.cpu().tolist())
        trues_all.extend(ys.cpu().tolist())
        
        if batch_idx % 10 == 0:  # 每10个batch打印进度
            print(f"  验证进度: {batch_idx}/{len(loader)}")

    logits_t = torch.tensor(logits_all)
    probs = torch.softmax(logits_t, dim=-1)[:, 1].numpy()

    thresholds = np.linspace(CFG["THRESH_MIN"], CFG["THRESH_MAX"], CFG["THRESH_STEPS"])
    all_metrics = {}
    best = {"f1": -1.0, "thresh": thresholds[0]}

    print("📊 计算验证指标...")
    for th in thresholds:
        preds = (probs >= th).astype(np.int32).tolist()
        acc = accuracy_score(trues_all, preds)
        prec = precision_score(trues_all, preds, zero_division=0)
        rec = recall_score(trues_all, preds, zero_division=0)
        f1 = f1_score(trues_all, preds, zero_division=0)

        all_metrics[f"thresh_{th:.2f}"] = OrderedDict(
            acc=acc, precision=prec, recall=rec, f1=f1
        )
        if f1 > best["f1"]:
            best["f1"], best["thresh"] = f1, th

    return all_metrics, best

# =========================
# Training
# =========================
def main():
    try:
        set_seed(CFG["SEED"])
        torch.backends.cudnn.benchmark = True
        
        device = get_device()
        tb_writer = SummaryWriter(log_dir=os.path.join(CFG["CKPT_DIR"], "tensorboard"))
        # --------- 检查数据文件 ---------
        print("🔍 检查数据文件...")
        train_files = sorted(glob.glob(os.path.join(CFG["TRAIN_DATA_DIR"], "*.npy")))
        val_files = sorted(glob.glob(os.path.join(CFG["VAL_DATA_DIR"], "*.npy")))
        print(f"📁 训练文件: {len(train_files):,}")
        print(f"📁 验证文件: {len(val_files):,}")

        if len(train_files) == 0:
            raise RuntimeError("❌ 没有找到训练文件!")
        if len(val_files) == 0:
            print("⚠️  警告: 没有找到验证文件，将使用训练集进行验证")
            val_files = train_files[:100]

        # --------- Datasets ---------
        print("🗂️  创建数据集...")
        tr_ds = BalancedKeyframeDataset(
            data_files=train_files,
            window=CFG["WINDOW"],
            img_size=CFG["IMG_SIZE"],
            mode="train",
            neg_to_pos_ratio=CFG["NEG_TO_POS_RATIO"]
        )
        
        va_ds = BalancedKeyframeDataset(
            data_files=val_files,
            window=CFG["WINDOW"],
            img_size=CFG["IMG_SIZE"],
            mode="val"
        )

        if len(tr_ds) == 0:
            raise RuntimeError("❌ 训练数据集为空!")

        # --------- DataLoaders ---------
        print("🔄 创建数据加载器...")
        tr_ld = DataLoader(
            tr_ds,
            batch_size=CFG["BATCH_SIZE"],
            num_workers=CFG["NUM_WORKERS"],
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=CFG["PERSISTENT_WORKERS"],
            prefetch_factor=CFG["PREFETCH_FACTOR"],
            drop_last=CFG["DROP_LAST"],
            shuffle=True
        )
        va_ld = DataLoader(
            va_ds,
            batch_size=CFG["VAL_BATCH_SIZE"],
            num_workers=CFG["NUM_WORKERS"],
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=CFG["PERSISTENT_WORKERS"],
            prefetch_factor=CFG["PREFETCH_FACTOR"],
            drop_last=False,
        )

        # --------- Model / Optim ---------
        print("🧠 加载模型...")
        '''cfg = VideoMAEConfig.from_pretrained(
            CFG["MODEL_NAME"],
            num_frames=CFG["WINDOW"],
            num_labels=CFG["NUM_LABELS"],
        )
        model = VideoMAEForVideoClassification.from_pretrained(CFG["MODEL_NAME"], config=cfg).to(device)'''
        cfg = VideoMAEConfig.from_pretrained(
            "/mnt/mnt/public/jzn/workspace/WMPO/videomae-base-8frame",  # 本地路径
            num_frames=CFG["WINDOW"],
            num_labels=CFG["NUM_LABELS"],
        )
        model = VideoMAEForVideoClassification.from_pretrained(
            "/mnt/mnt/public/jzn/workspace/WMPO/videomae-base-8frame",  # 本地路径
            config=cfg
        ).to(device)
        checkpoint_path = "/mnt/mnt/public/jzn/workspace/ckpts_videomae_single/best_videomae_f10.8415.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        print(f"从检查点恢复模型: {checkpoint_path}")
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📐 模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["LR"], weight_decay=CFG["WEIGHT_DECAY"])

        # --------- Step-based Train loop ---------
        os.makedirs(CFG["CKPT_DIR"], exist_ok=True)
        global_step, best_f1 = 0, -1.0

        print(f"🚀 开始训练!")
        print(f"🎯 目标步数: {CFG['MAX_STEPS']:,}")
        print(f"📊 评估间隔: {CFG['EVAL_STEPS']} 步")
        print(f"📦 Batch大小: {CFG['BATCH_SIZE']}")
        print(f"🔄 Workers数量: {CFG['NUM_WORKERS']}")
        print("-" * 50)

        # 训练循环
        while global_step < CFG["MAX_STEPS"]:
            model.train()
            epoch_loss = 0
            batch_count = 0
            
            for batch_idx, (vids, ys, _) in enumerate(tr_ld):
                vids = vids.to(device, non_blocking=True)
                ys = ys.to(device, non_blocking=True)

                logits = model(pixel_values=vids).logits
                loss = criterion(logits, ys)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                global_step += 1
                epoch_loss += loss.item()
                batch_count += 1
                
                # ==================== 指标计算和TensorBoard记录 ====================
                # 计算预测结果
                preds = torch.argmax(logits, dim=1)
                
                # 计算准确率
                acc = (preds == ys).float().mean().item()
                
                # 计算精确率、召回率、F1
                preds_np = preds.cpu().numpy()
                ys_np = ys.cpu().numpy()
                
                # 避免除零错误
                if len(np.unique(ys_np)) > 1:  # 确保有两个类别
                    precision = precision_score(ys_np, preds_np, zero_division=0)
                    recall = recall_score(ys_np, preds_np, zero_division=0)
                    f1 = f1_score(ys_np, preds_np, zero_division=0)
                else:
                    precision = 0.0
                    recall = 0.0
                    f1 = 0.0
                
                # 记录到TensorBoard
                tb_writer.add_scalar('Train/Loss', loss.item(), global_step)
                tb_writer.add_scalar('Train/Accuracy', acc, global_step)
                tb_writer.add_scalar('Train/Precision', precision, global_step)
                tb_writer.add_scalar('Train/Recall', recall, global_step)
                tb_writer.add_scalar('Train/F1', f1, global_step)
                tb_writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)

                # 验证和保存
                if global_step % CFG["EVAL_STEPS"] == 0:
                    print(f"\n⭐ 在步数 {global_step} 进行评估...")
                    all_metrics, best = evaluate(model, va_ld, device)
                    
                    print(f"\n📊 验证结果 @ 步数 {global_step}:")
                    for k, v in all_metrics.items():
                        acc = v["acc"]; prec = v["precision"]; rec = v["recall"]; f1 = v["f1"]
                        print(f"   {k}: 准确率={acc:.4f} 精确率={prec:.4f} 召回率={rec:.4f} F1={f1:.4f}")
                    
                    print(f"🏆 最佳F1={best['f1']:.4f} @ 阈值={best['thresh']:.2f}")

                    # ==================== 记录验证指标到TensorBoard ====================
                    best_thresh_key = f"thresh_{best['thresh']:.2f}"
                    if best_thresh_key in all_metrics:
                        val_metrics = all_metrics[best_thresh_key]
                        tb_writer.add_scalar('Val/Best_Accuracy', val_metrics["acc"], global_step)
                        tb_writer.add_scalar('Val/Best_Precision', val_metrics["precision"], global_step)
                        tb_writer.add_scalar('Val/Best_Recall', val_metrics["recall"], global_step)
                        tb_writer.add_scalar('Val/Best_F1', val_metrics["f1"], global_step)
                        tb_writer.add_scalar('Val/Best_Threshold', best['thresh'], global_step)

                    # 保存检查点
                    step_pth = os.path.join(
                        CFG["CKPT_DIR"],
                        f"videomae_step{global_step}_f1{best['f1']:.4f}.pth"
                    )
                    torch.save(model.state_dict(), step_pth)
                    print(f"💾 检查点已保存: {step_pth}")

                    # 保存最佳检查点
                    if best["f1"] > best_f1:
                        best_f1 = best["f1"]
                        best_pth = os.path.join(
                            CFG["CKPT_DIR"],
                            f"best_videomae_f1{best_f1:.4f}.pth"
                        )
                        torch.save(model.state_dict(), best_pth)
                        print(f"⭐ 最佳检查点已保存: {best_pth}")
                        
                        # 记录最佳F1
                        tb_writer.add_scalar('Val/Best_F1_So_Far', best_f1, global_step)

                    print("-" * 50)

                if global_step >= CFG["MAX_STEPS"]:
                    break

        print(f"✅ 训练完成! 总步数: {global_step}")
        # 关闭TensorBoard
        tb_writer.close()
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        traceback.print_exc()
        # 确保异常时关闭TensorBoard
        if 'tb_writer' in locals():
            tb_writer.close()

if __name__ == "__main__":
    main()