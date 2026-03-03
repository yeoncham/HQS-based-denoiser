# HQS Unrolled 학습 코드입니다
# run_nafnet.py랑은 다른 학습 코드입니다.
"""
사용법:
    python run_hqs_fanf.py --full-tuning --full-lr 1e-6   # 전체 파라미터 학습 (낮은 LR)
    python run_hqs_fanf.py --use-lora                     # LoRA 사용 (기존)
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DIV2KDataset
from models import HQS_Unrolled
from utils import get_psnr
import matplotlib.pyplot as plt
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='HQS-NAFNet Training & Testing')
    
    # 모드 선택
    parser.add_argument('--test-only', action='store_true', help='테스트만 실행')
    parser.add_argument('--train-only', action='store_true', help='학습만 실행')
    
    # 학습 파라미터
    parser.add_argument('--epochs', type=int, default=20, help='학습 epoch 수')
    parser.add_argument('--batch-size', type=int, default=16, help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-4, help='일반 학습 시 NAFNet learning rate')
    parser.add_argument('--rho-lr', type=float, default=1e-3, help='log_rho learning rate')
    parser.add_argument('--iterations', type=int, default=8, help='HQS iteration 횟수')
    parser.add_argument('--patch-size', type=int, default=64, help='패치 크기')
    
    # LoRA 파라미터
    parser.add_argument('--use-lora', action='store_true', help='LoRA 사용')
    parser.add_argument('--lora-rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=float, default=8.0, help='LoRA alpha')

    # Full Fine-tuning 파라미터
    parser.add_argument('--full-tuning', action='store_true', help='모든 파라미터 학습 (Full Fine-tuning)')
    parser.add_argument('--full-lr', type=float, default=5e-5, help='Full tuning 시 사용할 낮은 LR')
    
    # 경로 설정
    parser.add_argument('--train-dir', type=str, default='./DIV2K_data/DIV2K_train_HR')
    parser.add_argument('--valid-dir', type=str, default='./DIV2K_data/DIV2K_valid_HR')
    parser.add_argument('--weights', type=str, default='./weights/NAFNet-SIDD-width32.pth')
    parser.add_argument('--save-dir', type=str, default='./experiments/fnaf')
    
    return parser.parse_args()


def train(args):
    """학습 함수"""
    print("="*60)
    print("TRAINING START")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 모델 생성
    model = HQS_Unrolled(iterations=args.iterations).to(device)
    
    # Pretrained weights 로드
    if os.path.exists(args.weights):
        checkpoint = torch.load(args.weights, map_location=device)
        state_dict = checkpoint['params'] if 'params' in checkpoint else checkpoint
        
        # 채널 확장: 3 -> 4
        w = state_dict['intro.weight']
        new_w = torch.zeros((w.size(0), 4, 3, 3), device=w.device)
        new_w[:, :3] = w
        new_w[:, 3:4] = w.mean(dim=1, keepdim=True)
        state_dict['intro.weight'] = new_w
        
        model.nafnet.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {args.weights}")

    # --- 학습 모드 및 Optimizer 설정 ---
    if args.full_tuning:
        # 1. Full Fine-tuning: 모든 파라미터를 업데이트 가능하게 설정
        print(f"\n[MODE] Full Fine-tuning (LR: {args.full_lr})")
        for param in model.nafnet.parameters():
            param.requires_grad = True
        
        optimizer = torch.optim.Adam([
            {'params': model.nafnet.parameters(), 'lr': args.full_lr, 'weight_decay': 1e-4},
            {'params': model.log_rhos, 'lr': args.rho_lr},
        ])
        trainable_count = sum(p.numel() for p in model.nafnet.parameters() if p.requires_grad)

    elif args.use_lora:
        # 2. LoRA 적용
        from lora import apply_lora_to_nafnet
        lora_params = apply_lora_to_nafnet(
            model.nafnet, rank=args.lora_rank, alpha=args.lora_alpha,
            target_modules=['encoders', 'decoders', 'middle_blks']
        )
        intro_ending_params = []
        for name, param in model.nafnet.named_parameters():
            if "intro" in name or "ending" in name:
                param.requires_grad = True
                intro_ending_params.append(param)
        
        optimizer = torch.optim.Adam([
            {'params': lora_params, 'lr': args.lr, 'weight_decay': 1e-4},
            {'params': intro_ending_params, 'lr': args.lr, 'weight_decay': 1e-4},
            {'params': model.log_rhos, 'lr': args.rho_lr},
        ])
        trainable_count = sum(p.numel() for p in lora_params) + sum(p.numel() for p in intro_ending_params)
        print(f"\n[MODE] LoRA Training (Rank: {args.lora_rank})")

    else:
        # 3. 기본 방식: Intro + Ending만 학습
        print("\n[MODE] Standard Fine-tuning (Intro + Ending only)")
        trainable_params = []
        for name, param in model.nafnet.named_parameters():
            if "intro" in name or "ending" in name:
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False
        
        optimizer = torch.optim.Adam([
            {'params': trainable_params, 'lr': args.lr, 'weight_decay': 1e-4},
            {'params': model.log_rhos, 'lr': args.rho_lr},
        ])
        trainable_count = sum(p.numel() for p in trainable_params)

    print(f"Total Trainable Parameters: {trainable_count:,}")
    print("="*60)

    # 데이터 로더
    train_dataset = DIV2KDataset(args.train_dir, patch_size=args.patch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    criterion = nn.L1Loss()
    
    loss_history = []
    best_loss = float('inf')

    # 학습 루프
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        for batch_idx, (x, y, m, h) in enumerate(train_loader):
            x, y, m, h = x.to(device), y.to(device), m.to(device), h.to(device)
            
            output = model(y, h, m)
            loss = criterion(output, x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch [{epoch}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.6f}", end='\r')
        
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"\n  Epoch [{epoch}/{args.epochs}] Avg Loss: {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            print("best model is recorded")
            best_loss = avg_loss
            torch.save({'model_state_dict': model.state_dict(), 'args': vars(args)}, 
                       os.path.join(args.save_dir, 'best_model.pth'))
        
        if epoch % 5 == 0:
            save_visual_result(args.save_dir, epoch, x, y, output)
        
        torch.save(model.state_dict(), os.path.join(args.save_dir, "latest_model.pth"))

    # Loss plot 저장
    plt.plot(loss_history)
    plt.savefig(os.path.join(args.save_dir, "loss_plot.png"))
    plt.close()


def save_visual_result(save_dir, epoch, x, y, x_hat):
    def to_img(t): return t[0].detach().cpu().permute(1, 2, 0).numpy()
    plt.figure(figsize=(15, 5))
    imgs, titles = [to_img(x), to_img(y), to_img(x_hat)], ["GT", "Input", f"Restored (E{epoch})"]
    for i, (img, title) in enumerate(zip(imgs, titles)):
        plt.subplot(1, 3, i+1); plt.imshow(img); plt.title(title); plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"result_epoch_{epoch}.png"))
    plt.close()


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(args.save_dir, 'best_model.pth')
    if not os.path.exists(model_path): return
    
    model = HQS_Unrolled(iterations=args.iterations).to(device)
    if args.use_lora:
        from lora import apply_lora_to_nafnet
        apply_lora_to_nafnet(model.nafnet, rank=args.lora_rank, alpha=args.lora_alpha, target_modules=['encoders', 'decoders', 'middle_blks'])

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()

    valid_dataset = DIV2KDataset(args.valid_dir, patch_size=args.patch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    total_psnr = 0
    with torch.no_grad():
        for x, y, m, h in valid_loader:
            x, y, m, h = x.to(device), y.to(device), m.to(device), h.to(device)
            x_hat = model(y, h, m)
            total_psnr += get_psnr(x_hat, x)
    print(f"\n[TEST] Average PSNR: {total_psnr/len(valid_loader):.2f} dB")


def main():
    args = parse_args()
    if args.test_only: test(args)
    else:
        train(args)
        test(args)

if __name__ == "__main__":
    main()