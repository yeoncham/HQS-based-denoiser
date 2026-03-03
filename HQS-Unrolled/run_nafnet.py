# NAFNet만 학습시킨 코드입니다.
"""
run_nafnet.py - HQS 제외하고 NAFNet만으로 학습 및 테스트

사용법:
    python run_nafnet.py                              # 기본 학습+테스트
    python run_nafnet.py --use-lora                   # LoRA 사용
    python run_nafnet.py --use-lora --lora-rank 16    # LoRA rank 변경
    python run_nafnet.py --test-only                  # 테스트만
    python run_nafnet.py --epochs 50 --batch-size 16  # 파라미터 변경
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DIV2KDataset
from models import NAFNet
from utils import get_psnr
import matplotlib.pyplot as plt
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Pure NAFNet Training & Testing (No HQS)')
    
    # 모드 선택
    parser.add_argument('--test-only', action='store_true', help='테스트만 실행')
    parser.add_argument('--train-only', action='store_true', help='학습만 실행')
    
    # 학습 파라미터
    parser.add_argument('--epochs', type=int, default=20, help='학습 epoch 수')
    parser.add_argument('--batch-size', type=int, default=16, help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-4, help='NAFNet learning rate')
    parser.add_argument('--patch-size', type=int, default=64, help='패치 크기')
    
    # LoRA 파라미터
    parser.add_argument('--use-lora', action='store_true', help='LoRA 사용')
    parser.add_argument('--lora-rank', type=int, default=8, help='LoRA rank (4, 8, 16 추천)')
    parser.add_argument('--lora-alpha', type=float, default=8.0, help='LoRA alpha')
    
    # 경로 설정
    parser.add_argument('--train-dir', type=str, default='./DIV2K_data/DIV2K_train_HR', 
                       help='학습 데이터 경로')
    parser.add_argument('--valid-dir', type=str, default='./DIV2K_data/DIV2K_valid_HR',
                       help='검증 데이터 경로')
    parser.add_argument('--weights', type=str, default='./weights/NAFNet-SIDD-width32.pth',
                       help='Pretrained weights 경로')
    parser.add_argument('--save-dir', type=str, default='./experiments/nafnet_only',
                       help='결과 저장 경로')
    
    return parser.parse_args()


def train(args):
    """학습 함수 - NAFNet만 사용"""
    print("="*60)
    print("PURE NAFNET TRAINING (No HQS)")
    print("="*60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    if args.use_lora:
        print(f"LoRA: Enabled (rank={args.lora_rank}, alpha={args.lora_alpha})")
    else:
        print(f"LoRA: Disabled (Intro+Ending only)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 저장 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 데이터 로더
    print(f"\nLoading training data from {args.train_dir}...")
    train_dataset = DIV2KDataset(args.train_dir, patch_size=args.patch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4)
    print(f"Training samples: {len(train_dataset)}")
    
    # NAFNet 모델 생성 (입력 채널 4: RGB + Mask)
    print(f"\nCreating NAFNet model...")
    model = NAFNet(img_channel=4, width=32).to(device)
    print(model)
    # Pretrained weights 로드
    if os.path.exists(args.weights):
        print(f"Loading pretrained weights from {args.weights}...")
        checkpoint = torch.load(args.weights, map_location=device)
        state_dict = checkpoint['params'] if 'params' in checkpoint else checkpoint
        
        # 채널 확장: 3 -> 4 (RGB + Mask)
        w = state_dict['intro.weight']
        new_w = torch.zeros((w.size(0), 4, 3, 3), device=w.device)
        new_w[:, :3] = w
        new_w[:, 3:4] = w.mean(dim=1, keepdim=True)
        state_dict['intro.weight'] = new_w
        
        model.load_state_dict(state_dict, strict=False)
        print("Pretrained weights loaded successfully")
    else:
        print(f"Warning: Pretrained weights not found at {args.weights}")
        print("  Training from scratch...")
    
    # LoRA 또는 기본 Freeze 설정
    if args.use_lora:
        from lora import apply_lora_to_nafnet
        
        print("\n" + "="*60)
        print("LORA CONFIGURATION")
        print("="*60)
        
        # LoRA 적용
        lora_params = apply_lora_to_nafnet(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            target_modules=['encoders', 'decoders', 'middle_blks']
        )
        
        # Intro, Ending도 학습 가능하게
        intro_ending_params = []
        for name, param in model.named_parameters():
            if "intro" in name or "ending" in name:
                param.requires_grad = True
                intro_ending_params.append(param)
        
        # Optimizer: LoRA + intro/ending
        optimizer = torch.optim.Adam([
            {'params': lora_params, 'lr': args.lr, 'weight_decay': 1e-4},
            {'params': intro_ending_params, 'lr': args.lr, 'weight_decay': 1e-4},
        ])
        
        total_trainable = sum(p.numel() for p in lora_params) + \
                         sum(p.numel() for p in intro_ending_params)
        
        print()
        print("Trainable Parameters:")
        print(f"  LoRA: {sum(p.numel() for p in lora_params):,}")
        print(f"  Intro/Ending: {sum(p.numel() for p in intro_ending_params):,}")
        print(f"  Total: {total_trainable:,}")
        print("="*60)
        
    else:
        # 기존 방식: Intro + Ending만 학습
        print("\n" + "="*60)
        print("STANDARD FINE-TUNING (Intro + Ending only)")
        print("="*60)
        
        trainable_count = 0
        for name, param in model.named_parameters():
            if "intro" in name or "ending" in name:
                param.requires_grad = True
                trainable_count += param.numel()
            else:
                param.requires_grad = False
        
        print(f"Trainable parameters: {trainable_count:,}")
        print("="*60)
        
        # Optimizer
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr,
            weight_decay=1e-4
        )
    
    criterion = nn.L1Loss()
    
    # 학습 시작
    print("\n" + "="*60)
    print("Training Progress")
    print("="*60)
    
    loss_history = []
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (x, y, m, h) in enumerate(train_loader):
            x, y, m = x.to(device), y.to(device), m.to(device)
            
            # NAFNet 입력: noisy image (y) + mask (m)
            input_cat = torch.cat([y, m], dim=1)  # [B, 4, H, W]
            
            # NAFNet 순전파
            output = model(input_cat)
            
            # Loss 계산
            loss = criterion(output, x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Progress 출력
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch [{epoch}/{args.epochs}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.6f}", end='\r')
        
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print()
        is_best = avg_loss < best_loss
        print(f"  Epoch [{epoch}/{args.epochs}] "
              f"Avg Loss: {avg_loss:.6f} {'✓ NEW BEST!' if is_best else ''}")
        
        # Best model 저장
        if is_best:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': vars(args),
            }, os.path.join(args.save_dir, 'best_model.pth'))
        
        # 시각화 (5 epoch마다)
        if epoch % 5 == 0:
            save_visual_result(args.save_dir, epoch, x, y, output)
        
        # 최신 모델 저장
        torch.save(model.state_dict(), 
                  os.path.join(args.save_dir, "latest_model.pth"))
    
    # Loss curve 저장
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('L1 Loss', fontsize=12)
    plt.title('NAFNet Training Loss (No HQS)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "loss_plot.png"), dpi=100)
    plt.close()
    
    print("\n" + "="*60)
    print(f"✓ Training completed!")
    print(f"  Best Loss: {best_loss:.6f}")
    print(f"  Results saved to: {args.save_dir}")
    print("="*60)


def save_visual_result(save_dir, epoch, x, y, x_hat):
    """시각화 저장"""
    def to_img(t):
        return t[0].detach().cpu().permute(1, 2, 0).numpy()
    
    plt.figure(figsize=(15, 5))
    titles = ["Ground Truth", "Noisy Input", f"NAFNet Output (Epoch {epoch})"]
    imgs = [to_img(x), to_img(y), to_img(x_hat)]
    
    for i, (img, title) in enumerate(zip(imgs, titles)):
        plt.subplot(1, 3, i+1)
        plt.imshow(img)
        plt.title(title, fontsize=12, fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"result_epoch_{epoch}.png"), dpi=100)
    plt.close()


def test(args):
    """테스트 함수 - 순수 NAFNet만 사용"""
    print("\n" + "="*60)
    print("PURE NAFNET TESTING (No HQS)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 로드
    model_path = os.path.join(args.save_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(args.save_dir, 'latest_model.pth')
    
    if not os.path.exists(model_path):
        print(f"✗ Error: No trained model found in {args.save_dir}")
        return
    
    print(f"Loading model from {model_path}...")
    model = NAFNet(img_channel=4, width=32).to(device)
    
    # LoRA 사용했다면 다시 적용
    if args.use_lora:
        from lora import apply_lora_to_nafnet
        print("\nRe-applying LoRA architecture...")
        apply_lora_to_nafnet(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            target_modules=['encoders', 'decoders', 'middle_blks']
        )
    
    # Checkpoint 로드
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Trained epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Training loss: {checkpoint.get('loss', 'N/A'):.6f}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # 검증 데이터
    print(f"\nLoading validation data from {args.valid_dir}...")
    valid_dataset = DIV2KDataset(args.valid_dir, patch_size=args.patch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Validation samples: {len(valid_dataset)}")
    
    # 테스트
    print("\nTesting...")
    total_psnr = 0
    psnr_list = []
    
    test_results_dir = os.path.join(args.save_dir, 'test_results')
    os.makedirs(test_results_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (x, y, m, h) in enumerate(valid_loader):
            x, y, m = x.to(device), y.to(device), m.to(device)
            
            # NAFNet 입력
            input_cat = torch.cat([y, m], dim=1)
            
            # NAFNet 순전파
            x_hat = model(input_cat)
            
            # PSNR 계산
            psnr = get_psnr(x_hat, x)
            total_psnr += psnr
            psnr_list.append(psnr)
            
            print(f"  Batch [{i+1}/{len(valid_loader)}] PSNR: {psnr:.2f} dB", end='\r')
            
            # 첫 5개 배치만 시각화
            if i < 5:
                res = x_hat[0].cpu().permute(1, 2, 0).numpy()
                plt.imsave(os.path.join(test_results_dir, f'test_res_{i}.png'), res)
    
    avg_psnr = total_psnr / len(valid_loader)
    std_psnr = torch.std(torch.tensor(psnr_list)).item()
    
    print("\n" + "="*60)
    print(f"✓ Testing completed!")
    print(f"  Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"  Test results saved to: {test_results_dir}")
    print("="*60)


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("Pure NAFNet Training & Testing Pipeline")
    print("(No HQS Iterations)")
    print("="*60)
    
    # 데이터 경로 확인
    if not args.test_only:
        if not os.path.exists(args.train_dir):
            print(f"✗ Error: Training data not found at {args.train_dir}")
            sys.exit(1)
    
    if not args.train_only:
        if not os.path.exists(args.valid_dir):
            print(f"✗ Error: Validation data not found at {args.valid_dir}")
            sys.exit(1)
    
    # 실행
    try:
        if args.test_only:
            test(args)
        elif args.train_only:
            train(args)
        else:
            train(args)
            test(args)
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()