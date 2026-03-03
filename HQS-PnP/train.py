# NAFNet training 코드입니다.
"""
사용법:
  python train.py --full-tuning --random-init --epochs 200 --lr 1e-3
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DIV2KDataset
from models import NAFNet
import matplotlib.pyplot as plt
import os
import sys
import json
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='NAFNet Training')
    
    # 학습 파라미터
    parser.add_argument('--epochs', type=int, default=200, help='학습 epoch 수')
    parser.add_argument('--batch-size', type=int, default=16, help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-3, help='시작 Learning rate')
    parser.add_argument('--patch-size', type=int, default=128, help='패치 크기')
    
    # 전략 설정
    parser.add_argument('--full-tuning', action='store_true', help='전체 파라미터 학습')
    parser.add_argument('--random-init', action='store_true', help='랜덤 초기화 사용')
    
    # 경로 설정
    parser.add_argument('--train-dir', type=str, default='./DIV2K/DIV2K_train_HR', help='데이터 경로')
    parser.add_argument('--weights', type=str, default='./weights/NAFNet-SIDD-width32.pth', help='가중치 경로')
    parser.add_argument('--save-dir', type=str, default='./experiments', help='저장 경로')
    
    return parser.parse_args()


def initialize_weights(model): # Kaiming Normal 초기화 적용
    for m in model.modules(): # model.modules(): 모델 내부에 있는 모든 레이어(Conv2d, ReLU, Normalization, ...)
        if isinstance(m, nn.Conv2d): # 꺼낸 레이어 m이 Conv2d 층이 맞다면
            # m.weight: 이 레이어가 가진 가중치, kaiming_normal: Kaiming 초기화 적용. 평균이 0, 표준편차가 sqrt(2/n)
            # fan_out: 출력 채널 수를 기준으로 가중치의 크기 조절
            # nonlinearity: 이 가중치가 나중에 ReLU를 거칠 것을 미리 계산에 넣겠다는 의미
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: 
                nn.init.constant_(m.bias, 0) # 가중치 뒤에 편향은 0으로 채움
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)): # m이 정규화층이라면
            if m.weight is not None: nn.init.constant_(m.weight, 1)
            if m.bias is not None: nn.init.constant_(m.bias, 0)    

def train(args):
    # 실험 폴더 생성 및 명세서(Config) 저장
    exp_name = f"{'random' if args.random_init else 'pretrained'}_{datetime.now().strftime('%m%d_%H%M')}"
    save_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ 학습 시작: {exp_name} | Device: {device}")

    # 데이터셋 및 모델 준비
    train_dataset = DIV2KDataset(args.train_dir, patch_size=args.patch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    model = NAFNet(img_channel=4, width=32).to(device)
    
    # 가중치 로드 또는 초기화
    if not args.random_init and os.path.exists(args.weights):
        checkpoint = torch.load(args.weights, map_location=device)
        state_dict = checkpoint['params'] if 'params' in checkpoint else checkpoint
        if 'intro.weight' in state_dict and state_dict['intro.weight'].shape[1] == 3:
            w = state_dict['intro.weight']
            new_w = torch.zeros((w.size(0), 4, 3, 3), device=w.device)
            new_w[:, :3] = w
            new_w[:, 3:4] = w.mean(dim=1, keepdim=True)
            state_dict['intro.weight'] = new_w
        model.load_state_dict(state_dict, strict=False)
        print("✓ Pretrained 가중치를 로드했습니다.")
    else:
        initialize_weights(model)
        print("✓ Kaiming 초기화를 적용합니다.")

    # 옵티마이저(AdamW) 및 스케줄러(Cosine) 설정
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-3  
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    criterion = nn.L1Loss()
    loss_history = []
    best_loss = float('inf')

    # 학습 루프
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        current_lr = optimizer.param_groups[0]['lr']
        
        for batch_idx, (x, y, m) in enumerate(train_loader):
            x, y, m = x.to(device), y.to(device), m.to(device)
            input_cat = torch.cat([y, m], dim=1)
            
            output = model(input_cat)
            loss = criterion(output, x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch [{epoch}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.6f}", end='\r')
        
        scheduler.step() # 학습률 업데이트
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"\n  Epoch [{epoch}/{args.epochs}] Avg Loss: {avg_loss:.6f} | LR: {current_lr:.8f}")
        
        # Best 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': vars(args)
            }, os.path.join(save_dir, 'best_model.pth'))

    # 최종 결과 저장 (그래프 및 Summary)
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))
    
    summary = {
        'best_loss': best_loss,
        'final_loss': avg_loss,
        'total_epochs': args.epochs,
        'final_lr': optimizer.param_groups[0]['lr']
    }
    with open(os.path.join(save_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"\n✓ 학습 완료! 결과 저장: {save_dir}")

if __name__ == "__main__":
    train(parse_args())