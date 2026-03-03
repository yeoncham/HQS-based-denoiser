# LoRA 적용 코드입니다.
"""
lora.py
NAFNet에 LoRA를 적용하는 모듈
"""

import torch
import torch.nn as nn
import math


class Conv2dLoRA(nn.Module):
    """Conv2d with LoRA (Low-Rank Adaptation)"""
    
    def __init__(self, conv_layer, rank=4, alpha=1.0):
        super().__init__()
        
        # 원본 Conv2d layer (frozen)
        self.conv = conv_layer
        self.conv.weight.requires_grad = False # 원본 레이어 가중치 고정
        if self.conv.bias is not None: # bias값이 있으면 이것도 학습 안 되게 고정
            self.conv.bias.requires_grad = False
        device = conv_layer.weight.device
        # LoRA 파라미터
        in_ch = conv_layer.in_channels
        out_ch = conv_layer.out_channels
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank # 학습 속도를 조절하는 스케일링 값
        
        # Low-rank decomposition: W + BA
        # A: in_ch -> rank, B: rank -> out_ch
        self.lora_A = nn.Conv2d(in_ch, rank, 1, bias=False).to(device)
        self.lora_B = nn.Conv2d(rank, out_ch, 1, bias=False).to(device)
        
        # 초기화
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        # 원본 convolution + LoRA adaptation
        result = self.conv(x)
        lora_result = self.lora_B(self.lora_A(x)) * self.scaling
        return result + lora_result


def apply_lora_to_nafnet(nafnet, rank=8, alpha=8.0, target_modules=None):
    """
    NAFNet의 Conv2d 레이어에 LoRA 적용
    
    Args:
        nafnet: NAFNet 모델
        rank: LoRA rank (4, 8, 16 추천. 낮을수록 파라미터 적음)
        alpha: LoRA scaling factor (보통 rank와 동일하게 설정)
        target_modules: LoRA를 적용할 모듈 이름 리스트
                       None이면 encoders, decoders, middle_blks
    
    Returns:
        lora_params: 학습 가능한 LoRA 파라미터 리스트
    """
    
    if target_modules is None:
        # 기본: intro, ending 제외한 중간 레이어들
        target_modules = ['encoders', 'decoders', 'middle_blks']
    
    lora_params = []
    total_original = 0
    total_lora = 0
    replaced_count = 0
    
    def replace_conv_recursive(module, prefix=''):
        nonlocal replaced_count, total_original, total_lora
        
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Conv2d 레이어인지 확인
            if isinstance(child, nn.Conv2d):
                # target_modules에 포함되는지 확인
                should_apply = any(target in full_name for target in target_modules)
                
                if should_apply:
                    # Conv2dLoRA로 교체
                    lora_conv = Conv2dLoRA(child, rank=rank, alpha=alpha)
                    setattr(module, name, lora_conv)
                    
                    # 통계 계산
                    orig_params = sum(p.numel() for p in child.parameters())
                    lora_params_count = (lora_conv.lora_A.weight.numel() + 
                                        lora_conv.lora_B.weight.numel())
                    
                    total_original += orig_params
                    total_lora += lora_params_count
                    replaced_count += 1
                    
                    # LoRA 파라미터 리스트에 추가
                    lora_params.append(lora_conv.lora_A.weight)
                    lora_params.append(lora_conv.lora_B.weight)
                    
                    if replaced_count <= 5:  # 처음 5개만 출력
                        print(f"  ✓ {full_name}: {orig_params:,} params "
                              f"(+{lora_params_count:,} LoRA)")
            else:
                # 재귀적으로 하위 모듈 탐색
                replace_conv_recursive(child, full_name)
    
    print("="*60)
    print("Applying LoRA to NAFNet")
    print("="*60)
    print(f"  Rank: {rank}")
    print(f"  Alpha: {alpha}")
    print(f"  Target modules: {target_modules}")
    print()
    
    # Conv2d 레이어들을 LoRA로 교체
    replace_conv_recursive(nafnet)
    
    if replaced_count > 5:
        print(f"  ... ({replaced_count - 5} more layers)")
    
    print()
    print("Summary:")
    print(f"  Replaced {replaced_count} Conv2d layers")
    print(f"  Original parameters (frozen): {total_original:,}")
    print(f"  LoRA parameters (trainable): {total_lora:,}")
    print(f"  Additional parameters: {total_lora/total_original*100:.2f}% of original")
    
    return lora_params