# rho값 확인하는 코드입니다.
import torch
import matplotlib.pyplot as plt
import numpy as np

def check_learned_rhos():
    weight_path = './experiments/full_tuning_batch8_iter12_epoch50/best_model.pth'
    print(f"Loading weights from {weight_path}...")
    try:
        checkpoint = torch.load(weight_path, map_location='cpu')
    except FileNotFoundError:
        print("Error: 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # 'log_rhos' 파라미터 찾기
    if 'log_rhos' in state_dict:
        log_rhos = state_dict['log_rhos']
        
        rhos = torch.exp(log_rhos).numpy()
        
        print("\n=== Learned Rho Values ===")
        print(f"Total Iterations: {len(rhos)}")
        for i, rho in enumerate(rhos):
            print(f"Iter {i+1:02d}: {rho:.6f}")
            
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(rhos) + 1), rhos, marker='o', linestyle='-', color='b')
        plt.title("Learned HQS Penalty Parameters (Rho)")
        plt.xlabel("Iteration")
        plt.ylabel("Rho Value")
        plt.grid(True)
        plt.show()
        
    else:
        print("Error: 'log_rhos' 키를 찾을 수 없습니다. 모델 구조를 확인해주세요.")

if __name__ == "__main__":
    check_learned_rhos()