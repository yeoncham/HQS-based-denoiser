# ∎ Fourier-Optimized HQS-Based Image Restoration

본 프로젝트는 2026 GIST Winter Internship Program 기간 동안 진행한 연구입니다. 복잡하고 풀기 어려운 이미지 복원 문제를 풀기 쉬운 두 개의 작은 문제(데이터 일치 + 노이즈 제거)로 쪼개서 번갈아 가며 푸는 **HQS(Half-Quadratic Splitting)** 알고리즘 기반의 Denoiser 및 Deblurrer 구현 code입니다.

## ∎ Algorithm Overview

### HQS Optimization
흐릿한 이미지($y$)에서 깨끗한 이미지($x$)를 찾으려면 다음을 최소화해야 합니다.

$$\hat{x} = \arg\min_x ||Ax-y||^2 + \lambda\Phi(x)$$ 

여기서 $||Ax-y||^2$는 데이터항(Data Term), $\lambda\Phi(x)$는 규제항(Prior Term)을 의미합니다. HQS는 보조 변수 $z$를 도입하여 이 문제를 두 단계로 쪼개어 반복(Iteration)해서 풉니다.

**1. Data Step (x-update):** 노이즈 제거는 무시하고, 우선 블러(Blur)부터 역으로 해결하는 단계입니다. FFT(고속 푸리에 변환)를 이용해 닫힌 형태(closed-form)로 연산합니다.  

$$x_{k+1} = \arg\min_x ||Ax-y||^2 + \mu||x-z_k||^2$$ 

**2. Prior Step (z-update):** 반대로 블러는 무시하고, 이미지를 깨끗하게 Denoising 하는 것에 집중하는 단계입니다. 이 과정에 잘 학습된 Denoiser(NAFNet)를 활용합니다.  

$$z_{k+1} = \arg\min_z \frac{\mu}{2}||x_{k+1}-z||^2 + \lambda\Phi(z)$$ 

### Network Structure
```text
Input (noisy y + noise map) ──→ [HQS Iterations] ──→ Output (restored x)
                                     ↓
                         ┌───────────┴───────────┐
                         │                       │
                  Fourier Data Step         NAFNet Prior
                    (closed-form)           (learnable)
                         │                       │
                         └───────────┬───────────┘
                                     ↓
                               Next Iteration

```

## ∎ Architecture & Methodologies
본 레포지토리는 HQS 알고리즘을 두 가지 방식으로 구현 및 비교합니다.


- HQS-PnP (Plug-and-Play): 수학적 틀(HQS)은 그대로 두고, Prior Step의 Denoiser(NAFNet)만 따로 학습시켜 부품처럼 갈아 끼우는 방식입니다.


- HQS-Unrolled: FFT 연산과 NAFNet이 서로 주고받는 반복 과정 전체를 하나의 네트워크로 보고 통째로 학습시키는 방식입니다.

## ∎ Repository Structure
프로젝트는 크게 두 가지 방법론에 따라 폴더가 분리되어 있습니다. 


```text
├── HQS-PnP/                 # Plug-and-Play 기반 알고리즘 구현부
│   ├── kernels_12.npy       # 12종류의 Blur Kernel 데이터 (공통)
│   ├── train.py             # NAFNet Fine-tuning  
│   ├── valid.py             # Denoising 성능 검증  
│   ├── valid_deblur.py      # Deblurring 일반화 성능 검증
│   ├── dataset.py           # Noise 추가 혹은 Blur Kernel 적용한 데이터셋 return 
│   ├── compares.py          # patch 단위 결과 비교용 code
│   ├── compares_full.py     # origianl resolution 비교용 code
│   ├── utils.py             # PSNR 계산 code 
│   └── models.py            # HQS_PnP 클래스 및 NAFNet 구조  
│
└── HQS-Unrolled/            # Unrolled Network 기반 구현부
    ├── kernels_12.npy       # 12종류의 Blur Kernel 데이터 (공통)
    ├── check_rho.py         # rho 값 확인 code
    ├── dataset.py           # Noise 추가 혹은 Blur Kernel 적용한 데이터셋 return
    ├── run_hqs_fnaf.py      # HQS Unrolled 모델 학습 및 테스트 (Full-tuning, LoRA 등)  
    ├── run_nafnet.py        # 순수 NAFNet(HQS 제외) 학습 코드  
    ├── compare_models.py    # 학습한 15개 모델들에 대한 비교 code  
    ├── experi.py            # 실제 노이즈와 맵 노이즈 불일치(Mismatch) 실험 (Only Noise)
    ├── experiment_sigma_mismatch.py # 실제 노이즈와 맵 노이즈 불일치(Mismatch) 실험 (Noise + Blur)
    ├── utils.py             # PSNR 계산 및 Fourier Domain에서의 closed from 구현 code
    ├── lora.py              # NAFNet의 Conv2d 레이어에 적용할 LoRA 모듈  
    └── visualize_deblur.py  # 블러 커널 적용 및 복원 결과 시각화  
  
  
```

## ∎ Requirements  
- Python 3.8+

- PyTorch

- OpenCV (cv2)

- NumPy

- Matplotlib

## ∎ Usage
```text
1. HQS-PnP 모델 사용법
# NAFNet 모델 학습 (Fine-tuning)
python HQS-PnP/train.py 

# Denoising 검증
python HQS-PnP/valid.py 

# 특정 커널에 대한 Deblurring 시각화 및 검증
python HQS-PnP/valid_deblur.py

2. HQS-Unrolled 모델 사용법
다양한 튜닝 전략(Full Fine-tuning, LoRA 등)을 지원합니다.

# 전체 파라미터 학습 (Full Fine-tuning)
python HQS-Unrolled/run_hqs_fnaf.py --full-tuning --full-lr 1e-6

# LoRA(Low-Rank Adaptation)를 적용하여 가볍게 학습
python HQS-Unrolled/run_hqs_fnaf.py --use-lora --lora-rank 8 --lora-alpha 8.0

# 특정 커널(예: Kernel 10)에 대한 복원 결과 시각화
python HQS-Unrolled/visualize_deblur.py --kernel_idx 10 --sigma 50

# 학습된 여러 모델들의 PSNR 및 시각적 결과 비교
python HQS-Unrolled/compare_models.py --sigma 50 --num-samples 10
```


## ∎ Key Results
- Denoising Performance: 강한 노이즈($\sigma=60, 70$) 환경에서도 정량적(PSNR) 수치 향상 및 유의미한 복원 성능을 달성했습니다.   
- Deblurring Generalization: Denoising Task만으로 학습된 모델이 12종의 다양한 Blur Kernel 환경에서도 강력한 Deblurring 일반화 성능을 입증했습니다.  
- Tuning Strategies: HQS-Unrolled 환경에서 모델 전체를 통째로 학습시킨 FFF-NAF 모델이 가장 우수한 성능을 보였습니다.  

## ∎ Future & Extra Works  
- Artifact Removal: 이미지 내 객체 주변에 생기는 Ringing Artifact를 처리하는 방법론 연구가 필요합니다.  
- Other Restoration Tasks: HQS의 Prior Term을 NAFNet 대신 Diffusion 모델로 대체하여 Inpainting이나 Super-Resolution Task로의 확장을 고려할 수 있습니다.  
- Noise Mismatch Experiment: 실제 노이즈(Actual Noise)와 모델에 주입되는 노이즈 맵(Noise Map)의 수치가 불일치할 때(예: $\sigma_{real}=40, \sigma_{map}=70$)의 실험을 통해 선명도를 조절하는 장치로서의 가능성을 확인했습니다.

