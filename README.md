🌟 Fourier-Optimized HQS-Based Image Restoration본 프로젝트는 2026 GIST Intelligent Vision Lab (지도교수: 소재웅) 겨울 인턴십 기간 동안 숭실대학교 김종민이 진행한 연구로, HQS(Half-Quadratic Splitting) 알고리즘과 NAFNet을 결합하여 이미지 복원(Denoising & Deblurring)을 수행하는 구현체입니다.복잡한 역문제(Inverse Problem)를 풀기 쉬운 두 개의 작은 하위 문제(Data Term & Prior Term)로 분할하여 최적화하며, 퓨리에 도메인(Fourier domain) 최적화를 활용하여 연산 효율을 높였습니다.🧠 Algorithm OverviewHQS Optimization이미지 복원의 핵심 목적은 노이즈와 블러가 섞인 이미지 $y$에서 깨끗한 원본 이미지 $x$를 복원하는 것이며, 이를 위해 다음의 최적화 문제를 풉니다.$$\min_x ||y - Hx||^2 + \lambda\Phi(x)$$$y$: Noisy observation (입력 이미지)$H$: Degradation operator (블러, 다운샘플링 등)$\Phi(x)$: Prior term (NAFNet denoiser를 통한 규제)$\lambda$: Regularization parameterHQS 알고리즘은 보조 변수 $z$를 도입하여 위 수식을 두 개의 서브 문제(Subproblem)로 분리하여 번갈아가며 풉니다.1. Data Subproblem (Fourier domain closed-form)$$x^{k+1} = \arg\min_x ||y - Hx||^2 + \rho||x - z^{k}||^2$$이를 고속 푸리에 변환(FFT)을 통해 계산하면 블러(Blur) 연산을 닫힌 형태(Closed-form)로 역산할 수 있습니다.$$x^{k+1} = \mathcal{F}^{-1} \left[ \frac{\mathcal{F}(H)^* \cdot \mathcal{F}(y) + \rho \cdot \mathcal{F}(z^k)}{|\mathcal{F}(H)|^2 + \rho} \right]$$2. Prior Subproblem (NAFNet denoiser)$$z^{k+1} = \text{NAFNet}(x^{k+1}, \sigma)$$사전 학습된 NAFNet을 활용하여 이미지의 노이즈를 제거하는 데 집중합니다.Network StructurePlaintextInput (noisy y) ──→ [HQS Iterations] ──→ Output (restored x)
                           ↓
               ┌───────────┴───────────┐
               │                       │
         Fourier Data Step      NAFNet Prior
           (closed-form)         (learnable)
               │                       │
               └───────────┬───────────┘
                           ↓
                    Next Iteration
🏗️ Architecture & Methodologies본 레포지토리는 위 알고리즘을 두 가지 주요 접근 방식으로 구현했습니다.HQS-PnP (Plug-and-Play): 수학적 최적화 틀은 고정하고, Prior Step에 사전 학습된 NAFNet을 결합하는 방식HQS-Unrolled: FFT 기반의 Data Step과 NAFNet 기반의 Prior Step이 상호작용하는 반복(Iteration) 과정 전체를 하나의 네트워크로 취급하여 End-to-End로 학습시키는 방식📂 Repository Structure프로젝트는 크게 두 가지 방법론에 따라 폴더가 분리되어 있습니다. 공통적으로 사용되는 12가지 Blur Kernel은 kernels_12.npy에 정의되어 있습니다.Plaintext├── HQS-PnP/                 # Plug-and-Play 기반 알고리즘 구현부
│   ├── train.py             # NAFNet Fine-tuning
│   ├── valid.py             # Denoising 성능 검증
│   ├── valid_deblur.py      # Deblurring 일반화 성능 검증
│   ├── compares.py          # 모델 간 성능 비교 스크립트
│   └── models.py            # HQS_PnP 클래스 및 NAFNet 구조
│
├── HQS-Unrolled/            # Unrolled Network 기반 구현부
│   ├── run_hqs_fnaf.py      # HQS Unrolled 모델 학습 및 테스트 (Full-tuning, LoRA 등)
│   ├── run_nafnet.py        # 순수 NAFNet(HQS 제외) 학습 코드
│   ├── compare_models.py    # 다양한 튜닝 전략이 적용된 모델 성능 일괄 비교
│   ├── experi.py            # 실제 노이즈와 맵 노이즈 불일치(Mismatch) 실험
│   ├── lora.py              # NAFNet의 Conv2d 레이어에 적용할 LoRA 모듈
│   └── visualize_deblur.py  # 블러 커널 적용 및 복원 결과 시각화
│
└── kernels_12.npy           # 12종류의 Blur Kernel 데이터 (공통)
⚙️ RequirementsPython 3.8+PyTorchOpenCV (cv2)NumPyMatplotlib🚀 Usage1. HQS-PnP 모델 사용법HQS 구조 내에서 NAFNet을 부품처럼 활용하는 방식입니다.Bash# NAFNet 모델 학습 (Fine-tuning)
python HQS-PnP/train.py 

# Denoising 검증
python HQS-PnP/valid.py 

# 특정 커널에 대한 Deblurring 시각화 및 검증
python HQS-PnP/valid_deblur.py 
2. HQS-Unrolled 모델 사용법HQS 과정을 Network에 풀어내어 (Unrolled) 학습하는 방식입니다. 다양한 튜닝 전략(Full Fine-tuning, LoRA 등)을 지원합니다.Bash# 전체 파라미터 학습 (Full Fine-tuning)
python HQS-Unrolled/run_hqs_fnaf.py --full-tuning --full-lr 1e-6

# LoRA(Low-Rank Adaptation)를 적용하여 가볍게 학습
python HQS-Unrolled/run_hqs_fnaf.py --use-lora --lora-rank 8 --lora-alpha 8.0

# 특정 커널(예: Kernel 10)에 대한 복원 결과 시각화
python HQS-Unrolled/visualize_deblur.py --kernel_idx 10 --sigma 50

# 학습된 여러 모델들의 PSNR 및 시각적 결과 비교
python HQS-Unrolled/compare_models.py --sigma 50 --num-samples 10
📊 Key ResultsDenoising Performance: 강한 노이즈($\sigma=50, 70$) 환경에서도 정량적(PSNR) 수치 향상 및 고주파 디테일의 성공적인 복원을 확인했습니다.Deblurring Generalization: Denoising Task만으로 학습된 FFF-NAF(Full Fine-Tuning) 모델이 12종의 다양한 Blur Kernel 환경에서도 강력한 Deblurring 일반화 성능을 입증했습니다.Tuning Strategies: HQS-Unrolled 환경에서 모델 전체를 학습하는 FFF-NAF가 가장 높은 성능을 보였으며, 효율적인 파라미터 학습을 위해 LoRA를 도입한 LoF-NAF 모델도 유의미한 성능을 기록했습니다.
