# SNN-MNIST 스파이킹 신경망 탐구
# SNN-MNIST Spiking Neural Network Exploration

---

## 🚀 프로젝트 개요 / Project Overview

이 저장소는 고전적인 **MNIST 손글씨 숫자 분류** 작업을 위한 간단한 SNN 모델을 구현하여 **스파이킹 신경망(SNN)**의 매혹적인 세계를 탐구합니다. 기존의 인공 신경망(ANN)과 달리, SNN은 생물학적 뇌를 모방하여 이산적인 스파이크 이벤트를 사용하여 작동하며, 특히 뉴로모픽 하드웨어에서 뛰어난 에너지 효율성과 처리 능력을 약속합니다.

This repository delves into the fascinating world of **Spiking Neural Networks (SNNs)** by implementing a simple SNN model for the classic **MNIST handwritten digit classification** task. Unlike traditional Artificial Neural Networks (ANNs), SNNs operate using discrete spike events, mimicking the biological brain, which promises superior energy efficiency and processing capabilities for certain applications, especially on neuromorphic hardware.

이 프로젝트는 특별히 다음에 중점을 둡니다:
* SNN의 핵심 개념인 스파이크 인코딩과 스파이크 시퀀스 이해
* PyTorch의 강력한 프레임워크를 활용하는 사용자 친화적인 `snnTorch` 라이브러리를 사용한 SNN 분류기 구축
* 뉴런 스파이킹 활동과 멤브레인 전위와 같은 SNN의 동적 행동 시각화로 직관적 통찰 획득
* 잘 알려진 데이터세트에 대한 SNN 훈련 과정 시연

This project specifically focuses on:
* Understanding the core concepts of SNNs, including spike encoding and spike sequences
* Building an SNN classifier using the user-friendly `snnTorch` library, which leverages PyTorch's robust framework
* Visualizing the dynamic behavior of SNNs, such as neuron spiking activity and membrane potentials, to gain intuitive insights
* Demonstrating the training process of SNNs for a well-known dataset

---

## 🎬 SNN 네트워크 시각화 / SNN Network Visualization

다음은 SNN이 MNIST 숫자를 처리하는 실시간 과정을 보여주는 애니메이션입니다:

Here's an animation showing the real-time process of how our SNN processes MNIST digits:

![SNN Network Animation](snn_networkx_animation.gif)

**시각화 설명 / Visualization Description:**
- **입력층 (Input Layer)**: 784개 픽셀 (28×28 격자 배치) / 784 pixels arranged in 28×28 grid
- **은닉층 (Hidden Layer)**: 100개 뉴런 (원형 배치) / 100 neurons in circular arrangement  
- **출력층 (Output Layer)**: 10개 클래스 뉴런 (수직 배치) / 10 class neurons in vertical arrangement
- **빨간색 노드**: 스파이크 발생 / Spiking neurons
- **색상 그라데이션**: 멤브레인 전압 강도 / Membrane potential intensity
- **연결선**: 활성화된 시냅스 연결 / Active synaptic connections

---

## ✨ 주요 기능 / Features

**한국어:**
* **MNIST 데이터 준비**: 정적 이미지 데이터를 위한 맞춤형 스파이크 인코딩 (예: Rate Coding)
* **간단한 SNN 구조**: `snntorch.Leaky` 뉴런을 사용한 다층 SNN 구현
* **훈련 및 평가**: SNN 훈련을 위한 대리 기울기를 사용한 시간 역전파(BPTT)
* **스파이크 동역학 시각화**: 시간에 따른 스파이크 발생과 뉴런 활성화 플롯
* **3D 네트워크 시각화**: NetworkX를 활용한 실시간 뉴런 네트워크 애니메이션
* **M1 Mac 최적화**: Apple Silicon에서 가속 훈련을 위한 PyTorch의 Metal Performance Shaders (MPS) 활용

**English:**
* **MNIST Data Preparation**: Custom spike encoding (e.g., Rate Coding) for static image data
* **Simple SNN Architecture**: Implementation of a multi-layer SNN using `snntorch.Leaky` neurons
* **Training & Evaluation**: Backpropagation Through Time (BPTT) with surrogate gradients for SNN training
* **Spike Dynamics Visualization**: Plotting spike occurrences and neuron activations over time
* **3D Network Visualization**: Real-time neuron network animation using NetworkX
* **M1 Mac Optimization**: Utilizes PyTorch's Metal Performance Shaders (MPS) for accelerated training on Apple Silicon

---

## 🛠️ 사용 기술 / Technologies Used

**Python 기반 도구들 / Python-based Tools:**
* **Python 3.x**
* **PyTorch**: 기본 딥러닝 프레임워크 / The underlying deep learning framework
* **snnTorch**: PyTorch 기반 SNN 라이브러리 / A PyTorch-based library for SNNs
* **NetworkX**: 네트워크 그래프 시각화 / Network graph visualization
* **Matplotlib**: 데이터 및 스파이크 시각화 / For data and spike visualization
* **Jupyter Notebook/Lab**: 대화형 개발 및 프리젠테이션 / For interactive development and presentation
* **Anaconda/Miniforge**: 환경 관리 / For environment management

---

## 🚀 시작하기 / Getting Started

다음 단계를 따라 환경을 설정하고 프로젝트를 로컬에서 실행하세요.

Follow these steps to set up the environment and run the project locally.

### 사전 요구사항 / Prerequisites

`conda` (M1 Mac의 경우 Miniforge 권장)가 설치되어 있는지 확인하세요.

Ensure you have `conda` (Miniforge recommended for M1 Mac) installed.

### 설치 / Installation

1. **저장소 복제 / Clone the repository:**
   ```bash
   git clone https://github.com/YourGitHubUsername/SNN-MNIST.git
   cd SNN-MNIST
   ```

2. **Conda 환경 생성 및 활성화 / Create and activate a Conda environment:**
   ```bash
   conda create -n snn_env python=3.10
   conda activate snn_env
   ```

3. **필요한 패키지 설치 / Install the required packages:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install snntorch matplotlib ipykernel notebook networkx
   ```

### 노트북 실행 / Running the Notebook

1. **Jupyter Notebook/Lab 실행 / Launch Jupyter Notebook/Lab:**
   ```bash
   jupyter notebook
   ```

2. **노트북 열기 및 실행 / Open and run the notebook:**
   `snn_mnist_classifier.ipynb` 파일로 이동하여 셀을 실행합니다.
   
   Navigate to the `snn_mnist_classifier.ipynb` file and execute the cells.

### 네트워크 시각화 실행 / Running Network Visualization

**실시간 시각화 / Real-time visualization:**
```bash
python snn_networkx_animation.py
```

**GIF 생성 / Generate GIF:**
파일 내에서 `save_gif=True`로 설정하거나 코드를 수정하세요.

Set `save_gif=True` in the file or modify the code accordingly.

---

## 📊 결과 및 시각화 / Results & Visualizations

**한국어:**
* **훈련 손실 및 정확도 플롯**: 에포크에 따른 모델 성능의 발전 과정 표시
* **스파이크 인코딩 이미지 예시**: 정적 MNIST 숫자가 스파이크 시퀀스로 변환되는 과정 설명
* **출력 뉴런 스파이크 활동**: 주어진 입력에 대한 출력 뉴런의 발화 패턴 시각화로 SNN의 의사결정 과정 시연
* **실시간 네트워크 애니메이션**: 전체 네트워크에서 스파이크 전파와 뉴런 활동의 동적 시각화

**English:**
* **Training Loss & Accuracy Plots**: Showcase how the model's performance evolves over epochs
* **Example Spike Encoded Image**: Illustrate how a static MNIST digit is converted into a spike sequence
* **Output Neuron Spike Activity**: Visualize the firing patterns of the output neurons for a given input, demonstrating the SNN's decision-making process
* **Real-time Network Animation**: Dynamic visualization of spike propagation and neuron activity across the entire network

**모델 성능 / Model Performance:**
- 테스트 정확도 / Test Accuracy: **97.55%**
- 아키텍처 / Architecture: 784 → 100 → 10
- 시간 스텝 / Time Steps: 25

---

## 💡 향후 작업 / Future Work

**한국어:**
* 다양한 스파이크 인코딩 방식 실험 (예: Time-To-First-Spike)
* 더 복잡한 SNN 구조 탐구 (예: SNN-CNN 하이브리드)
* 다양한 SNN 학습 규칙 연구 (예: STDP)
* SNN의 에너지 소비 메트릭 평가
* 진정한 뉴로모픽 데이터를 위한 이벤트 기반 데이터세트(예: N-MNIST)에 모델 적용
* 3D 및 VR/AR 환경에서의 고급 시각화 개발

**English:**
* Experiment with different spike encoding schemes (e.g., Time-To-First-Spike)
* Explore more complex SNN architectures (e.g., SNN-CNN hybrids)
* Investigate different SNN learning rules (e.g., STDP)
* Evaluate energy consumption metrics for the SNN
* Apply the model to event-based datasets (e.g., N-MNIST) for true neuromorphic data
* Develop advanced visualizations in 3D and VR/AR environments

---

## 🎯 주요 파일 / Key Files

* `snn_mnist_classifier.ipynb`: 메인 훈련 및 분석 노트북 / Main training and analysis notebook
* `snn_networkx_animation.py`: NetworkX 네트워크 시각화 스크립트 / NetworkX network visualization script
* `snn_mnist_model.pth`: 훈련된 모델 가중치 / Trained model weights
* `snn_networkx_animation.gif`: 생성된 네트워크 애니메이션 / Generated network animation

---

## 🤝 기여 / Contributing

기여를 환영합니다! 제안이나 개선 사항이 있으시면 이슈를 열거나 풀 리퀘스트를 제출해 주세요.

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

---

## 📄 라이선스 / License

이 프로젝트는 MIT 라이선스 하에 라이선스가 부여됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## 📞 연락처 / Contact

질문이 있으시면 언제든지 humblefirm@gmail.com으로 연락주세요.

If you have any questions, feel free to reach out to me at humblefirm@gmail.com