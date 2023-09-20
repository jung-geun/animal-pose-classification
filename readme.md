# Animal pose Classification

## Introduction

> 카메라 또는 영상을 이용하여 동물의 행동을 인식하고 분류하는 프로젝트입니다.
> <br>
> 웨어러블 센서를 이용하지 않고 영상만으로 동물의 행동을 분류합니다.
> <br>
> 이 프로젝트에서는 객체를 탐지하는데 YOLOv8을 사용하였고, 객체의 관절 좌표를 추출하는데 deeplabcut을 사용하였습니다.

## Requirements

> - python 3.10
> - deeplabcut
> - tensorflow 2.11
> - yolov8
> - torch 1.12

## Installation

### yolo 설치

yolo 의 경우 의존성 문제가 중요함으로 가장 먼저 설치를 진행합니다.

```python
pip install ultralytics
```

### deeplabcut 설치

```python
pip install "deeplabcut[tf,gui,modelzoo]"
```

deeplabcut 이 pytorch 버전이 1.12 이하만 지원이 되기 때문에 pytorch 버전을 1.12로 맞춰줍니다.

```python
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

윈도우의 의존성 문제로 deeplabcut이 tensorflow 2.10 으로 강제로 내려 설치가 된다
<br>
tensorflow 2.11 버전으로 설치를 진행합니다.

```python
pip install tensorflow==2.11
```

## 파일 구조

```
├── model/
│   ├── LSTM/
│   │   ├── {best_model.h5}         // 가장 성능이 좋은 모델
│   │   ├── {history.pkl}           // 모델의 학습 과정 중 loss, accuracy 등의 정보를 가진 객체
│   │   ├── {model_history.png}     // 모델의 학습 과정 중 loss, accuracy 등의 정보를 그래프로 나타낸 이미지
│   │   └── {model.h5}              // keras 모델
│   ├── CNN/
│   ├── GRU/
│   └── onehot_encoder.pkl          // 행동을 onehot encoding 한 객체
├── angle.py
├── classification.py   // 모델 학습하는 코드
├── data_sensor.py      // 이미지의 관절 좌표를 추출하는 코드
├── detect.py           // yolo를 이용하여 동물의 위치를 찾는 코드
├── predict.py          // 영상에서 동물의 행동을 분류하는 코드
└── README.md
```

## 사용법

### 분류 모델 학습

```python
python classification.py
```

### yolo를 이용하여 동물의 위치를 찾기

```python
python detect.py
```

### deeplabcut을 이용하여 이미지의 관절 좌표를 추출하기

```python
python data_sensor.py
```

### 모델을 이용하여 동물을 탐지, 행동을 분류하기

```python
python predict.py
```
