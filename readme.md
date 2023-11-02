# Animal pose Classification

## Introduction

> 한국정보처리학회 에 투고한 논문의 연구 결과 입니다.
> <br>
> '반려견 영상 실시간 행동 인식 시스템' 논문을 참고하시면 더 자세한 내용을 확인하실 수 있습니다.
> <br>
> <br>
> 이 repository 는 논문 이후에 추가로 연구하여 변경된 사항들이 기록되는 곳 입니다
> <br>
> 웨어러블 센서를 이용하지 않고 영상만으로 동물의 행동을 분류합니다.
> <br>
> 이 프로젝트에서는 객체를 탐지하는데 YOLOv8을 사용하였고, 객체의 관절 좌표를 추출하는데 deeplabcut을 사용하였습니다.

## 파일 구조

```
├── model/
│   ├── LSTM_{window_size}_{label_count}/
│   │   ├── {best_model.h5}         // 가장 성능이 좋은 모델
│   │   ├── {history.pkl}           // 모델의 학습 과정 중 loss, accuracy 등의 정보를 가진 객체
│   │   ├── {model_history.png}     // 모델의 학습 과정 중 loss, accuracy 등의 정보를 그래프로 나타낸 이미지
│   │   └── {model.h5}              // keras 모델
│   ├── GRU_{window_size}_{label_count}/
│   ├── CNN1D_{window_size}_{label_count}/
│   ├── CNN2D_{window_size}_{label_count}/
│   └── onehot_encoder.pkl          // 행동을 onehot encoding 한 객체
├── angle.py
├── data_sensor.py      // 이미지의 관절 좌표를 추출하는 코드
├── detect.py           // yolo를 이용하여 동물의 위치를 찾는 코드
├── predict.py          // 영상에서 동물의 행동을 분류하는 코드
├── train.py            // 모델 학습하는 코드
├── demo_gradio.py      // gradio를 이용하여 모델을 데모하는 코드
├── demo_predict.py     // 모델을 이용하여 데모하는 코드
└── README.md
```

## Requirements

> - python == 3.10
> - deeplabcut[tf,gui]
> - tensorflow == 2.11
> - ultralytics
> - torch == 1.12
> - gradio

## Installation

### yolo 설치

yolo 와 deeplabcut 의 의존성 문제로 패키지들이 깨질 위험이 있기 때문에 conda 환경을 추천합니다.

```bash
conda create -n animal python=3.10
conda activate animal
```

yolo 의 경우 의존성 문제가 중요함으로 가장 먼저 설치를 진행합니다.

```python
pip install ultralytics
```

### deeplabcut 설치

```python
pip install "deeplabcut[tf,gui]"
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

### gradio 설치

```python
pip install gradio
```

## 사용법

### 분류 모델 학습

```python
python train.py
```

### yolo를 이용하여 동물의 위치를 찾기

```python
python detect.py
```

### deeplabcut을 이용하여 이미지의 관절 좌표를 추출하기

data_sensor.py 에서 ini_DLC 함수를 사용하여 deeplabcut 의 모델을 적용
<br>
get_img_coord 함수를 사용하여 이미지의 관절 좌표를 추출 할 수 있습니다

### 모델을 이용하여 동물을 탐지, 행동을 분류하기

```python
python predict.py
```

### 모델을 이용하여 데모하기

웹을 통해 모델을 데모할 수 있습니다.

```python
python demo_gradio.py
```

## 결과

### 모델 학습 결과

#### accuracy

![result.png](./model/result/result.png)

#### validation accuracy

![val_result.png](./model/result/val_result.png)


## collaborators

[@alsdk6720](https://github.com/alsdk6720)

[@everna12](https://github.com/everna12)
