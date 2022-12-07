# Yoga Pose Recognition

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Run](#run)
- [Directory Structure](#directory-structure)

## Description

[//]: # "Title of the project"

### 2022-1 Capstone Design Project

이 프로젝트는 실시간 영상을 통해 요가 자세를 인식하는 프로그램입니다. <br>
요가 자세를 인식하고 잘못된 자세를 보정해주는 프로그램을 만들기 위해 시작되었습니다.<br>
현재 데이터셋은 5가지 요가 자세로 구성되어 있습니다.<br>

```
frames/
├── downdog
├── goddess
├── plank
├── tree
└── warrior
```

## Installation

[//]: # "Title of the project"

가상환경이 이미 설치되어 있을 경우 아래의 명령어를 통해 프로그램을 실행할 수 있습니다.<br>

```bash
$ git clone https://github.com/jung-geun/yoga-pose-recognition.git
$ cd fitness
$ pip install -r requirements.txt
```

시스템에 anaconda를 설치 하지 않을때는 아래의 명령어를 통해 설치해주세요.<br>
간단하게 miniconda를 설치하고 가상환경을 생성해주는 스크립트를 실행하면 됩니다.<br>
conda 가상환경 yoga를 생성하고, 필요한 패키지를 설치합니다.<br>

```bash
$ git clone https://github.com/jung-geun/yoga-pose-recognition.git
$ cd fitness
$ install.bat
```

## Run

[//]: # "Title of the project"

본인의 환경으로 실행

```bash
$ python gui.py
```

기본 권장 실행

```bash
$ run.bat
```

실행 중 모델이 정상적으로 저장이 되지 않을 경우 model 디렉토리가 존재하는지 확인해주세요.<br>
model 디렉토리가 정상적으로 생성되지 않을 경우 권한이 없는 경우가 많습니다.<br>
윈도우의 경우 권한이 없는 경우 관리자 권한으로 실행해주세요.<br>
리눅스의 경우 현재 디텍토리의 권한을 확인해주세요.<br>

## Remove

생성한 miniconda 가상환경을 삭제합니다.

```bash
$ remove.bat
```

## Directory Structure

[//]: # "Title of the project"

```
├── README.md
├── .gitignore
├── /frames - 학습할 이미지들이 들어가는 디렉토리
│   ├── /downdog - y 값
│   │   ├── 1.jpg - x 값
│   │   ├── 2.jpg
│   │   └── ...
│   ├── /goddess
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   └── ...
├── /model - 학습된 모델이 저장되는 디렉토리
│   ├── *.pkl - 학습된 모델
├── /install.bat - 설치 스크립트
├── /run.bat - 실행 스크립트
├── /remove.bat - 삭제 스크립트
├── /gui.py - pyqt5로 구현한 프로그램
├── /gui.ui - pyqt5로 구현한 프로그램의 ui
├── /video_Preprocessing.py - 프레임 단위로 나뉜 영상에서
├── /video_inference.py - 비디오로 추론하는 코드
├── /data.csv - 학습에 사용할 데이터 정보가 담긴 csv 파일
├── /pose.json - 전처리된 데이터
└── /wget.exe - 파일 다운로드를 위한 wget

```

install.bat - 기본 패키지 설치 및 환경 설정

run.bat - 프로그램 실행

remove.bat - 프로그램 및 패키지, 작업환경 제거

\*\* 프로그램 경로에 아스키코드로 표현되지 않는 문자가 포함되어 있으면 작동하지 않습니다

1. 영상 전처리 preprocessing (frames - > skeleton data)

- 영상의 경로를 csv 로 저장
- 프레임단위로 나뉘어져있는 영상을 스켈레톤 데이터로 변환
- 영상의 스켈레톤 추출후 각 관절의 각도를 json 파일로 저장

2. 모델 학습 train

- json 파일을 호출
- 모델 학습(random forest)
- 학습된 모델을 저장

3. 모델 테스트 test

- 실시간 영상 스트리밍
- 학습된 모델을 불러옴
- 모델 테스트

### 참고자료 <br>
-- 스켈레톤 추정
https://google.github.io/mediapipe/solutions/pose.html <br>
-- wget 다운로드
https://www.gnu.org/software/wget/ <br>
