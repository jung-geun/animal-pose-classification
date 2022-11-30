# fitness

## Description

[//]: # "Short description of the project"

[사용법]

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
   <!-- - 테스트 결과를 저장 -->

[출처]
-- 스켈레톤 추정
https://google.github.io/mediapipe/solutions/pose.html

<!-- -- 영상 프레임 단위로 추출 -->
<!-- https://thinking-developer.tistory.com/61 -->
