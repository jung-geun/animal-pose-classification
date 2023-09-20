from ctypes import resize
import cv2
import tensorflow as tf
from dlclive import DLCLive, Processor
# https://github.com/DeepLabCut/DeepLabCut-live

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
        


def main():
    # Processor 초기화 (필요한 경우)
    dlc_proc = Processor()

    # DLCLive 객체 초기화
    dlc_live = DLCLive('/home/dlc/DLC/_mina/project/DLC/preLabeled_type(Labrador_Retriever)-test01-2023-09-19/exported-models/DLC_preLabeled_type(Labrador_Retriever)_resnet_50_iteration-0_shuffle-1', processor=dlc_proc,
                       resize=0.5)

    # 비디오 캡처 시작
    # cap = cv2.VideoCapture(0)  # 웹캠 사용
    cap = cv2.VideoCapture('/home/dlc/DLC/_mina/project/DLC/preLabeled_type(Labrador_Retriever)-test01-2023-09-19/videos/dog-walkrun-085352.avi')  # 비디오 파일 사용

    # 좌표를 저장할 리스트 초기화
    coordinates = []
    
    # 사용자에게 어떤 함수를 실행할지 물어보기
    choice = input("Which function do you want to run? (1: get_coordinates, 2: draw_pose): ")

    if choice == '1':
        get_coordinates(cap, dlc_live, coordinates)
    elif choice == '2':
        draw_pose(cap, dlc_live)
    else:
        print("잘못된 입력입니다. Please enter 1 or 2.")


# 예측된 키포인트 좌표를 리스트에 저장
def get_coordinates(cap, dlc_live, coordinates):
    while True:
        # 비디오에서 프레임 캡처
        ret, frame = cap.read()

        if not ret:
            break

        dlc_live.init_inference(frame)
        # 프레임에 대해 키포인트 매핑 수행
        pose = dlc_live.get_pose(frame)

        # 좌표 출력 및 저장
        print(pose)
        coordinates.append(pose)


# 예측된 키포인트를 프레임에 그리기
def draw_pose(cap, dlc_live):
    while True:
        # 비디오에서 프레임 캡처
        ret, frame = cap.read()

        if not ret:
            break

        dlc_live.init_inference(frame)
        # 프레임에 대해 키포인트 매핑 수행
        pose = dlc_live.get_pose(frame)

        # 예측된 키포인트를 프레임에 그리기
        for i in range(len(pose)):
            cv2.circle(frame, (int(pose[i][0]), int(pose[i][1])), 3, (0, 0, 255), -1)
            cv2.putText(frame, str(i), (int(pose[i][0]), int(pose[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 프레임 표시
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.imshow("Frame", frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 비디오 캡처 종료 및 창 닫기
    cap.release()
    cv2.destroyAllWindows()


main()


# from dlclive import DLCLive, Processor
# dlc_proc = Processor()
# dlc_live = DLCLive(<path to exported model directory>, processor=dlc_proc)
# dlc_live.init_inference(<your image>)
# dlc_live.get_pose(<your image>)
