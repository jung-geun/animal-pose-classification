# 모듈 불러오기: import 모듈 이름 - 이미지 사용을 위한 opencv,
# 이미지 저장 파일명 사용을 위한 datetime

# import cv2
# import datetime
#
# video_capture = cv2.VideoCapture(0)
#
# while (True):
#
#     grabbed, frame = video_capture.read()
#     cv2.imshow('Original Video', frame)
#
#     key = cv2.waitKey(1);
#     if key == ord('q'):
#         break
#     elif key == ord('s'):
#         file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") + '.jpg'
#         cv2.imwrite(file, frame)
#         print(file, ' saved')
#
# video_capture.release()
# cv2.destroyAllWindows()

# main.py

# 라이브러리 import
# StreamingResponse를 가져와야함
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from fastapi import FastAPI, File, UploadFile
from typing import List

import uvicorn

# cv2 모듈 import
import cv2
from cv2_conn import get_stream_video

# FastAPI객체 생성
app = FastAPI()

# openCV에서 이미지 불러오는 함수


def video_streaming():
    return get_stream_video()


templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
@app.post("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})


# 스트리밍 경로를 /video 경로로 설정.
@app.get("/video")
def main():
    # StringResponse함수를 return하고,
    # 인자로 OpenCV에서 가져온 "바이트"이미지와 type을 명시
    return StreamingResponse(video_streaming(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/files/")
async def create_files(files: List[bytes] = File()):
    return {"file_sizes": [len(file) for file in files]}


@app.post("/uploadfiles/")
async def create_upload_files(files: List[UploadFile]):
    return {"filenames": [file.filename for file in files]}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=9000)
