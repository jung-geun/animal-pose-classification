"""
  pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
"""
import gradio as gr
import glob
import os

# from detect import analyze
from demo_predict import analyze

theme = gr.themes.Monochrome()
vdo_dir = "./videos2/experimental_videos/"


def video_identity(video):
    label = "Output Video"
    name = os.path.basename(video)  # animal_-_111251 720p.mp4
    # 파일 이름에서 첫 번째 공백의 위치를 찾는다.
    space_index = name.find(" ")
    # 파일 이름에서 마지막 '.'의 위치를 찾는다.
    dot_index = name.rfind(".")
    # 괄호 추가
    vdo_name = (
        name[: space_index + 1]
        + "("
        + name[space_index + 1 : dot_index]
        + ")"
        + name[dot_index:]
    )  # animal_-_111251 (720p).mp4
    name = os.path.splitext(vdo_name)[0]  # animal_-_111251 (720p)
    model_name = glob.glob("./model/" + name + "*")[
        0
    ]  # /home/dlc/DLC/model/animal_-_111251 (720p)-1-2023-10-24
    video = analyze(video, name, model_name)
    return video


with gr.Blocks(theme=theme) as demo:
    with gr.Row():
        org = gr.Video(label="Original Video")
        out = gr.Video(label="Output Video")
    gr.Examples(
        examples=[
            os.path.join(vdo_dir, "animal_-_111251 (720p).mp4"),
            os.path.join(vdo_dir, "chromakey_-_11342 (1080p).mp4"),
            os.path.join(vdo_dir, "dog_-_2296 (720p).mp4"),
            os.path.join(vdo_dir, "dog_-_47312 (720p).mp4"),
            os.path.join(vdo_dir, "dog_-_77503 (720p).mp4"),
            os.path.join(vdo_dir, "dog_-_77534 (720p).mp4"),
            os.path.join(vdo_dir, "dog_-_109814 (720p).mp4"),
            os.path.join(vdo_dir, "dog_-_119587 (720p).mp4"),
        ],
        inputs=org,
        outputs=out,
        cache_examples=False,
    )
    anal_btn = gr.Button("Analyze Video")
    anal_btn.click(fn=video_identity, inputs=org, outputs=out, api_name="dog_behavior")


if __name__ == "__main__":
    demo.launch(share=True)
