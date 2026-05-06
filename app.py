# ==========================================================
# PHẦN VÁ LỖI (BẮT BUỘC PHẢI Ở TRÊN CÙNG - DÒNG 1)
# ==========================================================
import sys
import torch
import huggingface_hub

# Cách vá lỗi triệt để cho 'cached_download'
if not hasattr(huggingface_hub, 'cached_download'):
    # Gán trực tiếp vào module để các lệnh 'from huggingface_hub import cached_download' không bị lỗi
    setattr(huggingface_hub, 'cached_download', huggingface_hub.hf_hub_download)
    sys.modules['huggingface_hub'].cached_download = huggingface_hub.hf_hub_download

# Cách vá lỗi triệt để cho 'torch.xpu'
if not hasattr(torch, 'xpu'):
    class MockXPU:
        def empty_cache(self): pass
        def is_available(self): return False
    torch.xpu = MockXPU()

# ==========================================================
# BẮT ĐẦU IMPORT CÁC THÀNH PHẦN CỦA PROJECT
# ==========================================================
from src.demo.download import download_all
# Chạy download model sau khi đã vá lỗi xong
download_all()

from src.demo.demo import (
    create_demo_move, 
    create_demo_appearance, 
    create_demo_drag, 
    create_demo_face_drag, 
    create_demo_paste
)
from src.demo.model import DragonModels

import cv2
import gradio as gr

# Khởi tạo model
# Lưu ý: "runwayml/stable-diffusion-v1-5" sẽ tự tải về nếu chưa có
pretrained_model_path = "runwayml/stable-diffusion-v1-5"
model = DragonModels(pretrained_model_path=pretrained_model_path)

DESCRIPTION = '# 🐉🐉[DragonDiffusion V1.0](https://github.com/MC-E/DragonDiffusion)🐉🐉'
DESCRIPTION += f'<p>Gradio demo for [DragonDiffusion](https://arxiv.org/abs/2307.02421). Nếu hữu ích hãy ủng hộ repo GitHub nhé! 😊 </p>'

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.TabItem('Appearance Modulation'):
            create_demo_appearance(model.run_appearance)
        with gr.TabItem('Object Moving & Resizing'):
            create_demo_move(model.run_move)
        with gr.TabItem('Face Modulation'):
            create_demo_face_drag(model.run_drag_face)
        with gr.TabItem('Content Dragging'):
            create_demo_drag(model.run_drag)
        with gr.TabItem('Object Pasting'):
            create_demo_paste(model.run_paste)

# Khởi chạy Gradio
demo.queue(max_size=20)
demo.launch(server_name="0.0.0.0")