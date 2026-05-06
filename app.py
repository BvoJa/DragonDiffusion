from src.demo.download import download_all
download_all()

from src.demo.demo import create_demo_move, create_demo_appearance, create_demo_drag, create_demo_face_drag, create_demo_paste
from src.demo.model import DragonModels

import cv2
import gradio as gr

if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

# Giải quyết lỗi: AttributeError: module 'torch' has no attribute 'xpu' (nếu có)
if not hasattr(torch, 'xpu'):
    class MockXPU:
        def empty_cache(self): pass
    torch.xpu = MockXPU()

# main demo
pretrained_model_path = "runwayml/stable-diffusion-v1-5"
model = DragonModels(pretrained_model_path=pretrained_model_path)

DESCRIPTION = '# 🐉🐉[DragonDiffusion V1.0](https://github.com/MC-E/DragonDiffusion)🐉🐉'

DESCRIPTION += f'<p>Gradio demo for [DragonDiffusion](https://arxiv.org/abs/2307.02421) and [DiffEditor](https://arxiv.org/abs/2307.02421). If it is helpful, please help to recommend [[GitHub Repo]](https://github.com/MC-E/DragonDiffusion) to your friends 😊 </p>'

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

demo.queue(concurrency_count=3, max_size=20)
demo.launch(server_name="0.0.0.0")