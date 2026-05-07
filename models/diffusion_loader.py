"""Stable Diffusion 1.5 加载与推理。"""
import os
import torch
import streamlit as st
from utils.session_manager import get_device


def load_sd_pipeline():
    """加载 SD 1.5 pipeline（FP16, attention slicing for 8GB VRAM）。"""
    from diffusers import StableDiffusionPipeline

    # 从 Streamlit Secrets 获取 HF Token（本地 .streamlit/secrets.toml 或 Cloud Dashboard）
    hf_token = None
    try:
        hf_token = st.secrets.get("HF_TOKEN")
    except Exception:
        hf_token = os.environ.get("HF_TOKEN")

    device = get_device()
    pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        safety_checker=None,
        token=hf_token,
    )
    if device.type == 'cuda':
        pipe.enable_attention_slicing()
    pipe = pipe.to(device)
    return pipe


def generate_image(pipe, prompt: str, negative_prompt: str = "",
                   num_inference_steps: int = 20, guidance_scale: float = 7.5,
                   seed: int | None = None, width: int = 512, height: int = 512,
                   progress_callback=None):
    """使用 SD pipeline 生成图像。

    Args:
        progress_callback: 可选，签名为 callback(step: int, total_steps: int)
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        width=width,
        height=height,
        callback=progress_callback,
        callback_steps=1,
    )
    return result.images[0]
