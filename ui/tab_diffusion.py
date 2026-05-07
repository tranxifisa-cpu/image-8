"""Tab 3: 文本到图像（Diffusers）。"""
import time
import streamlit as st
from PIL import Image

from utils.session_manager import get_device, clear_cuda_memory
from models.diffusion_loader import load_sd_pipeline, generate_image


def render_diffusion_tab() -> None:
    """渲染 '文本到图像' Tab 的全部 UI。"""
    st.header("文本到图像生成 (Stable Diffusion 1.5)")

    # ── Session state ──
    for key, default in [
        ('diffusion_pipe', None), ('diffusion_loaded', False),
        ('diffusion_runs', []),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── 侧边栏 ──
    with st.sidebar:
        st.subheader("生成参数")

        prompt = st.text_area("正向提示词",
            "A cute cat sitting on a cloud, digital art, high quality",
            key="diff_prompt")
        neg_prompt = st.text_area("负向提示词",
            "blurry, low quality, distorted, ugly, bad anatomy",
            key="diff_neg_prompt")

        col1, col2 = st.columns(2)
        with col1:
            steps = st.slider("采样步数", 5, 50, 20, 5, key="diff_steps")
        with col2:
            cfg = st.slider("CFG 引导强度", 1.0, 20.0, 7.5, 0.5, key="diff_cfg")

        col3, col4 = st.columns(2)
        with col3:
            use_seed = st.checkbox("固定种子", True, key="diff_use_seed")
            seed = st.number_input("随机种子", 0, 2**31 - 1, 42, key="diff_seed",
                                   disabled=not use_seed)
        with col4:
            resolution = st.selectbox("分辨率", [256, 384, 512], index=2,
                                      key="diff_resolution")

        st.divider()

        load_btn = st.button("加载 SD 模型", key="diff_load_btn")
        gen_btn = st.button("生成图像", key="diff_gen_btn", type="primary",
                            use_container_width=True)

        st.divider()
        st.caption(f"已保存结果: {len(st.session_state.diffusion_runs)}/4")
        if st.button("清空全部结果", key="diff_clear_btn"):
            st.session_state.diffusion_runs = []
            st.rerun()

    # ── 加载模型 ──
    if load_btn:
        with st.spinner("正在加载 Stable Diffusion 1.5... (首次需下载约 5GB)"):
            st.session_state.diffusion_pipe = load_sd_pipeline()
            st.session_state.diffusion_loaded = True
        st.success("SD 1.5 模型已加载！")

    if not st.session_state.diffusion_loaded:
        auto_load = st.checkbox("自动加载模型（需约 4GB 显存，首次下载约 5GB）",
                                key="diff_auto_load")
        if auto_load:
            with st.spinner("正在加载 Stable Diffusion 1.5..."):
                st.session_state.diffusion_pipe = load_sd_pipeline()
                st.session_state.diffusion_loaded = True
            st.success("模型已加载！")
        else:
            st.info("请先点击侧边栏「加载 SD 模型」。首次加载需下载约 5GB 模型文件。")
            return

    # ── 生成 ──
    if gen_btn:
        if len(st.session_state.diffusion_runs) >= 4:
            st.warning("最多保存 4 组结果。请先清空后再生成。")
        else:
            with st.spinner(f"生成中... (分辨率 {resolution}, {steps} 步)"):
                t0 = time.time()
                img = generate_image(
                    st.session_state.diffusion_pipe,
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    seed=seed if use_seed else None,
                    width=resolution,
                    height=resolution,
                )
                elapsed = time.time() - t0

            run = {
                'params': {
                    'prompt': prompt,
                    'negative_prompt': neg_prompt,
                    'steps': steps,
                    'cfg': cfg,
                    'seed': seed if use_seed else 'random',
                    'resolution': resolution,
                },
                'image': img,
                'time': elapsed,
            }
            st.session_state.diffusion_runs.append(run)
            st.success(f"生成完成！耗时 {elapsed:.1f}s")

    # ── 结果展示 ──
    runs = st.session_state.diffusion_runs
    if runs:
        st.subheader("生成结果对比")

        view_mode = st.radio("对比模式",
            ["2×2 网格", "并排对比 (A/B)", "滑块分割"],
            horizontal=True, key="diff_view_mode")

        if view_mode == "2×2 网格":
            cols = st.columns(2)
            for i, run in enumerate(runs):
                p = run['params']
                with cols[i % 2]:
                    st.image(run['image'], use_container_width=True)
                    st.caption(
                        f"**#{i+1}** | Steps: {p['steps']} | CFG: {p['cfg']:.1f} | "
                        f"Seed: {p['seed']} | {p['resolution']}px | {run['time']:.1f}s"
                    )

        elif view_mode == "并排对比 (A/B)":
            if len(runs) >= 2:
                col_a, col_b = st.columns(2)
                a_idx = st.selectbox("选择 A", range(len(runs)), key="diff_ab_a",
                    format_func=lambda i: f"结果 {i+1}")
                b_idx = st.selectbox("选择 B", range(len(runs)), key="diff_ab_b",
                    format_func=lambda i: f"结果 {i+1}")
                with col_a:
                    st.image(runs[a_idx]['image'], use_container_width=True)
                    st.caption(f"**A** — Steps: {runs[a_idx]['params']['steps']}, "
                               f"CFG: {runs[a_idx]['params']['cfg']:.1f}")
                with col_b:
                    st.image(runs[b_idx]['image'], use_container_width=True)
                    st.caption(f"**B** — Steps: {runs[b_idx]['params']['steps']}, "
                               f"CFG: {runs[b_idx]['params']['cfg']:.1f}")
            else:
                st.info("需要至少 2 个结果才能进行 A/B 对比。")

        elif view_mode == "滑块分割":
            if len(runs) >= 2:
                a_idx = st.selectbox("选择左图", range(len(runs)), key="diff_split_a",
                    format_func=lambda i: f"结果 {i+1}")
                b_idx = st.selectbox("选择右图", range(len(runs)), key="diff_split_b",
                    format_func=lambda i: f"结果 {i+1}")

                split_pos = st.slider("分割位置", 0, 100, 50, key="diff_split_pos")

                # Resize to common size for split
                w = 512
                img_a = runs[a_idx]['image'].resize((w, w))
                img_b = runs[b_idx]['image'].resize((w, w))

                split_x = int(w * split_pos / 100)
                left_part = img_a.crop((0, 0, split_x, w))
                right_part = img_b.crop((split_x, 0, w, w))
                combined = Image.new('RGB', (w, w))
                combined.paste(left_part, (0, 0))
                combined.paste(right_part, (split_x, 0))

                st.image(combined, use_container_width=True,
                    caption=f"A (左): Steps={runs[a_idx]['params']['steps']} | "
                            f"B (右): Steps={runs[b_idx]['params']['steps']}")
            else:
                st.info("需要至少 2 个结果才能进行滑块分割对比。")
