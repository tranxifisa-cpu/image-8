"""Tab 2: DCGAN 生成。"""
import os
import io
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from utils.data_loader import get_dataloader
from utils.visualization import make_image_grid
from utils.session_manager import get_device
from models.dcgan import (Generator, Discriminator, train_dcgan,
                          save_dcgan_checkpoint, load_dcgan_checkpoint)

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')


def render_dcgan_tab() -> None:
    """渲染 'DCGAN 生成' Tab 的全部 UI。"""
    st.header("DCGAN 图像生成")

    # ── 控制栏 ──
    col1, col2, col3 = st.columns(3)
    with col1:
        dataset_name = st.selectbox("数据集", ["mnist", "fashion_mnist"], key="dcgan_dataset")
    with col2:
        noise_dim = st.slider("噪声维度", 20, 200, 100, 10, key="dcgan_noise_dim")
    with col3:
        grid_size = st.slider("每行样本数", 4, 12, 8, key="dcgan_grid_size")

    # ── Session state ──
    for key, default in [
        ('dcgan_g', None), ('dcgan_d', None), ('dcgan_trained', False),
        ('dcgan_noise', None), ('dcgan_fake_imgs', None), ('dcgan_scores', None),
        ('dcgan_ref_grid', None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    g_ckpt = os.path.join(CHECKPOINT_DIR, f'dcgan_g_{dataset_name}.pt')
    d_ckpt = os.path.join(CHECKPOINT_DIR, f'dcgan_d_{dataset_name}.pt')

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        train_btn = st.button("训练 DCGAN", key="dcgan_train_btn", type="primary")
    with col_btn2:
        gen_btn = st.button("生成新样本", key="dcgan_gen_btn")

    # ── 训练 ──
    if train_btn:
        with st.status("训练 DCGAN 中...", expanded=True) as status:
            loader = get_dataloader(dataset_name, batch_size=128, train=True)
            progress_bar = st.progress(0)

            def callback(epoch, total, g_loss, d_loss):
                progress_bar.progress(epoch / total,
                    text=f"Epoch {epoch}/{total} — G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}")

            g, d = train_dcgan(loader, noise_dim=noise_dim, epochs=50,
                               progress_callback=callback)
            save_dcgan_checkpoint(g, d, g_ckpt, d_ckpt)
            st.session_state.dcgan_g = g
            st.session_state.dcgan_d = d
            st.session_state.dcgan_trained = True
            progress_bar.empty()
            status.update(label="训练完成!", state="complete")

    # ── 自动加载 ──
    if not st.session_state.dcgan_trained:
        if os.path.exists(g_ckpt) and os.path.exists(d_ckpt):
            st.info("发现已有模型权重，正在加载...")
            g, d = load_dcgan_checkpoint(g_ckpt, d_ckpt)
            st.session_state.dcgan_g = g
            st.session_state.dcgan_d = d
            st.session_state.dcgan_trained = True
            st.success("模型权重已加载！")

    # ── 生成新样本 ──
    if gen_btn and st.session_state.dcgan_trained:
        device = get_device()
        g = st.session_state.dcgan_g.to(device)
        d = st.session_state.dcgan_d.to(device)
        g.eval()
        d.eval()
        n = grid_size * grid_size
        noise = torch.randn(n, noise_dim, device=device)
        with torch.no_grad():
            fake_imgs = g(noise)
            scores = d(fake_imgs)
        st.session_state.dcgan_noise = noise.cpu()
        st.session_state.dcgan_fake_imgs = fake_imgs.cpu()
        st.session_state.dcgan_scores = scores.cpu().numpy().flatten()

    # ── 结果展示 ──
    if st.session_state.dcgan_fake_imgs is not None:
        st.subheader("生成样本网格")

        col_grid, col_score = st.columns([3, 1])
        with col_grid:
            st.image(make_image_grid(st.session_state.dcgan_fake_imgs, n_cols=grid_size),
                     use_container_width=True)
        with col_score:
            st.caption("**判别器分数 D(G(z))**")
            scores = st.session_state.dcgan_scores
            sorted_idx = np.argsort(scores)[::-1]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=np.arange(len(scores)),
                y=scores[sorted_idx],
                marker_color=np.where(scores[sorted_idx] > 0.5, '#2ca02c', '#d62728'),
            ))
            fig.update_layout(
                title='降序排列', xaxis_title='样本序号', yaxis_title='D(G(z))',
                height=400, yaxis_range=[0, 1], margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── 噪声热力图 ──
        st.subheader("噪声向量可视化")
        selected_idx = st.slider("选择样本查看噪声", 0, len(scores) - 1, 0,
                                 key="noise_inspect")
        noise_vec = st.session_state.dcgan_noise[selected_idx].numpy()
        noise_2d = noise_vec.reshape(10, -1)
        fig2, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(noise_2d, cmap='RdBu_r', aspect='auto')
        plt.colorbar(im, ax=ax)
        ax.set_title(f'噪声向量 (样本 {selected_idx})')
        buf = io.BytesIO()
        fig2.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig2)
        buf.seek(0)
        from PIL import Image
        st.image(Image.open(buf), use_container_width=True)

        # ── 保存参考 ──
        if st.button("保存当前网格作为参考", key="dcgan_ref_btn"):
            st.session_state.dcgan_ref_grid = st.session_state.dcgan_fake_imgs.clone()
            st.success("参考网格已保存！")

        # ── 对比 ──
        if st.session_state.dcgan_ref_grid is not None:
            st.subheader("当前 vs 参考对比")
            col_cur, col_ref = st.columns(2)
            with col_cur:
                st.caption("**当前生成**")
                st.image(make_image_grid(st.session_state.dcgan_fake_imgs, n_cols=grid_size),
                         use_container_width=True)
            with col_ref:
                st.caption("**参考网格**")
                st.image(make_image_grid(st.session_state.dcgan_ref_grid, n_cols=grid_size),
                         use_container_width=True)
    else:
        if not st.session_state.dcgan_trained:
            st.info("👆 请先点击「训练 DCGAN」开始训练（约需 5 分钟），或将权重放入 checkpoints/。")
