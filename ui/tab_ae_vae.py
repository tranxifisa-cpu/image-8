"""Tab 1: AE vs VAE 重构对比。"""
import os
import numpy as np
import torch
import streamlit as st

from utils.data_loader import get_dataloader, get_test_images, get_full_test_dataset
from utils.visualization import (
    plot_loss_curves, plot_latent_scatter, make_error_heatmaps, make_image_grid
)
from utils.session_manager import get_device, clear_cuda_memory
from models.autoencoder import Autoencoder, train_autoencoder, save_ae_checkpoint, load_ae_checkpoint
from models.vae import VAE, train_vae, save_vae_checkpoint, load_vae_checkpoint

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')

CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
FASHION_CLASS_NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def render_ae_vae_tab() -> None:
    """渲染 'AE vs VAE' Tab 的全部 UI。"""
    st.header("自编码器 vs VAE 重构对比")

    # ── 控制栏 ──
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 1])
    with col_ctrl1:
        dataset_name = st.selectbox("数据集", ["mnist", "fashion_mnist"], key="ae_vae_dataset")
    with col_ctrl2:
        latent_dim = st.slider("潜空间维度", 2, 20, 2, key="ae_vae_latent_dim")
    with col_ctrl3:
        st.write("")
        train_btn = st.button("训练模型", key="ae_vae_train_btn", type="primary")

    class_names = CLASS_NAMES if dataset_name == 'mnist' else FASHION_CLASS_NAMES

    # ── 初始化 session state ──
    for key, default in [
        ('ae_model', None), ('vae_model', None),
        ('ae_train_losses', []), ('ae_val_losses', []),
        ('vae_total_losses', []), ('vae_recon_losses', []),
        ('vae_kl_losses', []), ('vae_val_total_losses', []),
        ('ae_vae_trained', False), ('test_images', None),
        ('test_labels', None), ('latent_scatter_data', None),
        ('last_dataset', None), ('last_latent_dim', None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # 检测参数变化，清除缓存
    if (st.session_state.last_dataset != dataset_name or
            st.session_state.last_latent_dim != latent_dim):
        st.session_state.ae_vae_trained = False
        st.session_state.latent_scatter_data = None
        st.session_state.test_images = None

    ae_ckpt = os.path.join(CHECKPOINT_DIR, f'ae_{dataset_name}_ld{latent_dim}.pt')
    vae_ckpt = os.path.join(CHECKPOINT_DIR, f'vae_{dataset_name}_ld{latent_dim}.pt')

    # ── 训练 ──
    if train_btn:
        with st.status("训练中...", expanded=True) as status:
            train_loader = get_dataloader(dataset_name, batch_size=128, train=True)
            val_loader = get_dataloader(dataset_name, batch_size=256, train=False)

            progress_bar = st.progress(0)
            status_text = st.empty()

            def ae_callback(epoch, total, train_loss, val_loss):
                progress_bar.progress(epoch / (total * 2),
                                      text=f"AE Epoch {epoch}/{total}")
                status_text.text(
                    f"AE — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            st.write("**训练 Autoencoder...**")
            ae_model, ae_train, ae_val = train_autoencoder(
                train_loader, val_loader, latent_dim=latent_dim, epochs=20,
                progress_callback=ae_callback)
            save_ae_checkpoint(ae_model, ae_ckpt)

            st.write("**训练 VAE...**")

            def vae_callback(epoch, total, train_loss, val_loss):
                progress_bar.progress((20 + epoch) / (total * 2),
                                      text=f"VAE Epoch {epoch}/{total}")
                status_text.text(
                    f"VAE — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            vae_model, vae_total, vae_recon, vae_kl, vae_val = train_vae(
                train_loader, val_loader, latent_dim=latent_dim, epochs=20,
                progress_callback=vae_callback)
            save_vae_checkpoint(vae_model, vae_ckpt)

            st.session_state.ae_model = ae_model
            st.session_state.vae_model = vae_model
            st.session_state.ae_train_losses = ae_train
            st.session_state.ae_val_losses = ae_val
            st.session_state.vae_total_losses = vae_total
            st.session_state.vae_recon_losses = vae_recon
            st.session_state.vae_kl_losses = vae_kl
            st.session_state.vae_val_total_losses = vae_val
            st.session_state.ae_vae_trained = True
            st.session_state.last_dataset = dataset_name
            st.session_state.last_latent_dim = latent_dim

            progress_bar.empty()
            status.update(label="训练完成!", state="complete")
            st.rerun()

    # ── 加载已有权重 ──
    if not st.session_state.ae_vae_trained:
        if os.path.exists(ae_ckpt) and os.path.exists(vae_ckpt):
            st.info("发现已有模型权重，正在加载...")
            st.session_state.ae_model = load_ae_checkpoint(ae_ckpt)
            st.session_state.vae_model = load_vae_checkpoint(vae_ckpt)
            st.session_state.ae_vae_trained = True
            st.session_state.last_dataset = dataset_name
            st.session_state.last_latent_dim = latent_dim
            st.success("模型权重已加载！")
            st.rerun()

    # ── 展示结果 ──
    if st.session_state.ae_vae_trained:
        ae_model = st.session_state.ae_model
        vae_model = st.session_state.vae_model
        device = get_device()

        ae_model.to(device)
        vae_model.to(device)

        # 加载测试图像
        if st.session_state.test_images is None:
            imgs, lbls = get_test_images(dataset_name, n=25)
            st.session_state.test_images = imgs
            st.session_state.test_labels = lbls

        test_imgs = st.session_state.test_images.to(device)
        test_lbls = st.session_state.test_labels

        # ── 重构对比 ──
        st.subheader("重构对比")
        ae_model.eval()
        vae_model.eval()
        with torch.no_grad():
            ae_recon, _ = ae_model(test_imgs)
            vae_recon, _, _ = vae_model(test_imgs)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("**原始输入**")
            st.image(make_image_grid(test_imgs.cpu(), n_cols=5), use_container_width=True)
        with col2:
            st.caption("**AE 重构**")
            st.image(make_image_grid(ae_recon.cpu(), n_cols=5), use_container_width=True)
        with col3:
            st.caption("**VAE 重构**")
            st.image(make_image_grid(vae_recon.cpu(), n_cols=5), use_container_width=True)

        # ── 误差热力图 ──
        st.subheader("重构误差热力图 (|原图 - 重构|)")
        col_heat1, col_heat2 = st.columns(2)
        with col_heat1:
            st.caption("**AE 重构误差**")
            st.image(make_error_heatmaps(test_imgs.cpu(), ae_recon.cpu(), n_cols=5),
                     use_container_width=True)
        with col_heat2:
            st.caption("**VAE 重构误差**")
            st.image(make_error_heatmaps(test_imgs.cpu(), vae_recon.cpu(), n_cols=5),
                     use_container_width=True)

        # ── Loss 曲线 ──
        st.subheader("训练 Loss 曲线")
        loss_fig = plot_loss_curves(
            st.session_state.ae_train_losses,
            st.session_state.ae_val_losses,
            st.session_state.vae_total_losses,
            st.session_state.vae_recon_losses,
            st.session_state.vae_kl_losses,
        )
        st.plotly_chart(loss_fig, use_container_width=True)

        # ── 潜空间可视化（仅 latent_dim=2）──
        if latent_dim == 2:
            st.subheader("潜空间可视化")

            # 生成全量测试集的潜空间编码
            if st.session_state.latent_scatter_data is None:
                with st.spinner("正在编码测试集到潜空间..."):
                    full_imgs, full_lbls = get_full_test_dataset(dataset_name)
                    vae_model.eval()
                    with torch.no_grad():
                        mu, _ = vae_model.encode(full_imgs.to(device))
                    st.session_state.latent_scatter_data = {
                        'z': mu.cpu().numpy(),
                        'labels': full_lbls.numpy(),
                        'images': full_imgs,
                    }

            scatter_data = st.session_state.latent_scatter_data

            col_scatter, col_interp = st.columns([3, 2])

            with col_scatter:
                scatter_fig = plot_latent_scatter(
                    scatter_data['z'], scatter_data['labels'], class_names)

                # 点击解码
                clicked = st.plotly_chart(
                    scatter_fig, use_container_width=True,
                    key="latent_scatter", on_select="rerun")

                if (clicked and hasattr(clicked, 'selection') and
                        clicked.selection and clicked.selection.points):
                    pt = clicked.selection.points[0]
                    idx = pt.get('point_index', 0)
                    z_point = torch.tensor(
                        scatter_data['z'][idx], dtype=torch.float32
                    ).unsqueeze(0).to(device)
                    vae_model.eval()
                    with torch.no_grad():
                        decoded = vae_model.decode_from_latent(z_point)
                    st.image(make_image_grid(decoded.cpu(), n_cols=1),
                             width=150,
                             caption=f"索引 {idx}, 类别 {class_names[scatter_data['labels'][idx]]}")

            with col_interp:
                st.caption("**潜空间插值**")
                n_total = len(scatter_data['labels'])
                idx_a = st.number_input("样本 A 索引", 0, n_total - 1, 0, key="interp_a")
                idx_b = st.number_input("样本 B 索引", 0, n_total - 1,
                                        min(100, n_total - 1), key="interp_b")
                n_steps = st.slider("插值步数", 2, 16, 8, key="interp_steps")

                if st.button("生成插值序列", key="interp_btn"):
                    z_a = torch.tensor(scatter_data['z'][idx_a], dtype=torch.float32)
                    z_b = torch.tensor(scatter_data['z'][idx_b], dtype=torch.float32)
                    alphas = torch.linspace(0, 1, n_steps)
                    z_interp = torch.stack(
                        [(1 - a) * z_a + a * z_b for a in alphas]).to(device)
                    vae_model.eval()
                    with torch.no_grad():
                        decoded_seq = vae_model.decode_from_latent(z_interp)
                    st.image(make_image_grid(decoded_seq.cpu(), n_cols=n_steps),
                             use_container_width=True,
                             caption=f"{class_names[scatter_data['labels'][idx_a]]} → "
                                     f"{class_names[scatter_data['labels'][idx_b]]}")
        else:
            st.info("💡 潜空间散点图仅在 latent_dim=2 时可用。请将潜空间维度设为 2 后重新训练。")
    else:
        if not train_btn:
            st.info("👆 请先点击「训练模型」开始训练（约需 2 分钟），或将已有权重放入 checkpoints/ 目录。")
