"""图像处理 Web 应用 — 主入口。

运行: streamlit run image7/app.py
"""
import streamlit as st

st.set_page_config(
    page_title="图像处理实验室",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="auto",
)

from ui.tab_ae_vae import render_ae_vae_tab
from ui.tab_dcgan import render_dcgan_tab
from ui.tab_diffusion import render_diffusion_tab
from utils.session_manager import clear_cuda_memory


def main():
    st.title("🎨 图像处理实验室")
    st.markdown(
        "自编码器 vs VAE 对比 · DCGAN 生成 · 文本到图像 — "
        "实时调参交互 · 前后效果对比"
    )

    tab_labels = ["AE vs VAE 重构对比", "DCGAN 生成", "文本到图像 (SD 1.5)"]
    tab_funcs = [render_ae_vae_tab, render_dcgan_tab, render_diffusion_tab]

    tabs = st.tabs(tab_labels)

    if "current_tab" not in st.session_state:
        st.session_state.current_tab = 0

    for i, (tab, func) in enumerate(zip(tabs, tab_funcs)):
        with tab:
            if st.session_state.current_tab != i:
                clear_cuda_memory()
                st.session_state.current_tab = i
            func()

    # 侧边栏显存监控
    with st.sidebar:
        st.divider()
        st.subheader("💾 显存状态")
        try:
            import torch
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                total = torch.cuda.get_device_properties(0).total_memory / 1024**2
                st.metric("已分配", f"{alloc:.0f} MB")
                st.metric("已保留", f"{reserved:.0f} MB")
                st.progress(min(alloc / total, 1.0),
                            text=f"占用率 {alloc/total*100:.1f}%")
        except Exception:
            st.caption("显存信息不可用")


if __name__ == "__main__":
    main()
