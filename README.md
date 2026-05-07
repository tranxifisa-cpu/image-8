# 图像生成模型交互实验室

`image-processor-8` 是一个 Streamlit Web 应用，用于演示图像重构、生成式模型和文本到图像生成。

## 功能特点

- AE 与 VAE 重构对比：支持 MNIST 和 Fashion-MNIST，展示重构图像、loss 曲线和潜空间分布。
- DCGAN 图像生成：支持训练生成器和判别器，查看生成样本网格和判别器分数。
- Stable Diffusion 文本到图像：可设置提示词、负向提示词、采样步数、CFG 强度、随机种子和分辨率。
- 结果对比：支持网格、并排和分割式对比。
- 显存监控：侧边栏展示 CUDA 显存状态。

## 运行说明

```bash
cd image-processor-8
pip install -r requirements.txt
streamlit run app.py
```

打开 Streamlit 输出的本地地址即可使用，通常是：

```text
http://localhost:8501
```

## 部署说明

部署到 Streamlit Community Cloud 时，主文件路径填写：

```text
image-processor-8/app.py
```

## 注意事项

Stable Diffusion 首次加载需要下载模型权重，对网络、磁盘和显存要求较高。CPU 模式可以运行部分功能，但速度会明显降低。
