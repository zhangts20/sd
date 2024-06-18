## Native Pytorch Inference and TensorRT Inference
NVIDIA A100-SXM4-80GB, CUDA 12.1, Pytorch 2.2.0+cu121, Python 3.10.12

**txt2img**

由于 UNet占整个 Pipeline 速度的 90% 左右，为了不修改 diffuers 代码，这里统计整个 Pipeline 的速度，即统计的是 m_infer 函数的耗时。
| 模型 | Pipeline 速度 (50 Unet) | 备注 |
| --- | --- | --- |
| pytorch | 3.8673 | 原始模型为 float32 类型的 |
| pytorch + tensorrt | 0.9724 | 将 UNet 部分转换为 TensorRT 的 float16 类型 |

**img2img**
| 模型 | Pipeline 速度 (40 Unet) | 备注 |
| --- | --- | --- |
| pytorch | 7.2833 | 原始模型为 float32 类型的 |
| pytorch + tensorrt | 1.6238 | 将 UNet 部分转换为 TensorRT 的 float16 类型 |

## 模型转换

~~UNet 部分有三个输入形状分别是 `[2,4,64,64]` scaler 和 `[2,77,768]`，输出形状是 `[2,4,64,64]`~~。UNet 有三个输入，第一个输入是动态可变的，暂时搞成除了第二维都可变。

导出模型精度为 float16，同时模型的输入类型为 float16，int64 和 float16。由于在导出 ONNX 模型时，输入类型和权重类型必须匹配（导出模型是会走 Pytorch 的前向推理），所以输入类型必须为 float16 的（尽管在实际推理中 UNet 部分的输入类型是 float32 的）。**否则会出现 Nan 的输出，导致黑图的出现，导出模型精度只有两个九，导致生成的图有些许抽象**

## Run
由于 txt2img 和 img2img 可以共用一个导出的 UNet 的 TensorRT 模型，所以导出一个动态的模型即可。导出模型：
```shell
python tools/export_trt.py --sd-dir /data/models/stable-diffusion-v1-4
```
使用 --force-onnx 或者 --force-engine 强制重新导出模型，否则使用已有模型。导出模型位于 unet 目录下的 onnx 和 trt 下。

**txt2img**
```shell
python tools/txt2img --sd-dir /data/models/stable-diffusion-v1-4
```
其他可选选项：`--prompt` 指定输入 prompt，`--negative-prompts` 指定不想在生成的图中出现的元素，`--use-trt` 是否使用 TensorRT 推理 UNet 部分，`--out-img-path` 指定输出图像路径，`--use-pipeline` 使用 diffuers 库推理并将结果存到 out/pipeline.jpg 中。

**img2img**
```shell
python tools/img2img --sd-dir /data/models/stable-diffusion-v1-4 --in-img-path ./images/sketch-mountains-input.jpg
```
其他可选选项同上。

## Architecture
### text_encoder
这部分的输入形状是 `(batch, model_max_length, hidden_size=768)`，输出形状不变。第二维从 `tokenizer/tokenizer_config.json` 获得，第三维从 `text_encoder/config.json` 获得。这部分负责将输入文本编码成特征向量。在编码文本输入时，如果指定了反向提示词（negative prompt），则还会编码这部分内容且和前面内容在第一维拼接，即 `(batch*2, model_max_length, 768)`。

### timesteps
UNet 的第二个输入，类似于大预言模型中的位置编码。

### prepare latents
UNet 的第三个输入。在文生图中，随机生成固定（可指定？）形状 `(batch, in_channels, sample_size, sample_size)`的张量，后两个变量从 `unet/config.json`获得；在图生图中，首先会编码输入图像生成 8 倍下采样的特征向量，然后生成随机噪声叠加到图像特征上。

### unet infer
UNet 部分有三个输入，是整个 Pipeline 的核心流程。通过指定的循环次数后，得到输出 latents。

### decode latents
这部分的功能是将输出 latents 解码成图像格式。

### vae
见 prepare_latents 的图生图部分。

## Q & A

Q: getPluginCreator could not find plugin: InstanceNormalization_TRT version: 1
[pluginV2Runner.cpp::load::308] Error Code 1: Serialization (Serialization assertion creator failed.Cannot deserialize plugin since corresponding IPluginCreator not found in Plugin Registry)

A: 在推理最前部分加上 `trt.init_libnvinfer_plugins(None, "")`，当时使用 C++ 推理与遇到了类似问题。
