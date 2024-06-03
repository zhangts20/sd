## Native Pytorch Inference and TensorRT Inference

| 模型 | 速度 (50-UNet, s) | 备注 |
| --- | --- | --- |
| pytorch | 3.7933 | 原始模型为 float32 类型的 |
| pytorch + tensorrt | 0.9615 | 将 UNet 部分转换为 TensorRT 的 float16 类型 |

## 模型转换


UNet 部分有三个输入形状分别是 `[2,4,64,64]` scaler 和 `[2,77,768]`，输出形状是 `[2,4,64,64]`。

导出一个 float16 和 bfloat16 类型的模型，同时模型的输入类型为 float16，int64 和 float16。关于导出模型的输入类型的说明：由于在导出 ONNX 模型时，输入类型和
权重类型必须匹配（导出模型是会走 Pytorch 的前向推理），所以输入类型必须为 float16 的（尽管在实际推理中 UNet 部分的输入类型是 float32 的）。**否则会出现 Nan 的输出，导致黑图的出现**

## Architecture
#### text_encoder
```txt
CLIPTextModel(
  (text_model): CLIPTextTransformer(
    (embeddings): CLIPTextEmbeddings(
      (token_embedding): Embedding(49408, 768)
      (position_embedding): Embedding(77, 768)
    )
    (encoder): CLIPEncoder(
      (layers): ModuleList(
        (0-11): 12 x CLIPEncoderLayer(
          (self_attn): CLIPAttention(
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (layer_norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): CLIPMLP(
            (activation_fn): QuickGELUActivation()
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
          )
          (layer_norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
)
```

## Q & A

Q: getPluginCreator could not find plugin: InstanceNormalization_TRT version: 1
[pluginV2Runner.cpp::load::308] Error Code 1: Serialization (Serialization assertion creator failed.Cannot deserialize plugin since corresponding IPluginCreator not found in Plugin Registry)

A: 在推理最前部分加上 `trt.init_libnvinfer_plugins(None, "")`，当时使用 C++ 推理与遇到了类似问题。
