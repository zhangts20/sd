## Native Pytorch Inference
```txt
Function load_model took 2.0607 seconds to execute
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:04<00:00, 12.24it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:03<00:00, 13.19it/s]
Function p_infer took 3.9064 seconds to execute
Function load_model took 1.2591 seconds to execute
Function encode_prompt took 0.0163 seconds to execute
Function prepare_latents took 0.0001 seconds to execute
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 51/51 [00:03<00:00, 13.46it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 51/51 [00:03<00:00, 13.45it/s]
Function unet_infer took 3.7933 seconds to execute
Function decode_latents took 0.0523 seconds to execute
Function m_infer took 3.868 seconds to execute
```

## TensorRT Inference


初步来看，unet 部分的迭代推理占据了绝大部分时间，先使用 TensorRT 跑一下这个部分。
这部分的输入有三个，暂时看到的固定形状是 `[2,4,64,64]` scaler 和 `[2,77,768]`，输出形状是 `[2,4,64,64]`，先导出一个固定形状的模型。

Note: 导出的时候发现 UNet 部分的偏执是 float32 类型的。

## Q & A

Q: getPluginCreator could not find plugin: InstanceNormalization_TRT version: 1
[pluginV2Runner.cpp::load::308] Error Code 1: Serialization (Serialization assertion creator failed.Cannot deserialize plugin since corresponding IPluginCreator not found in Plugin Registry)

A: 在推理最前部分加上 `trt.init_libnvinfer_plugins(None, "")`，当时使用 C++ 推理与遇到了类似问题。

Q: 最后出的是黑图，看了 ONNX 和 TensorRT 的精度，大概是两个九的相似度。

A: 对了下 PyTorch 模型的精度发现 ONNX 和其对得差不多，TensorRT 有差别。