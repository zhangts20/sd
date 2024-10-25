## 模型转换
经过 profile 后本项目暂时只对 UNet 部分采用 TensorRT 推理。该网络部分有三个输入：
1. 如果 do_classifier_free_guidance 为 True 即 guidance_scale 大于 1.0 时，输入会 cat 两个 latents，否则为一个 latents，这会直接影响该变量的**第一个维度**，为 1 或者 2。对于最后两个维度，在文生图任务中由模型配置决定，为 64x64；在图生图任务中，形状为输入图像的 8 倍下采样，所以**最后两个维度为动态的**
2. 第二个输入是 scale 没什么好说的
3. 第三个输入的第一维和第一个输入类型，第二个输入来自 tokenizer 编码 text 输入，长度会被 pad 到 77（固定的，当输入大于该值也会截取到该值），第三个维度的 768 是固定的，所以**第一个维度是动态的**就行
```shell
python tools/export_unet.py --model-dir /data/models/stable-diffusion-v1-4
```
按照上述配置导出后，支持输入的图像的宽高范围为 [256,1024]。

## Run
当输入指定了 --image-path 时，pipeline 为图生图，否则为文生图；同时，可以指定是否使用 TensorRT 推理 UNet 部分。
```shell
python tools/offline_inference.py --model-dir /data/models/stable-diffusion-v1-4 --image-path assets/dog.jpg <--use-trt>
```

## Q & A

Q: 报错 getPluginCreator could not find plugin: InstanceNormalization_TRT version: 1
[pluginV2Runner.cpp::load::308] Error Code 1: Serialization (Serialization assertion creator failed.Cannot deserialize plugin since corresponding IPluginCreator not found in Plugin Registry)

A: 在推理最前部分加上 `trt.init_libnvinfer_plugins(None, "")`，当时使用 C++ 推理与遇到了类似问题。

Q: 当导出的 ONNX 模型大于 2GB 时报错 `onnx.onnx_cpp2py_export.checker.ValidationError: The model does not have an ir_version set properly.`

A: 将 protobuf 的版本降至 3.20.3.

Q: 运行 TensorRT 的 img2img 时报错 `IExecutionContext::getTensorShape: Error Code 7: Internal Error (/up_blocks.1/Concat: axis 3 dimensions must be equal for concatenation on axis 1. Condition '==' violated: 16 != 15. Instruction: CHECK_EQUAL 16 15.)`

A: TODO
