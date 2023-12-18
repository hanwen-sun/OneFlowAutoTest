## oneflow自动化性能测试脚本
### 简介
为了方便的测试不同GPU下, 最新版oneflow eager模式与pytorch的训练性能对比，构建自动化测试脚本。
旧版本的测评脚本: https://github.com/Oneflow-Inc/OneAutoTest 可读性较差且不适用于现版本的oneflow与pytorch。

### 环境准备
* 拉取最新版nvidia的pytorch docker, 在docker中进行性能对比, 注意指定cuda版本为11.8, 因为oneflow目前还不支持cuda 12, 这里需要与oneflow对齐。
```shell
docker pull nvcr.io/nvidia/pytorch:23.10-py3-cuda11.8
```

### 进行实验
* 运行docker: 
```shell
docker run --gpus all -it --shm-size 16G --ulimit memlock=-1 --name eager_test nvcr.io/nvidia/pytorch:23.10-py3-cuda11.8
```
* 先进行resnet50实验, 再进行libai实验(参考readme)