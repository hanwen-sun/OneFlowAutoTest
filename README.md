## oneflow自动化性能测试脚本
### 简介
为了方便的测试不同GPU下, 最新版oneflow eager模式与pytorch的训练性能对比，构建自动化测试脚本。
旧版本的测评脚本: https://github.com/Oneflow-Inc/OneAutoTest 可读性较差且不适用于现版本的oneflow与pytorch。

### 环境准备
* megatron最好拉取nvidia官方提供的docker进行实验: https://github.com/NVIDIA/Megatron-LM
  - 拉取docker:
```shell
docker run --gpus all -it --shm-size 16G --ulimit memlock=-1 --name eager_test nvcr.io/nvidia/pytorch:23.10-py3
# 更换清华源
python -m pip install --upgrade pip   # 可以不用执行
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
* oneflow可以采用源码编译或pip安装方式:
```shell
python3 -m pip install --pre oneflow -f https://oneflow-staging.oss-cn-beijing.aliyuncs.com/branch/master/cu118

python3 -m oneflow --doctor # 查看oneflow版本;
```
* 运行prepare.sh脚本一键安装

### 进行实验
* 先进行libai实验, 再进行resnet50实验(参考对应readme)
