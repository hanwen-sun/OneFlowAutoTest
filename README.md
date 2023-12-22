## oneflow自动化性能测试脚本
### 简介
为了方便的测试不同GPU下, 最新版oneflow eager模式与pytorch的训练性能对比，构建自动化测试脚本。
旧版本的测评脚本: https://github.com/Oneflow-Inc/OneAutoTest 可读性较差且不适用于现版本的oneflow与pytorch。

### 环境准备

* 由于megatron在cuda 11.8下没跑通，这里我们测试resnet50时将pytorch与oneflow cuda对齐，在测试bert时, libai使用11.8, pytorch使用12.2; 

```shell
docker run --gpus all -it --shm-size 16G --ulimit memlock=-1 --name eager_test nvcr.io/nvidia/pytorch:23.10-py3
# 更换清华源
python -m pip install --upgrade pip   # 可以不用执行
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

* oneflow安装(nightly)
```shell
python3 -m pip install --pre oneflow -f https://oneflow-staging.oss-cn-beijing.aliyuncs.com/branch/master/cu118

python3 -m oneflow --doctor # 查看oneflow版本;
```
* 运行prepare.sh脚本一键安装

### 进行实验
* 先进行libai实验, 再进行resnet50实验(参考对应readme)
