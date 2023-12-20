## oneflow自动化性能测试脚本
### 简介
为了方便的测试不同GPU下, 最新版oneflow eager模式与pytorch的训练性能对比，构建自动化测试脚本。
旧版本的测评脚本: https://github.com/Oneflow-Inc/OneAutoTest 可读性较差且不适用于现版本的oneflow与pytorch。

### 环境准备


* 拉取nvidia的cuda docker, 安装对应依赖,在docker中进行性能对比, 注意指定cuda版本为11.8, 因为oneflow目前还不支持cuda 12, 这里需要与oneflow对齐。(暂时存在问题，改为拉取nvidia-pytorch docker)
```shell
docker run --gpus all -it --shm-size 16G --ulimit memlock=-1 --name eager_test nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
```
* clone 自动化脚本
```shell
apt-get -y update
apt-get -y install git
git clone https://github.com/hanwen-sun/OneFlowAutoTest.git
```
* 以下操作可以运行prepare.sh一键运行;
* 安装对应版本python与pytorch:
```shell
apt-get -y install python3.10 # ubuntu22.04默认python3.10
ln -s /usr/bin/python3.10 /usr/bin/python # 创建软连接，方便直接使用python代替python3.10执行操作（软连接删除：rm /usr/bin/python）
apt-get -y install python3-pip # 安装pip
ln -s /usr/bin/pip3 /usr/bin/pip # 创建软连接，方便直接使用pip代替pip3


# 更换清华源
python -m pip install --upgrade pip   # 可以不用执行
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 大约需要20-30 min
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

* 由于megatron在cuda 11.8下没跑通，这里我们测试resnet50时将pytorch与oneflow cuda对齐，在测试bert时, libai使用11.8, pytorch使用12.2; 命令如下:
```shell
docker run --gpus all -it --shm-size 16G --ulimit memlock=-1 --name eager_test nvcr.io/nvidia/pytorch:23.10-py3
```

* oneflow安装(nightly)
```shell
python3 -m pip install --pre oneflow -f https://oneflow-staging.oss-cn-beijing.aliyuncs.com/branch/master/cu118

python3 -m oneflow --doctor # 查看oneflow版本;
```

### 进行实验

* 先进行libai实验, 再进行resnet50实验(参考对应readme)
