## oneflow自动化性能测试脚本
### 简介
为了方便的测试不同GPU下, 最新版oneflow eager模式与pytorch的训练性能对比，构建自动化测试脚本。
旧版本的测评脚本: https://github.com/Oneflow-Inc/OneAutoTest 可读性较差且不适用于现版本的oneflow与pytorch。

### 环境准备
* megatron最好拉取nvidia官方提供的docker进行实验: https://github.com/NVIDIA/Megatron-LM
  - 拉取docker:
  - 注: 最好使用nvidia官方最新的pytorch docker;
```shell
docker run --gpus all -it --shm-size 16G --ulimit memlock=-1 --name eager_test nvcr.io/nvidia/pytorch:24.01-py3
-v /data/hf_models/meta-llama/Llama-2-7b-hf/:/home/llama-model/
# 如果挂载数据需求，如针对llama, 需要添加 -v参数, 挂载/data/hf_models/meta-llama/Llama-2-7b-hf/tokenizer.model 至对应目录
# 更换清华源
python -m pip install --upgrade pip   # 可以不用执行
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
* oneflow可以采用源码编译或pip安装方式:
```shell
python3 -m pip install --pre oneflow -f https://oneflow-staging.oss-cn-beijing.aliyuncs.com/branch/master/cu121

python3 -m oneflow --doctor # 查看oneflow版本;
```
* 分别安装libai与megatron
* 可以运行prepare.sh脚本一键安装

### nvidia docker oneflow源码编译
* docker中没有cudnn, 将服务器中的cudnn挂载上:
```shell
docker run --gpus all -it --shm-size 16G --ulimit memlock=-1 -v /usr/local/cudnn:/usr/local/cudnn -v /data/hf_models/meta-llama/Llama-2-7b-hf/:/home/llama-model/ --name eager_test nvcr.io/nvidia/pytorch:23.10-py3
```
* 修改oneflow的Cmakelist.txt: set(CMAKE_CXX_STANDARD 14)
* 手动编译安装 OpenBLAS:
```shell
git clone https://github.com/xianyi/OpenBLAS.gi
make -j32
make install
```
* 编译oneflow:
```shell
mkdir build
cd build
cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.1/ -DBLA_VENDOR=OpenBLAS  -DCUDNN_ROOT_DIR=/usr/local/cudnn/ -DBUILD_PROFILER=ON  -C ../cmake/caches/cn/cuda.cmake ..
make -j32
```


### 进行实验
* resnet/bert/llama的具体实验操作分别见子目录。
