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
* 运行prepare.sh脚本一键安装

### 进行实验
* resnet/bert/llama的具体实验操作分别见子目录。
