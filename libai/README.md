## libai与megatron性能对比
### libai安装
```shell
git clone https://github.com/Oneflow-Inc/libai.git
cd libai

# 确认已安装oneflow版本
pip install pybind11
pip install -e .
```
### 修改libai代码
* libai默认为graph模式，需要手动开启eager模式
  * 在libai/configs/common/models/graph.py 中修改enabled=False;

### megatron安装
```shell
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install .
```

### 数据准备
```shell
mkdir bert_dataset && cd bert_dataset
# apt-get -y install wget
wget https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/bert-base-chinese-vocab.txt  &&
wget https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.bin && 
wget https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.idx
```

### 运行libai实验
* 修改args_libai_bert.sh中的train_net.py的路径;
* 修改bert_nl24_nah16_hs1024.py中data路径:
* nsys开关在args_libai_bert.sh中;
* `cp bert_nl24_nah16_hs1024.py libai/configs/`
* 修改 libai/configs/common/data/bert_dataset.py中data_prefix路径。
* 运行run_libai.sh

### 运行megatron实验
* 修改megatron_args_pretrain_bert.sh 中的data_path与vocab_file
* 运行megatron_args_test.sh

### 提取实验数据
* `python extract_bert_test.py --test_log $test_log --compare_log $compare_log --oneflow_commit $commit`

### 脚本运行
* 运行prepare.sh
* 运行run_all.sh
* 得到最终数据  在test_logs/libai下