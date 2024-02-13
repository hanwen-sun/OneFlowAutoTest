## libai与megatron性能对比(llama2-7B)
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
* 如果需要，参照switch_branch.sh切换libai分支
### megatron安装
```shell
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install .
```

### 数据准备
```shell
mkdir llama_dataset && cd llama_dataset
# apt-get -y install wget
wget https://oneflow-dataset.oss-cn-beijing.aliyuncs.com/libai/loss_compara_content_sentence.bin &&
wget https://oneflow-dataset.oss-cn-beijing.aliyuncs.com/libai/loss_compara_content_sentence.idx
```

### 运行libai实验
* 修改args_libai_bert.sh中的train_net.py的路径;
* 修改bert_nl24_nah16_hs1024.py中data路径:
* nsys开关在args_libai_bert.sh中;
* `cp bert_nl24_nah16_hs1024.py libai/configs/`
* 修改 libai/configs/common/data/bert_dataset.py中data_prefix路径。
* 运行run_libai.sh
* 如果想切换branch执行libai，运行switch_branch.sh

### 运行megatron实验
* 修改megatron_finetune_llama.sh 中的TOKENIZER_MODEL, CHECKPOINT_PATH与DATA_PATH
* 运行megatron_finetune_llama.sh
* nsys开关在megatron_finetune_llama.sh中

### 提取实验数据
* `python extract_bert_test.py --test_log $test_log --compare_log $compare_log --oneflow_commit $commit`

### 脚本运行
* 运行prepare.sh
* 运行run_all.sh
* 得到最终对比数据在test_logs/libai下