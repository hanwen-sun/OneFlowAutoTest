## libai与megatron性能对比(Bert)
### 环境准备
* 检查libai, megatron, oneflow与pytorch是否都已安装
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
* 如果想切换branch执行libai，运行switch_branch.sh

### 运行megatron实验
* 修改megatron_args_pretrain_bert.sh 中的data_path与vocab_file
* 运行megatron_args_test.sh
* nsys开关在megatron_args_pretrain_bert.sh中，运行nsys需要直接在shell中运行，不要写成echo cmd形式;

### 提取实验数据
* `python extract_bert_test.py --test_log $test_log --compare_log $compare_log --oneflow_commit $commit`

### 脚本运行
* 运行prepare.sh
* 运行run_all.sh
* 得到最终对比数据在test_logs/libai下