## libai与megatron性能对比(llama2-7B)
### 环境准备
* 检查libai, megatron, oneflow与pytorch是否都已安装
### 修改libai代码
* libai默认为graph模式，需要手动开启eager模式
  * 在libai/configs/common/models/graph.py 中修改enabled=False;
* 如果需要，参照switch_branch.sh切换libai分支

### 数据准备
```shell
mkdir llama_dataset && cd llama_dataset
# apt-get -y install wget
wget https://oneflow-dataset.oss-cn-beijing.aliyuncs.com/libai/loss_compara_content_sentence.bin &&
wget https://oneflow-dataset.oss-cn-beijing.aliyuncs.com/libai/loss_compara_content_sentence.idx
```

### 运行libai实验
* 下载alpace dataset: git clone https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM.git
* 将test_llama_config.py移动至 libai/projects/llama/configs下
* 修改 libai/projects/Llama 文件夹中导入模型和数据集的位置:
  * 在adapter_config.py和llama_config.py中修改两个pretrained_model_path;
  * 在adapter_sft.py与llama_sft.py中修改 dataset_path与pretrained_model_path;
  * 在llama_sft.py中修改output_dir(checkpoint位置)
* 运行libai_fintune_llama.sh 

### 运行megatron实验
* 修改megatron_finetune_llama.sh 中的TOKENIZER_MODEL,CHECKPOINT_PATH与DATA_PATH
* 运行megatron_finetune_llama.sh
* nsys开关在megatron_finetune_llama.sh中

### 脚本运行
* 运行prepare.sh, 一键进行环境准备
* 运行megatron_finetune_llama.sh 
* 运行libai_finetune_llama.sh
