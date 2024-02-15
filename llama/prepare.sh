# 准备数据
mkdir llama_dataset && cd llama_dataset

wget https://oneflow-dataset.oss-cn-beijing.aliyuncs.com/libai/loss_compara_content_sentence.bin &&
wget https://oneflow-dataset.oss-cn-beijing.aliyuncs.com/libai/loss_compara_content_sentence.idx

cp ../Megatron-LM/pretrain_gpt.py ./pretrain_gpt.py
cp test_llama_config.py ../libai/projects/Llama/configs/

git clone https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM.git

#lazy模式替换为eager模式; 
sed -i 's/enabled=True/enabled=False/g' /home/OneFlowAutoTest/libai/configs/common/models/graph.py
# 替换对应文件路径
# configs文件
sed -i 's|pretrained_model_path="meta-llama/Llama-2-7b-hf/"|pretrained_model_path="/home/llama-model/"|g' ../libai/projects/Llama/adapter/adapter_config.py
sed -i 's|pretrained_model_path="meta-llama/Llama-2-7b-hf/"|pretrained_model_path="/home/llama-model/"|g' ../libai/projects/Llama/configs/llama_config.py

sed -i 's|pretrained_model_path="Llama-2-7b-hf/tokenizer.model"|pretrained_model_path="/home/llama-model/tokenizer.model"|g' ../libai/projects/Llama/adapter/adapter_config.py
sed -i 's|pretrained_model_path="Llama-2-7b-hf/tokenizer.model"|pretrained_model_path="/home/llama-model/tokenizer.model"|g' ../libai/projects/Llama/configs/llama_config.py

# sft文件
sed -i 's|dataset_path = "alpaca_data"|dataset_path = "GPT-4-LLM/plots/data/alpaca_data.json"|g' ../libai/projects/Llama/adapter/adapter_sft.py
sed -i 's|pretrained_model_path = "meta-llama/Llama-2-7b-hf/"|pretrained_model_path = "/home/llama-model/"|g' ../libai/projects/Llama/adapter/adapter_sft.py

sed -i 's|dataset_path = "alpaca_data"|dataset_path = "GPT-4-LLM/plots/data/alpaca_data.json"|g' ../libai/projects/Llama/configs/llama_sft.py
sed -i 's|pretrained_model_path = "meta-llama/Llama-2-7b-hf/"|pretrained_model_path = "/home/llama-model/"|g' ../libai/projects/Llama/configs/llama_sft.py

# 在llama_sft.py中修改checkpoint位置;
