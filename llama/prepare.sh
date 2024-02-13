# 准备数据
mkdir llama_dataset && cd llama_dataset

wget https://oneflow-dataset.oss-cn-beijing.aliyuncs.com/libai/loss_compara_content_sentence.bin &&
wget https://oneflow-dataset.oss-cn-beijing.aliyuncs.com/libai/loss_compara_content_sentence.idx

# lazy模式替换为eager模式;
#sed -i 's/enabled=True/enabled=False/g' /home/OneFlowAutoTest/libai/configs/common/models/graph.py
#cp ../Megatron-LM/pretrain_gpt.py ./pretrain_gpt.py
