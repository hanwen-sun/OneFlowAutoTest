git clone https://github.com/Oneflow-Inc/libai.git
cd libai

# 确认已安装oneflow版本
pip install pybind11
pip install -e .

cd ..

git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install .

cd ..

mkdir bert_dataset && cd bert_dataset
# apt-get -y install wget
wget https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/bert-base-chinese-vocab.txt  &&
wget https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.bin && 
wget https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.idx

# 修改libai对应代码
cp bert_nl24_nah16_hs1024.py libai/configs/

sed -i 's/enabled=True/enable=False/g' libai/configs/common/models/graph.py
sed -i 's|data_prefix="/workspace/data/libai_dataset/loss_compara_content_sentence|data_prefix="/home/OneFlowAutoTest/libai/bert_datasetloss_compara_content_sentence/|g' libai/configs/common/data/bert_dataset.py

# 添加统计memory的模块
sed -i '/import oneflow as flow/a import os' libai/libai/engine/trainer.py
sed -i '/for self.iter in range(start_iter, max_iter):/a\                    if self.iter == 99: \
                      cmd = "nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv" \
                      os.system(cmd)' libai/libai/engine/trainer.py

sed -i '/import torch/a import os' Megatron-LM/megatron/training.py
sed -i '/if iteration % args.log_interval == 0:/a\        if iteration == 100: \
          cmd = "nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv" \
          os.system(cmd)' Megatron-LM/megatron/training.py
