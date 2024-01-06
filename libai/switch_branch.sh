cd libai
git config user.email "1"
git config user.name "1"
git add .
git commit -m "experiment"

git fetch origin write_metric_thread
git checkout write_metric_thread
cd ..

cp bert_nl24_nah16_hs1024.py libai/configs/

sed -i 's/enabled=True/enabled=False/g' libai/configs/common/models/graph.py
sed -i 's|data_prefix="/workspace/data/libai_dataset/loss_compara_content_sentence|data_prefix="/home/OneFlowAutoTest/libai/bert_dataset/loss_compara_content_sentence|g' libai/configs/common/data/bert_dataset.py

# 添加统计memory的模块
sed -i '/import oneflow as flow/a import os' libai/libai/engine/trainer.py
sed -i '/for self.iter in range(start_iter, max_iter):/a\                    if self.iter == 99: \
                      cmd = "nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv" \
                      os.system(cmd)' libai/libai/engine/trainer.py

#sed -i '/import torch/a import os' Megatron-LM/megatron/training.py
#sed -i '/if iteration % args.log_interval == 0:/a\        if iteration == 100: \
#          cmd = "nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv" \
#          os.system(cmd)' Megatron-LM/megatron/training.py