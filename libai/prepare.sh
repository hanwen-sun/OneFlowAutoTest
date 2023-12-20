git clone https://github.com/Oneflow-Inc/libai.git
cd libai

# 确认已安装oneflow版本
pip install pybind11
pip install -e .

git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install .

cd ..

mkdir bert_dataset && cd bert_dataset
# apt-get -y install wget
wget https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/bert-base-chinese-vocab.txt  &&
wget https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.bin && 
wget https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.idx