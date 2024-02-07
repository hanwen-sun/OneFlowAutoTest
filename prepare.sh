# 该脚本在进入docker环境后运行:

# 安装对应版本python, pytorch与pytorch
#apt-get -y update
#apt-get -y install python3.10 # ubuntu22.04默认python3.10
#ln -s /usr/bin/python3.10 /usr/bin/python # 创建软连接，方便直接使用python代替python3.10执行操作（软连接删除：rm /usr/bin/python）
#apt-get -y install python3-pip # 安装pip
#ln -s /usr/bin/pip3 /usr/bin/pip # 创建软连接，方便直接使用pip代替

# 更换清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装oneflow
# 这里安装cuda12.1版本，与pytorch保持一致;
python3 -m pip install --pre oneflow -f https://oneflow-staging.oss-cn-beijing.aliyuncs.com/branch/master/cu121

python3 -m oneflow --doctor # 查看oneflow版本;

# 安装libai与megatron
git clone https://github.com/Oneflow-Inc/libai.git
cd libai

pip install -e .
cd ..

git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install -e .