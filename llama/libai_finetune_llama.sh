#ninja -C ~/oneflow/build  -j32 

#source ~/oneflow/build/source.sh
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
log_filename="libai_llama_finetinue"

echo "save file: $log_filename.log"

# source /home/fuxu/work/oneflow/build/source.sh

#rm -r /data0/home/fuxu/libai/new5/sft_result
rm -rf ./sft_result
rm -rf results/libai/$log_filename.log

#export ONEFLOW_DEBUG=0

bash ../libai/tools/train.sh ../libai/tools/train_net.py ../libai/projects/Llama/configs/test_llama_config.py 8 > results/libai/$log_filename.log 2>&1
# 使用nsys存储相关的运行信息
# nsys profile --trace=cuda --output=/data0/home/fuxu/job_$my_pid.qdrep bash tools/train.sh projects/Llama/train_net.py projects/Llama/configs/llama_sft.py 8 >> job_$my_pid.log 2>&1
#rm -rf $log_filename.nsys-rep
#nsys profile \
#    --trace=nvtx --delay=62 --duration=6 --output=$log_filename.qdrep \
#    bash tools/train.sh tools/train_net.py projects/Llama/configs/test_llama_config.py 8 >> $log_filename.log 2>&1
# nsys profile \
#     --trace=cuda,nvtx --delay=70 --duration=45 --output=/data0/home/fuxu/job_$my_pid.qdrep bash tools/train.sh tools/train_net.py projects/Llama/configs/test_llama_config.py 8 >> job_$my_pid.log 2>&1


# bash tools/train.sh tools/train_net.py projects/Llama/configs/test_llama_config.py 8 >> job_$my_pid.log 2>&1
