## BERT
#  1n1g   
# remove the libai result here;
#export ONEFLOW_VM_COMPUTE_ON_WORKER_THREAD=0
rm -rf test_logs/libai/
./args_libai_bert.sh ../libai/configs/bert_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 false false 1 1

#./args_libai_bert.sh ../libai/configs/bert_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 false false 2 2

#./args_libai_bert.sh libai/configs/bert_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 false false 4 4

# 1n2g
#./args_libai_bert.sh ../libai/configs/bert_nl24_nah16_hs1024.py 1 2 0 127.0.0.1 1 1 false false 1 2

#./args_libai_bert.sh ../libai/configs/bert_nl24_nah16_hs1024.py 1 2 0 127.0.0.1 1 1 false false 2 4

#./args_libai_bert.sh libai/configs/bert_nl24_nah16_hs1024.py 1 2 0 127.0.0.1 1 1 false false 4 8
