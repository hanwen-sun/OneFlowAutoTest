## BERT
#export NCCL_IB_DISABLE=1
#export NCCL_DEBUG=INFO

#  1n1g         bert_nl24_nah16_hs1024_fp32_acfalse_mp1_pp1_mb1_gb1_1n1g
bash megatron_args_pretrain_bert.sh 1 1 0 127.0.0.1 1 1 false false 1 1
#  1n1g         bert_nl24_nah16_hs1024_fp32_acfalse_mp1_pp1_mb2_gb2_1n1g
bash megatron_args_pretrain_bert.sh 1 1 0 127.0.0.1 1 1 false false 2 2
#  1n1g         bert_nl24_nah16_hs1024_fp32_acfalse_mp1_pp1_mb4_gb4_1n1g
bash megatron_args_pretrain_bert.sh 1 1 0 127.0.0.1 1 1 false false 4 4
#  1n2g         bert_nl24_nah16_hs1024_fp32_acfalse_mp1_pp1_mb1_gb2_1n2g
bash megatron_args_pretrain_bert.sh 1 2 0 127.0.0.1 1 1 false false 1 2
#  1n2g         bert_nl24_nah16_hs1024_fp32_acfalse_mp1_pp1_mb2_gb4_1n2g
bash megatron_args_pretrain_bert.sh 1 2 0 127.0.0.1 1 1 false false 2 4
#  1n2g         bert_nl24_nah16_hs1024_fp32_acfalse_mp1_pp1_mb4_gb8_1n2g
bash megatron_args_pretrain_bert.sh 1 2 0 127.0.0.1 1 1 false false 4 8
