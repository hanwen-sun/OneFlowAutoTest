#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

#需要修改的路径位置
TOKENIZER_MODEL=/home/llama-model/tokenizer.model   # https://huggingface.co/meta-llama/Llama-2-7b-hf
CHECKPOINT_PATH=/home/Megatron-LM/ckpt
DATA_PATH=/home/OneFlowAutoTest/llama/llama_dataset/loss_compara_content_sentence

# 7 B   
HIDDEN_SIZE=4096 # e.g. llama-13b: 5120
FFN_HIDDEN_SIZE=11008 # e.g. llama-13b: 13824
NUM_LAYERS=32 # e.g. llama-13b: 40
NUM_HEADS=32 # e.g. llama-13b: 40
SEQ_LENGTH=4096
NUM_KV_HEADS=32 # llama2 70B uses GQA

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NUM_HEADS \
    --seq-length 512 \
    --max-position-embeddings 4096 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --load ${CHECKPOINT_PATH} \
    --no-load-optim \
    --no-load-rng \
    --fp16 \
    --untie-embeddings-and-output-weights \
    --use-rotary-position-embeddings \
    --normalization RMSNorm \
    --no-position-embedding \
    --no-masked-softmax-fusion \
"

#train-iters 迭代次数,设置的是 500
TRAINING_ARGS="
    --recompute-activations \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --lr 0.00015 \
    --train-iters 500 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --no-check-for-nan-in-loss-and-grad \
    --vocab-size 32000 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

#输出文件名
logfile="megatron_finetune_llama.log"
echo "save file: $logfile"

rm -rf $logfile
torchrun $DISTRIBUTED_ARGS ../Megatron-LM/pretrain_gpt.py \
    $GPT_ARGS \
    $TRAINING_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH >> $logfile 2>&1

#使用nsys存储相关的GPU kernel运行信息
# nsys profile --trace=cuda,nvtx --force-overwrite true --output=job_$my_pid.qdrep --stats=true torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
#     $GPT_ARGS \
#     $TRAINING_ARGS \
#     $DATA_ARGS \
#     $OUTPUT_ARGS \
#     --distributed-backend nccl \
#     --save $CHECKPOINT_PATH \
#     --load $CHECKPOINT_PATH >> job_$my_pid.log 2>&1