# bash tools/train.sh tools/train_net.py projects/Llama/configs/test_llama_config.py 8

import os
from omegaconf import DictConfig, OmegaConf

from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from libai.scheduler import WarmupExponentialLR

from configs.common.train import train
from configs.common.models.graph import graph
from configs.common.optim import optim

from projects.Llama.configs.llama_config import cfg
from projects.Llama.tokenizer import LlamaTokenizer
from projects.Llama.llama import LlamaForCausalLM

from configs.common.data.gpt_dataset import dataloader

# Hyperparameters
weight_decay = 0.1
learning_rate = 5e-5
pretrained_model_path = "/home/llama-model/"

data_prefix = "/home/OneFlowAutoTest/llama/llama_dataset/loss_compara_content_sentence"

# graph & optim
# graph["enabled"] = True
graph["enabled"] = False

optim.update(
    dict(
        lr=learning_rate,
        weight_decay=weight_decay,
    )
)

# tokenize
tokenization = OmegaConf.create()
tokenization.make_vocab_size_divisible_by = 1
tokenization.tokenizer = LazyCall(LlamaTokenizer)(
    pretrained_model_path=os.path.join(pretrained_model_path, "tokenizer.model")
)

# model
model = LazyCall(LlamaForCausalLM)(cfg=cfg)

# datasets
dataloader.train.dataset[0].data_prefix = data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix
dataloader.test[0].dataset.data_prefix = data_prefix
dataloader.test[0].dataset.indexed_dataset.data_prefix = data_prefix
for ds in dataloader.train.dataset:
    ds.max_seq_length = 512

train.update(
    dict(
        output_dir="/home/OneFlowAutoTest/llama/sft_result",
        train_micro_batch_size=1,
        test_micro_batch_size=1,
        train_epoch=0,
        train_iter=100,
        log_period=10,
        warmup_ratio=2 / 5,
        num_accumulation_steps=1,
        rdma_enabled=False,
        amp=dict(enabled=True),
        activation_checkpoint=dict(enabled=True),
        checkpointer=dict(
            period=100000,
            max_to_keep=1,
        ),
        dist=dict(
            data_parallel_size=1,
            tensor_parallel_size=8,
            pipeline_parallel_size=1,
            pipeline_num_layers=cfg.hidden_layers,
        ),
        evaluation=dict(
            enabled=True,
            evaluator=LazyCall(PPLEvaluator)(),
            eval_period=1000,
            eval_iter=1,
        ),
        scheduler=LazyCall(WarmupExponentialLR)(
            warmup_factor=0.0,
            gamma=1.0,
            warmup_method="linear",
        ),
    )
)

