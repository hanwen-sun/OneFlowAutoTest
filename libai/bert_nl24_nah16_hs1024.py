from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from .common.models.bert import pretrain_model as model
from .common.models.graph import graph
from .common.train import train
from .common.optim import optim
from .common.data.bert_dataset import dataloader, tokenization
import sys
import os

#os.chdir(sys.path[0])
vocab_file = "/home/sunhanwen/OneFlowAutoTest/libai/bert_dataset/bert-base-chinese-vocab.txt"
data_prefix = "/home/sunhanwen/OneFlowAutoTest/libai/bert_dataset/loss_compara_content_sentence"

tokenization.tokenizer.vocab_file = vocab_file
dataloader.train.dataset[0].data_prefix = data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix
# dataloader.train.num_workers = 4

# Bert-large model config
#model.cfg.hidden_layers = 24
model.cfg.num_attention_heads = 16
model.cfg.hidden_size = 1024

#train.dist.pipeline_num_layers = model.cfg.hidden_layers
train.test_micro_batch_size = 4

train.evaluation.evaluator = LazyCall(PPLEvaluator)()

train.input_placement_device = "cpu"


train.evaluation.enabled = False
train.evaluation.eval_iter = 30

