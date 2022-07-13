import json
import os
import transformers
transformers.logging.set_verbosity_error()
import pandas as pd
import numpy as np
import torch
from accelerate import Accelerator
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random
import os
import time
from sklearn.model_selection import *
from torch.autograd import Variable
import jieba
from transformers import BertTokenizer,XLMRobertaTokenizer,AdamW,get_cosine_schedule_with_warmup
import argparse
#import deepspeed
#deepspeed.init_distributed()
accelerator = Accelerator()
device = accelerator.device
transformers.logging.set_verbosity_error()
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0)
parser.add_argument("--deepspeed_config")

#torch.autograd.set_detect_anomaly(True)
class NoiseFunction(object):
  """
  Base class for segment noise generation
  """
  def __init__(self, config):
    self.config=config
  def apply_noise(self, segment):
    """
    Apply noise to a segment and return source and target versions.
    Target can be equal to the original segment, but not necessarily, depending on the noise function
    """
    raise NotImplementedError("apply_noise() not implemented")
    return "", ""
  def extract_special_prefix_tokens(self, sentence_tokens):
    """
    Utility function to extract special prefix tokens
    """
    return [], sentence_tokens
  def assemble_target(self, prefix_tokens, tokens):
    return ' '.join(tokens)


args = parser.parse_args()
CFG = { #训练的参数配置
    'fold_num': 5, #五折交叉验证
    'seed': 7234231,
    'model': 'CMPT', #预训练模型
    'max_len': 80, #文本截断的最大长度
    'epochs': 10,
    'train_bs': 36, #batch_size，可根据自己的显存调整
    'valid_bs': 36,
    'lr': 8e-5, #学习率
    'num_workers': 16,
    'accum_iter': 1, #梯度累积，相当于将batch_size*2
    'weight_decay': 1e-4, #权重衰减，防止过拟合
    'device': 0,
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['seed']) #固定随机种子

torch.cuda.set_device(CFG['device'])

#训练样本示例
data = [
    {
        'sent1':'你好世界',
        'sent2':'Hello World'
    }
]
tokenizer = XLMRobertaTokenizer.from_pretrained('hfl/cino-small-v2')



class MyDataset(Dataset):
    def __init__(self,data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        d = data[idx]

        return d



def collate_fn(data):
    input_ids, attention_mask, token_type_ids,label_ids,decoder_attention_mask = [], [], [],[],[]
    for x in data:
        source = tokenizer(x['sent1'],return_tensors='pt', padding='max_length', truncation=True, max_length=CFG['max_len'])
        input_ids.append(source['input_ids'][0])
        attention_mask.append(source['attention_mask'][0])
        label = tokenizer(x['sent2'],return_tensors='pt', padding='max_length', truncation=True, max_length=CFG['max_len'])
        l = torch.tensor(label['input_ids'][0])
        l = torch.where(l==1,-100,l)
        label_ids.append(l)

    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    label = torch.stack(label_ids)
    return input_ids, attention_mask,label
class AverageMeter:  # 为了tqdm实时显示loss和acc
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def compute_loss(y_pred, t=0.05, device="cpu"):
    idxs = torch.arange(0, y_pred.shape[0], device=device)
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarities = similarities - torch.eye(y_pred.shape[0], device=device) * 1e12
    similarities = similarities / t
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)
def compute_contrastive_loss(score_matrix, margin=0.5):
    '''
       margin: predefined margin to push similarity score away
       score_matrix: bsz x seqlen x seqlen; cosine similarity matrix
       input_ids: bsz x seqlen
    '''
    bsz, seqlen, _ = score_matrix.size()
    gold_score = torch.diagonal(score_matrix, offset=0, dim1=1, dim2=2) # bsz x seqlen
    gold_score = torch.unsqueeze(gold_score, -1)
    assert gold_score.size() == torch.Size([bsz, seqlen, 1])
    difference_matrix = gold_score - score_matrix
    assert difference_matrix.size() == torch.Size([bsz, seqlen, seqlen])
    loss_matrix = margin - difference_matrix # bsz x seqlen x seqlen
    loss_matrix = torch.nn.functional.relu(loss_matrix)
    cl_loss = torch.mean(loss_matrix)
    return cl_loss
def train_model(model, train_loader,epoch):  # 训练一个epoch
    model.train()

    losses_LM = AverageMeter()
    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True,disable=str(device)!='cuda:0',ncols=180)
    for step, (source_input_ids, source_attention_mask,  target_label_ids) in enumerate(tk):

        output = model(input_ids=source_input_ids,attention_mask=source_attention_mask,labels=target_label_ids)
        loss = output.loss
        model.backward(loss)
        model.step()

        losses_LM.update(loss.item(), target_label_ids.size(0))
        tk.set_postfix(lossLM=losses_LM.avg)
        if step % 3000 == 0 and str(device)=='cuda:0':
            model.module.save_pretrained('zhen/{}'.format(str(step)))
            tokenizer.save_pretrained('zhen/{}'.format(str(step)))

    return losses_LM.avg


train_set = MyDataset(data)
from BartModel_fintune import BartForConditionalGeneration
from collections import OrderedDict

model = BartForConditionalGeneration.from_pretrained(CFG['model'])
train_loader = DataLoader(train_set, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=True,
                              num_workers=16)
optimizer = AdamW(model.parameters(),lr=CFG['lr'], weight_decay=CFG['weight_decay'])
scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // CFG['accum_iter'],
                                            CFG['epochs'] * len(train_loader) // CFG['accum_iter'])
#from transformers.deepspeed import HfDeepSpeedConfig
#dschf = HfDeepSpeedConfig('ds_config.json')
model,_,_,_ = deepspeed.initialize(model=model, config_params='ds_config.json',optimizer=optimizer,lr_scheduler=scheduler)
train_loader = accelerator.prepare(train_loader)



for epoch in range(CFG['epochs']):
    train_loss = train_model(model, train_loader,epoch)
