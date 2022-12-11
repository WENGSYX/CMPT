# CMPT
A Multi-tasking and Multi-stage Chinese Minority Pre-Trained Language Model
###### 现有的少数民族语言预训练模型仍然较为稀缺，尽管国内少数民族语言模型CINO具有较强的理解能力，但仍然缺乏面向生成与翻译领域的研究。
###### **CMPT** (Chinese Minority Pre-Trained Language Model) 是在BART的基础上，加入DeepNorm预训练的超深层生成模型。其最大具有128+128层。其在超过10G的汉英维藏蒙语料中进行受限预训练。其具有较强的理解与生成性能。
<p align="center">
    <br>
    <img src="./image/main.png" width="500"/>
    <br>
</p>

### 检查点下载

| 模型简称 | 模型文件大小 | 模型层数 |百度网盘下载 | 
| :-------: | :---------: | :---------: | :---------: |
| **CMPT-Large** | **340MB** | **128+128** | **[PyTorch模型（密码1234）](https://pan.baidu.com/s/1YyMC7xHQF5KveGl3_0lylQ?pwd=1234)** |

### How to use

PyTorch版本包含3个文件：
```
pytorch_model.bin        # 模型权重
config.json              # 模型参数
sentencepiece.bpe.model  # 词表
special_tokens_map.json  # 特殊Token标记
tokenizer_config.json    # tokenizer参数
```

**CMPT**与BART较为相似，但加入了DeepNorm，因此请使用modeling_cmpt.py加载模型预定义层
```
from modeling_cmpt import CMPTForCir
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('./CMTP')
model = CMPTForCir.from_pretrained('./CMTP')
```

#### 预训练
`pretrain_train.py`
* 注意: 请安装deepspeed与apex方可开始预训练

## CITY
```
@InProceedings{10.1007/978-981-19-7960-6_10,
author="Li, Bin
and Weng, Yixuan
and Sun, Bin
and Li, Shutao",
editor="Xiao, Tong
and Pino, Juan",
title="A Multi-tasking and Multi-stage Chinese Minority Pre-trained Language Model",
booktitle="Machine Translation",
year="2022",
publisher="Springer Nature Singapore",
address="Singapore",
pages="93--105",
abstract="The existing multi-language generative model is considered as an important part of the multilingual field, which has received extensive attention in recent years. However, due to the scarcity of Chinese Minority corpus, developing a well-designed translation system is still a great challenge. To leverage the current corpus better, we design a pre-training method for the low resource domain, which can help the model better understand low resource text. The motivation is that the Chinese Minority languages have the characteristics of similarity and the adjacency of cultural transmission, and different multilingual translation pairs can provide the pre-trained model with sufficient semantic information. Therefore, we propose the Chinese Minority Pre-Trained (CMPT) language model with multi-tasking and multi-stage strategies to further leverage these low-resource corpora. Specifically, four pre-training tasks and two-stage strategies are adopted during pre-training for better results. Experiments show that our model outperforms the baseline method in Chinese Minority language translation. At the same time, we released the first generative pre-trained language model for the Chinese Minority to support the development of relevant research (All the experimental codes and the pre-trained language model are open-sourced on the website https://github.com/WENGSYX/CMPT).",
isbn="978-981-19-7960-6"
}
```
