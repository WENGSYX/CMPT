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
