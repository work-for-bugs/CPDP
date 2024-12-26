from gensim.models import Word2Vec
import os
from transformers import RobertTokenizer, RobertaModel
import torch

# 加载 CodeBERT 的分词器和模型
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")

def get_codebert_embedding(code_snippet):
    """
    使用 CodeBERT 提取代码片段的特征嵌入。
    """
    # 对代码进行分词，并转换为张量
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=512)
    
    # 获取模型输出
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 使用 [CLS] token 的输出作为代码嵌入
    embeddings = outputs.last_hidden_state[:, 0, :]  # shape: [batch_size, hidden_size]
    return embeddings.squeeze(0).numpy()  # 转为 NumPy 数组


def preprocess_cpg_with_codebert(graph, tokenizer, model):
    """
    使用 CodeBERT 提取节点特征并更新图。
    """
    for node, attrs in graph.nodes(data=True):
        code = attrs.get("code", "")
        if code:
            embedding = get_codebert_embedding(code)
            attrs["features"] = embedding
    return graph
