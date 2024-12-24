import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tempModel import GCN # 模型




def create_graph_data():
    # 节点特征 (假设有 5 个节点，每个节点有 3 个特征)
    x = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float)

    # 图边 (用边的索引表示)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)  # 0->1, 1->2, 2->3, 3->4

    # 数据集包装
    data = Data(x=x, edge_index=edge_index)
    return data


def train():
     # 创建图数据
    data = create_graph_data()

    # 初始化模型
    input_dim = data.x.size(1)
    hidden_dim = 16
    output_dim = 4
    model = GCN(input_dim, hidden_dim, output_dim)

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()  # 假设是分类任务

    # 模拟标签 (假设每个节点都有一个类别)
    labels = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long)

     # 训练循环
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)  # 前向传播
        loss = loss_fn(out, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
    

# 执行训练
train()
    




