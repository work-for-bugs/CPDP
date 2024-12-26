import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch

dot_path = "../dataset/PROMISE/apache-ant-1.6.0-joern-parse-dot/AntClassLoader.java/3-cpg.dot"


# edge_index = torch.tensor([[u, v] for u, v in G.edges()], dtype=torch.long).t().contiguous()

# 加载cpg文件
def load_cpg(dot_file):
    return nx.nx_agraph.read_dot(dot_file) # 返回 networkx 图对象

# 预处理cpg数据
def preprocess_cpg(G):
    # 提取节点信息
    node_features = []
    node_ids = []
    for node, attrs in G.nodes(data=True):
        # 提取节点类型并进行one-hot编码
        node_features.append(attrs.get('type', 'UNKNOWN'))
        
        node_ids.append(node)
    

print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")


# # 假设每个节点有类型属性，转换为特征向量
# node_features = []
# for node, attrs in cpg.nodes(data=True):
#     node_type = attrs.get('type', 'UNKNOWN')
#     node_features.append(one_hot_encode(node_type))  # 自定义 one-hot 编码
# x = torch.tensor(node_features, dtype=torch.float)
# # 构建 PyTorch Geometric 的数据对象
# data = Data(x=x, edge_index=edge_index)
# # 打印数据对象
# print(data)

for node, attributes in G.nodes(data=True):
    print(f"Node: {node}, Attributes: {attributes}")

for u, v, attributes in G.edges(data=True):
    print(f"Edge: ({u}, {v}), Attributes: {attributes}")

nx.draw(G)
plt.show()