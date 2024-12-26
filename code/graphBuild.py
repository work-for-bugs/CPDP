import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch

dot_path = "../dataset/PROMISE/apache-ant-1.6.0-joern-parse-dot/AntClassLoader.java/3-cpg.dot"


# edge_index = torch.tensor([[u, v] for u, v in G.edges()], dtype=torch.long).t().contiguous()

# 加载cpg文件
G = nx.nx_agraph.read_dot(dot_path) # 返回 networkx 图对象


    

print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")



for node, attributes in G.nodes(data=True):
    print(f"Node: {node}, Attributes: {attributes}")

for u, v, attributes in G.edges(data=True):
    print(f"Edge: ({u}, {v}), Attributes: {attributes}")

nx.draw(G)
# plt.show()