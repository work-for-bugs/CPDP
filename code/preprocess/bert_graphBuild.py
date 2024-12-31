import re
from transformers import BertTokenizer, BertModel
import torch
from torch_geometric.data import Data


# 1.加载预训练的Bert模型和分词器
tokenizer = BertTokenizer.from_pretrained("../bert-base-uncased")
model = BertModel.from_pretrained("../bert-base-uncased")


# 2.解析DOT文件
def read_graph_from_file(file_path):
    nodes = {}
    edges = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            # 解析节点信息
            node_match = re.match(r'"(\d+)" \[label\s*=\s*<(.*?)> \]', line)
            if node_match:
                node_id, label = node_match.groups()
                nodes[node_id] = label
                continue

            # 解析边信息
            edge_match = re.match(r'"(\d+)" -> "(\d+)"  \[ label\s*=\s*"(.*?)"\]', line)
            if edge_match:
                src_id, dst_id, edge_label = edge_match.groups()
                edges.append((src_id, dst_id, edge_label))
                continue
    
    return nodes, edges


# 3. 使用BERT对标签进行编码
def encode_with_bert(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # 平均池化
    return embeddings


# 4. 构建图数据结构
def build_graph_data(nodes, edges):
    # 使用BERT对节点和边标签进行嵌入
    node_ids = list(nodes.keys())
    node_labels = list(nodes.values())
    node_embeddings = encode_with_bert(node_labels)
    
    edge_index = []
    edge_embeddings = []
    
    for src_id, dst_id, edge_label in edges:
        edge_index.append((node_ids.index(src_id), node_ids.index(dst_id)))
        edge_embeddings.append(encode_with_bert([edge_label]).squeeze(0))  # 单个标签嵌入
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_embeddings) if edge_embeddings else torch.tensor([])
    
    return Data(x=node_embeddings, edge_index=edge_index, edge_attr=edge_attr)


# 输出图数据内容
def print_graph_data(graph_data):
    print("Graph Data Details:")
    print(f"Number of Nodes: {graph_data.num_nodes}")
    print(f"Number of Edges: {graph_data.num_edges}")
    print(f"Node Features (x):\n{graph_data.x}")
    print(f"Edge Index (edge_index):\n{graph_data.edge_index}")
    if graph_data.edge_attr is not None:
        print(f"Edge Features (edge_attr):\n{graph_data.edge_attr}")
    else:
        print("Edge Features (edge_attr): None")


# 5. 主函数
def main():
    # 文件路径
    file_path = '../../dataset/PROMISE/apache-ant-1.6.0-joern-parse-dot/AbstractAccessTask.java/0-cpg.dot'
    
    # 解析DOT文件
    nodes, edges = read_graph_from_file(file_path)

    # 检查解析结果
    if not nodes or not edges:
        print("警告：文件中未找到有效的节点或边。")
    else:
        print(f"成功解析到 {len(nodes)} 个节点和 {len(edges)} 条边。")
    
    # 构建图数据
    graph_data = build_graph_data(nodes, edges)
    
    print("Graph Data:")
    print(graph_data)
    # graph_data 是构建好的 Data 对象
    torch.save(graph_data, "graph_data.pt")
    # print_graph_data(graph_data)

    # 现在你可以将'data'传递给GCN层或者其他图神经网络组件进行训练或推理
    print("Graph loaded and encoded successfully.")


# 运行主函数
if __name__ == "__main__":
    main()
























































































