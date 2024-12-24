from graphviz import Digraph # 用于图可视化
import networkx as nx
import ast


def get_nodes_by_key(nodes, key):
    for node in nodes:
        if node['key'].strip() == key:
            return node
    return None

def visual_graph(all_nodes, index_map, edges, allowed_edge_types, ver_edge_type, file_name = 'graph'):
    graph = Digraph()
    nodes = set()
    for edge in edges:
        start, t, end = edge
        nodes.add(start)
        nodes.add(end)
        type_ = ver_edge_type[str(t)] # 将边类型编号映射为类型名
        if type_ in allowed_edge_types.keys(): # 仅处理允许的边类型
            color = allowed_edge_types[type_] # 获取边的颜色
            s = index_map[start]  # 获取起点的可视化标识符
            e = index_map[end] # 获取终点的可视化标识符
            graph.edge(s, e, color=color, label=type_) # 添加边到图
    for node in nodes:
        true_id = index_map[node] # 获取节点的可视化标识符
        #node_content = all_nodes[true_id]
        node_content = get_nodes_by_key(all_nodes, true_id) # 获取节点详细信息
        graph.node(name = true_id, label = true_id + '\n' + node_content['code'].strip() + '\n' + str(node_content['type'].strip()))
    graph.render(file_name, view = False)

def build_ast(starts, edges, ast_edges):
    if len(starts) == 0:
        return 
    new_starts = []
    for i in starts:
        ast = {}
        ast['start'] = i
        ast['end'] = []
        for edge in edges:
            if edge['start'].strip() == i and edge['type'].strip() == 'IS_AST_PARENT':
                ast['end'].append(edge['end'].strip())
                new_starts.append(edge['end'].strip())
        if len(ast['end']) > 0:
            ast_edges.append(ast)
    build_ast(new_starts, edges, ast_edges)
    pass
def graphGeneration(nodes, edges, edge_type_map, ver_edge_type_map):
    index_map = dict()
    index_map_ver = dict()
    all_nodes = set()
    all_ast_edges = []
    all_edges = list()
    for node in nodes:
        if node['isCFGNode'].strip() != 'True' or node['key'].strip() == 'File':
            continue
        all_nodes.add(node['key'])  
        # entry  exit
        if node['type'] in ['CFGEntryNode','CFGExitNode']:
            continue
        nodeKey = [node['key']]
        ast_edges = []
        build_ast(nodeKey, edges, ast_edges)
   
        if len(ast_edges) == 0:
            #break; continue; returns; (;;) 
            if spe_sent(node['type'], node['code'].strip().split()):
                dic = {}
                dic['start'] = nodeKey[0]
                dic['end'] = nodeKey
                ast_edges.append(dic)
            else:
                return None, None, None, True
        all_ast_edges.append(ast_edges)  
    # edge_count_before = 0
    # node_count_before = set()
    # for item in all_ast_edges:
    #     for ast in item:
    #         node_count_before.add(ast['start'])
    #         node_count_before = node_count_before | set(ast['end'])
    #         edge_count_before += len(ast['end'])
            
    # edge_count_after = 0
    # node_count_after = set()
    # for item in all_ast_edges:
    #     for ast in item:
    #         node_count_after.add(ast['start'])
    #         node_count_after = node_count_after | set(ast['end'])
    #         edge_count_after += len(ast['end'])
            
    nsc_edges = get_ncs_edges(all_ast_edges, 'NSC', nodes)
    ast_type = 'IS_AST_PARENT'
    for item in all_ast_edges:
        # break; continue; return; 
        if len(item) == 1 and len(item[0]['end']) == 1 and item[0]['start'] == item[0]['end'][0]:
            continue
        for x in item:
            start = x['start']
            for end in x['end']:
                all_edges.append([start, ast_type, end])  
    for e in edges:
        start, end, eType = e['start'], e['end'], e['type']    
        start_node = get_nodes_by_key(nodes, start)
        end_node = get_nodes_by_key(nodes, end)
        if start_node['isCFGNode'].strip() != 'True' or end_node['isCFGNode'].strip() != 'True':
            continue
        if eType != 'IS_FILE_OF' and eType != ast_type:
            if not eType in edge_type_map: #or not start in all_nodes or not end in all_nodes:
                continue
            all_edges.append([start, eType, end])  
    for e in all_edges:
        start, _, end = e
        all_nodes.add(start)
        all_nodes.add(end)
    if len(all_nodes) == 0 or len(all_nodes) > 500:  
        return None, None, None, None
    for i, node in enumerate(all_nodes):
        index_map[node] = i
        index_map_ver[i] = node
    all_edges_new = []   #original full graph
    for e in all_edges: # e = [start, type, end]
        e_new = [index_map[e[0]], edge_type_map[e[1]], index_map[e[2]]]
        all_edges_new.append(e_new)
    #add nsc_edge
    for e in nsc_edges:
        e_new = [index_map[e[0]], edge_type_map[e[1]], index_map[e[2]]]
        all_edges_new.append(e_new)
    #add self-loop
    loop = 'Self_loop'
    for node in all_nodes:
        self_loop = [index_map[node], edge_type_map[loop], index_map[node]]
        all_edges_new.append(self_loop)
    if len(all_edges_new) == 0:
        return None, None, None, None
    
    edges_num = {}
    for t in edge_type_map.keys():
        edges_num[t] = 0
    for e in all_edges_new:
        key = ver_edge_type_map[str(e[1])]
        edges_num[key] += 1
    edges_num['nodes'] = len(all_nodes)
    return index_map_ver, all_edges_new, len(index_map_ver), edges_num


