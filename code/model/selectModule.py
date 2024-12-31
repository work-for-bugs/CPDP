# 源项目选择模块
# 一、计算源项目与目标项目的KL散度
# 二、计算特征重要性
import pandas as pd
import numpy as np
from scipy.stats import entropy
from minepy import MINE


# 读取两个项目的数据
project1 = pd.read_csv("../baselines/content/ant-1.7.csv")
project2 = pd.read_csv("../baselines/content/lucene-2.0.csv")

# 一、计算源项目与目标项目的KL散度
def get_kl_divergence(project1, project2):
    # 统一列名 确保两个项目的指标一致
    assert list(project1.columns) == list(project2.columns)

    # 计算每个指标的KL散度
    kl_divergences = {}
    bins = 30  # 离散化的区间数
    epsilon = 1e-10  # 避免分布中有零概率


    for column in project1.columns:
        # 统计直方图分布
        p_hist, bin_edges = np.histogram(project1[column], bins=bins, density=True)
        q_hist, _ = np.histogram(project2[column], bins=bin_edges, density=True)

        # 添加平滑值避免零概率
        p_hist += epsilon
        q_hist += epsilon

        # 归一化为概率分布
        p_hist /= p_hist.sum()
        q_hist /= q_hist.sum()

        # 计算 KL 散度
        kl_div = entropy(p_hist, q_hist)
        kl_divergences[column] = kl_div


    # 打印每个指标的 KL 散度
    for metric, kl in kl_divergences.items():
        print(f"{metric}: KL Divergence = {kl}")

    # 总的 KL 散度（简单加权求和）
    total_kl_divergence = sum(kl_divergences.values())
    print(f"Total KL Divergence: {total_kl_divergence}")



# 二、计算特征重要性
def get_feature_importance(project1, project2):
    # 1.确定目标项目的每个指标的范围
    ranges = {}
    for column in project2.columns:
        min_val = project2[column].min()
        max_val = project2[column].max()
        ranges[column] = (min_val, max_val)

    # 2.计算源项目每个特征的重要性（MIC）
    feature_importance = {}
    target_variable = 'bug'
    mine = MINE() # 初始化最大信息系数计算器

    for column in project1.columns:
        if column == target_variable:  # 跳过目标变量
            continue
        mine.compute_score(project1[column], project1[target_variable])
        feature_importance[column] = mine.mic() # 记录MIC值

    # 3.计算实例的加权相似度
    instance_similarity = []

    for _, row in project1.iterrows():
        similarity = 0
        for column, (min_val, max_val) in ranges.items():
            if column == target_variable:  # 跳过目标变量
                continue
            if min_val <= row[column] <= max_val:  # 判断是否在目标项目的范围内
                similarity += feature_importance[column]
        instance_similarity.append(similarity)

    # 4.计算所有实例的平均相似度
    average_similarity = np.mean(instance_similarity)

    # 输出结果
    print("Feature Importance (MIC):", feature_importance)
    print("Instance Similarities:", instance_similarity)
    print("Average Weighted Similarity:", average_similarity)


get_kl_divergence(project1, project2)
get_feature_importance(project1, project2)