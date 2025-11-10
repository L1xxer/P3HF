import numpy as np
import torch

import p3graph

log = p3graph.utils.get_logger()

import torch
import dhg


def batch_graphify(node_features_a, node_features_v, each_len_tensor, wp, wf):
    batch_size, mx_len, g_dim = node_features_a.shape

    edge_list = []
    node_features = []

    global_node_id = 0

    for b in range(batch_size):
        length = int(each_len_tensor[b])
        a_base = global_node_id
        v_base = global_node_id + length

        # 收集该样本下所有节点特征
        for t in range(length):
            node_features.append(node_features_a[b, t].tolist())  # audio
        for t in range(length):
            node_features.append(node_features_v[b, t].tolist())  # visual

        # 构建同模态超边（audio）
        for t in range(length):
            group = []
            for offset in range(-wp, wf + 1):
                i = t + offset
                if 0 <= i < length:
                    group.append(a_base + i)
            if len(group) > 1:
                edge_list.append(group)

        # 构建同模态超边（visual）
        for t in range(length):
            group = []
            for offset in range(-wp, wf + 1):
                i = t + offset
                if 0 <= i < length:
                    group.append(v_base + i)
            if len(group) > 1:
                edge_list.append(group)

        # 构建跨模态超边：a[t] → v[t-wp : t+wf]
        for t in range(length):
            group = [a_base + t]
            for offset in range(-wp, wf + 1):
                i = t + offset
                if 0 <= i < length:
                    group.append(v_base + i)
            if len(group) > 1:
                edge_list.append(group)

        # 构建跨模态超边：v[t] → a[t-wp : t+wf]
        for t in range(length):
            group = [v_base + t]
            for offset in range(-wp, wf + 1):
                i = t + offset
                if 0 <= i < length:
                    group.append(a_base + i)
            if len(group) > 1:
                edge_list.append(group)

        # 更新 global_node_id（注意每个模态都用了 length 个节点）
        global_node_id += 2 * length

    # 建图
    H = dhg.Hypergraph(num_v=global_node_id, e_list=edge_list)

    # 节点特征拼接成矩阵
    node_feat_mat = torch.tensor(node_features)

    return H, node_feat_mat


def edge_perms(length, window_past, window_future):
    """
    Method to construct the edges of a graph (a utterance) considering the past and future window.
    return: list of tuples. tuple -> (vertice(int), neighbor(int))
    """

    all_perms = set()
    array = np.arange(length)
    for j in range(length):
        perms = set()

        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:  # use all past context
            eff_array = array[:min(length, j + window_future + 1)]
        elif window_future == -1:  # use all future context
            eff_array = array[max(0, j - window_past):]
        else:
            eff_array = array[max(0, j - window_past):min(length, j + window_future + 1)]

        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)


def split_a_v_node_outputs(node_outputs, each_len_tensor):
    a_nodes_list = []
    v_nodes_list = []
    start = 0
    for l in each_len_tensor:
        l = int(l)
        a_nodes = node_outputs[start: start + l]
        v_nodes = node_outputs[start + l: start + 2 * l]

        a_nodes_list.append(a_nodes)
        v_nodes_list.append(v_nodes)

        start += 2 * l

    # 返回分离后的音频节点和视觉节点列表
    return a_nodes_list, v_nodes_list