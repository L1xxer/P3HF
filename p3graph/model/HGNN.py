import torch
import torch.nn as nn
import dhg.nn as dnn


class TemporalHGNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 位置编码层
        self.position_encoder = nn.Linear(in_dim, in_dim)

        # 超图卷积层
        self.hgnn_conv = dnn.HGNNConv(in_dim, out_dim)

        # 归一化和激活函数
        self.norm = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, lengths=None):
        # 如果提供了lengths，添加位置编码
        if lengths is not None:
            x = self._add_position_encoding(x, lengths)

        # 直接将位置编码后的特征输入超图卷积
        output = self.hgnn_conv(x, H)
        output = self.norm(output)
        output = self.relu(output)

        return output

    def _add_position_encoding(self, x, lengths):
        """添加位置编码到节点特征"""
        position_enhanced = x.clone()
        start_idx = 0

        for i, length in enumerate(lengths):
            length = int(length)
            # 分别处理音频和视觉节点
            for modal_offset in [0, length]:  # 0表示音频节点起始位置，length表示视觉节点起始位置
                end_idx = start_idx + modal_offset + length
                modal_range = range(start_idx + modal_offset, end_idx)

                # 为当前模态的节点生成位置编码
                for pos, node_idx in enumerate(modal_range):
                    # 归一化位置 (0 到 1 之间)
                    normalized_pos = pos / max(1, length - 1)
                    # 位置编码: sin函数可以捕获周期性的时序关系
                    pos_encoding = torch.sin(torch.full_like(x[node_idx], normalized_pos) * torch.pi)
                    # 将位置信息融入节点特征
                    position_enhanced[node_idx] = x[node_idx] + self.position_encoder(pos_encoding)

            # 更新起始索引，移动到下一个样本
            start_idx += 2 * length

        return position_enhanced