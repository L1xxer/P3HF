import torch
import torch.nn as nn
import torch.nn.functional as F

from .Attention import MultiHeadAttention
from .EventDiscriminator import EventDiscriminator
from .SeqContext import SeqContext
from .Classifier import Classifier
from .functions import batch_graphify, split_a_v_node_outputs
from .HSICLoss import HSICLoss
from .HGNN import TemporalHGNN as HGNN
import p3graph

log = p3graph.utils.get_logger()


class P3G(nn.Module):

    def __init__(self, args):
        super(P3G, self).__init__()
        a_dim = 1024
        v_dim = 2048
        p_dim = 768

        """-----搜参修改-----"""
        if args.once:
            g_dim = 640
            h2_dim = 384    # 图的输出维度
            h3_dim = 512   # 子空间输出维度

        else:
            g_dim = args.g_dim
            h2_dim = args.h2_dim    # 图的输出维度
            h3_dim = args.h3_dim    # 子空间输出维度
        """-----------------"""

        self.weight_hsic = 0.1
        self.weight_adv = 0.05

        hc_dim = 100
        tag_size = 3
        num_head = 4
        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device

        self.rnn_a1 = SeqContext(a_dim, g_dim, args)
        self.rnn_a2 = SeqContext(a_dim, g_dim, args)
        self.rnn_a3 = SeqContext(a_dim, g_dim, args)
        self.rnn_v1 = SeqContext(v_dim, g_dim, args)
        self.rnn_v2 = SeqContext(v_dim, g_dim, args)
        self.rnn_v3 = SeqContext(v_dim, g_dim, args)
        self.rnn_p = SeqContext(p_dim, g_dim, args)

        self.gated = nn.Linear(g_dim, g_dim)
        self.hgnn = HGNN(g_dim, h2_dim).to(self.device)

        self.attention = MultiHeadAttention(
            d_model=h2_dim,
            num_heads=num_head,
            dropout=args.drop_rate
        ).to(self.device)

        self.pub_encoder = nn.Linear(2 * h2_dim, h3_dim)

        self.pri_encoder1 = nn.Linear(2 * h2_dim, h3_dim)
        self.pri_encoder2 = nn.Linear(2 * h2_dim, h3_dim)
        self.pri_encoder3 = nn.Linear(2 * h2_dim, h3_dim)

        self.pub_disc = EventDiscriminator(h3_dim)
        self.pri_cls = HSICLoss(sigma=1.0)

        self.clf = Classifier(4 * h3_dim, hc_dim, tag_size, args)

        self.edge_type_to_idx = {"0": 0, "1": 1}

    def get_rep(self, data):
        node_features_a1 = self.rnn_a1(data["each_len_tensor"], data["audio1_tensor"])                      # [batch_size, mx_len, g_dim]
        node_features_a2 = self.rnn_a2(data["each_len_tensor"], data["audio2_tensor"])
        node_features_a3 = self.rnn_a3(data["each_len_tensor"], data["audio3_tensor"])
        node_features_v1 = self.rnn_v1(data["each_len_tensor"], data["visual1_tensor"])
        node_features_v2 = self.rnn_v2(data["each_len_tensor"], data["visual2_tensor"])
        node_features_v3 = self.rnn_v3(data["each_len_tensor"], data["visual3_tensor"])
        node_feature_personality = self.rnn_p(data["each_len_tensor"], data["personality_tensor"])

        gate = torch.sigmoid(self.gated(node_feature_personality))
        node_features_a1 = node_features_a1 * gate + node_features_a1
        node_features_a2 = node_features_a2 * gate + node_features_a2
        node_features_a3 = node_features_a3 * gate + node_features_a3
        node_features_v1 = node_features_v1 * gate + node_features_v1
        node_features_v2 = node_features_v2 * gate + node_features_v2
        node_features_v3 = node_features_v3 * gate + node_features_v3


        H1, features1 = batch_graphify(node_features_a1, node_features_v1, data["each_len_tensor"], wp=self.wp, wf=self.wf)
        H2, features2 = batch_graphify(node_features_a2, node_features_v2, data["each_len_tensor"], wp=self.wp, wf=self.wf)
        H3, features3 = batch_graphify(node_features_a3, node_features_v3, data["each_len_tensor"], wp=self.wp, wf=self.wf)
        H1 = H1.to(self.device)
        H2 = H2.to(self.device)
        H3 = H3.to(self.device)
        features1 = features1.to(self.device)
        features2 = features2.to(self.device)
        features3 = features3.to(self.device)

        graph_out1 = self.hgnn(features1, H1)
        graph_out2 = self.hgnn(features2, H2)
        graph_out3 = self.hgnn(features3, H3)

        # 分离音频和视觉节点
        a_nodes1, v_nodes1 = split_a_v_node_outputs(graph_out1, data["each_len_tensor"])
        a_nodes2, v_nodes2 = split_a_v_node_outputs(graph_out2, data["each_len_tensor"])
        a_nodes3, v_nodes3 = split_a_v_node_outputs(graph_out3, data["each_len_tensor"])

        # 对每个样本分别应用自注意力
        model_fusion_list1 = []
        model_fusion_list2 = []
        model_fusion_list3 = []

        for i in range(len(a_nodes1)):
            # 对音频节点应用自注意力
            a_attn1, _ = self.attention(a_nodes1[i], a_nodes1[i], a_nodes1[i])
            a_attn2, _ = self.attention(a_nodes2[i], a_nodes2[i], a_nodes2[i])
            a_attn3, _ = self.attention(a_nodes3[i], a_nodes3[i], a_nodes3[i])

            # 对视觉节点应用自注意力
            v_attn1, _ = self.attention(v_nodes1[i], v_nodes1[i], v_nodes1[i])
            v_attn2, _ = self.attention(v_nodes2[i], v_nodes2[i], v_nodes2[i])
            v_attn3, _ = self.attention(v_nodes3[i], v_nodes3[i], v_nodes3[i])

            # 拼接增强后的特征
            fused1 = torch.cat([a_attn1, v_attn1], dim=-1)  # [l, 2 * feat_dim]
            fused2 = torch.cat([a_attn2, v_attn2], dim=-1)
            fused3 = torch.cat([a_attn3, v_attn3], dim=-1)

            model_fusion_list1.append(fused1)
            model_fusion_list2.append(fused2)
            model_fusion_list3.append(fused3)

        # 拼接所有样本的结果
        model_fusion1 = torch.cat(model_fusion_list1, dim=0)  # [sum_len, 2 * feat_dim]
        model_fusion2 = torch.cat(model_fusion_list2, dim=0)
        model_fusion3 = torch.cat(model_fusion_list3, dim=0)


        pub_feature1 = self.pub_encoder(model_fusion1)  # [sum_len, h3_dim]
        pub_feature2 = self.pub_encoder(model_fusion2)
        pub_feature3 = self.pub_encoder(model_fusion3)
        pri_feature1 = self.pri_encoder1(model_fusion1)
        pri_feature2 = self.pri_encoder2(model_fusion2)
        pri_feature3 = self.pri_encoder3(model_fusion3)

        pub_feature = torch.mean(torch.stack([pub_feature1, pub_feature2, pub_feature3], dim=0), dim=0)  # [sum_len, h3_dim]
        event_fusion = torch.cat([pub_feature, pri_feature1, pri_feature2, pri_feature3], dim=-1)  # [sum_len, 4*h3_dim]

        return event_fusion, pub_feature1, pub_feature2, pub_feature3, pri_feature1, pri_feature2, pri_feature3

    def prepare_disc_inputs(self, pub_feature1, pub_feature2, pub_feature3):
        sum_len = pub_feature1.size(0)
        labels = torch.arange(3, dtype=torch.long, device=self.device).repeat_interleave(sum_len)
        features = torch.cat([pub_feature1, pub_feature2, pub_feature3], dim=0)

        indices = torch.randperm(labels.size(0), device=self.device)
        return features[indices], labels[indices]

    def forward(self, data):
        out, pub_feature1, pub_feature2, pub_feature3, pri_feature1, pri_feature2, pri_feature3= self.get_rep(data)
        out = self.clf(out, data["each_len_tensor"])

        return out 

    def get_loss(self, data, mode='full'):
        out, pub_feature1, pub_feature2, pub_feature3, pri_feature1, pri_feature2, pri_feature3 = self.get_rep(data)

        if mode == 'disc_only':
            with torch.no_grad():
                all_pub_features, all_event_labels = self.prepare_disc_inputs(pub_feature1, pub_feature2, pub_feature3)
            disc_loss = self.pub_disc.get_loss(all_pub_features.detach(), all_event_labels)
            return None, disc_loss

        cls_loss = self.clf.get_loss(out, data["label_tensor"], data["each_len_tensor"])

        all_pub_features, all_event_labels = self.prepare_disc_inputs(pub_feature1, pub_feature2, pub_feature3)
        disc_loss = self.pub_disc.get_loss(all_pub_features.detach(), all_event_labels)
        adv_loss = -self.pub_disc.get_loss(all_pub_features, all_event_labels)
        # adv_loss = torch.clamp(adv_loss, min=-2, max=0)

        hsic_loss = (
                self.pri_cls(pri_feature1, pri_feature2) +
                self.pri_cls(pri_feature1, pri_feature3) +
                self.pri_cls(pri_feature2, pri_feature3)
        )

        total_loss = (1-self.weight_hsic-self.weight_adv) * cls_loss + self.weight_hsic * hsic_loss + self.weight_adv * adv_loss
        return total_loss, disc_loss
