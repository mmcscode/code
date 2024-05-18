from dgl.nn.pytorch import HeteroGraphConv
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair


class GATConv(nn.Module):
    def __init__(
            self,
            in_feats,  # 输入的节点特征维度
            out_feats,  # 输出的节点特征维度
            edge_feats,  # 输入边的特征维度
            num_heads=1,  # 注意力头数
            feat_drop=0.0,  # 节点特征dropout
            attn_drop=0.0,  # 注意力dropout
            edge_drop=0.0,  # 边特征dropout
            negative_slope=0.2,
            activation=None,
            allow_zero_in_degree=False,
            use_symmetric_norm=False,
    ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._use_symmetric_norm = use_symmetric_norm
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)

        self.fc_edge = nn.Linear(edge_feats, out_feats * num_heads, bias=False)

        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_edge = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.reset_parameters()
        self._activation = activation

    # 初始化参数
    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_edge, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            # feat[0]源节点的特征
            # feat[1]目标节点的特征
            # h_edge 边的特征
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            h_edge = self.feat_drop(graph.edata['feature'])

            if not hasattr(self, "fc_src"):
                self.fc_src, self.fc_dst = self.fc, self.fc

            # 特征赋值
            feat_src, feat_dst, feat_edge = h_src, h_dst, h_edge
            # 转换成多头注意力的形状
            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            feat_edge = self.fc_edge(h_edge).view(-1, self._num_heads, self._out_feats)

            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            # 简单来说就是拼接矩阵相乘和拆开分别矩阵相乘再相加的效果是一样的
            # 但是前者更加高效

            # 左节点的注意力权重
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            # 右节点的注意力权重
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.dstdata.update({"er": er})
            # 左节点权重+右节点权重 = 节点计算出的注意力权重（e）
            graph.apply_edges(fn.u_add_v("el", "er", "e"))

            # 边计算出来的注意力权重
            ee = (feat_edge * self.attn_edge).sum(dim=-1).unsqueeze(-1)
            # 边注意力权重加上节点注意力权重得到最终的注意力权重
            # 这里可能应该也是和那个拼接操作等价吧
            graph.edata.update({"e": graph.edata["e"] + ee})
            # 经过激活函数，一起激活和分别激活可能也是等价吧
            e = self.leaky_relu(graph.edata["e"])

            # 注意力权重的正则化
            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=graph.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                a = torch.zeros_like(e)
                a[eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
                graph.edata.update({"a": a})
            else:
                graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # 消息传递
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            # 标准化
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -1)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

        return rst


class RGATLayer(nn.Module):
    def __init__(
            self,
            in_feats,  # 输入的特征维度 （边和节点一样）
            hid_feats,  # 隐藏层维度
            out_feats,  # 输出的维度
            num_heads,  # 注意力头数
            rel_names,  # 关系的名称（用于异构图卷积）
    ):
        super().__init__()
        self.conv1 = HeteroGraphConv(
            {rel: GATConv(in_feats, hid_feats // num_heads, in_feats, num_heads) for rel in rel_names}, aggregate='sum')
        self.conv2 = HeteroGraphConv({rel: GATConv(hid_feats, out_feats, in_feats, num_heads) for rel in rel_names},
                                     aggregate='sum')
        self.hid_feats = hid_feats

    def forward(self, graph, inputs):
        # graph 输入的异构图
        # inputs 输入节点的特征
        h = self.conv1(graph, inputs)  # 第一层异构卷积
        h = {k: F.relu(v).view(-1, self.hid_feats) for k, v in h.items()}  # 经过激活函数，将注意力头数拉平
        h = self.conv2(graph, h)  # 第二层异构卷积
        return h


class RGAT(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads, rel_names):
        super().__init__()
        self.rgat = RGATLayer(in_features, hidden_features, out_features, num_heads, rel_names)

    def forward(self, g, x):
        h = self.rgat(g, x)
        # 输出的就是每个节点经过R-GAT后的特征
        for k, v in h.items():
            print(k, v.shape)
        return h
