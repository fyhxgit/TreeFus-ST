# model.py
# TreeFus-ST: Tree-Structured Spectral-Temporal Fusion for spoofed speech detection.
# This file contains the RawGAT-ST backbone and the proposed TreeFus-ST module.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
import random
from collections import OrderedDict

# ------------------------------
# GraphAttentionLayer (your original)
# ------------------------------
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        # apply activation
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


# ------------------------------
# TreeFus-ST module: tree-structured spectral-temporal fusion
# ------------------------------
class TreeFusSTLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 线性映射：自身、父节点、子节点特征
        self.lin_self = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_parent = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_children = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        """
        x: [batch, N, in_dim]  （输入为融合后或拼接后的节点特征）
        返回: [batch, N, out_dim]
        """
        B, N, D = x.size()
        device = x.device
        out = torch.zeros(B, N, self.lin_self.out_features, device=device)
        for b in range(B):
            h = x[b]  # 当前样本的节点特征 [N, D]
            # 1. 动态根节点选择（节点特征L2范数最大的作为根）
            scores = torch.norm(h, dim=1)  # [N]
            _, idx_sort = torch.sort(scores, descending=True)
            # 2. 构建二叉树父子映射
            parents = [-1] * N
            child_count = [0] * N
            root = idx_sort[0].item()
            parents[root] = -1
            # 按得分排序依次为节点分配父节点，每个父节点最多2个子节点
            for i in range(1, N):
                node = idx_sort[i].item()
                for j in range(i):
                    p = idx_sort[j].item()
                    if child_count[p] < 2:
                        parents[node] = p
                        child_count[p] += 1
                        break
            # 收集每个节点的子节点列表
            children = [[] for _ in range(N)]
            for node in range(N):
                p = parents[node]
                if p != -1:
                    children[p].append(node)
            # 3. 特征聚合：自身 + 父节点 + 子节点（对多个子节点取平均）
            for node in range(N):
                h_self = self.lin_self(h[node])
                h_parent = torch.zeros_like(h_self)
                h_child = torch.zeros_like(h_self)
                # 父节点贡献
                p = parents[node]
                if p != -1:
                    h_parent = self.lin_parent(h[p])
                # 子节点贡献（平均）
                c_list = children[node]
                if len(c_list) > 0:
                    child_feats = torch.stack([h[c] for c in c_list], dim=0)
                    mean_child = child_feats.mean(dim=0)
                    h_child = self.lin_children(mean_child)
                out[b, node] = h_self + h_parent + h_child
        # （可选）加入激活函数或归一化
        return out


# ------------------------------
# Pool (top-k pooling) 你原始实现
# ------------------------------
class Pool(nn.Module):
    def __init__(self, k: float, in_dim: int, p=0.0):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)
        return new_h

    def top_k_graph(self, scores, h, k):
        num_nodes = h.shape[1]
        batch_size = h.shape[0]

        # first reflect the weights and then rank them
        H = h * scores
        _, idx = torch.topk(scores, max(2, int(k * num_nodes)), dim=1)
        new_g = []

        for i in range(batch_size):
            new_g.append(H[i, idx[i][:int(len(idx[i]))], :])

        new_g = torch.stack(new_g, dim=0)

        return new_g


# ------------------------------
# Sinc-based CONV (原始)
# ------------------------------
class CONV(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, device, out_channels, kernel_size, in_channels=1, sample_rate=16000,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, mask=False):
        super(CONV, self).__init__()
        if in_channels != 1:
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.device = device
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')
        self.device = device

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)

    def forward(self, x, mask=False):
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(2 * fmax * self.hsupp / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * np.sinc(2 * fmin * self.hsupp / self.sample_rate)
            hideal = hHigh - hLow

            self.band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(hideal)

        band_pass_filter = self.band_pass.to(self.device)

        # Frequency masking: We randomly mask (1/5)th of no. of sinc filters channels (70)
        if (mask == True):
            for i1 in range(1):
                A = np.random.uniform(0, 14)
                A = int(A)
                A0 = random.randint(0, band_pass_filter.shape[0] - A)
                band_pass_filter[A0:A0 + A, :] = 0
        else:
            band_pass_filter = band_pass_filter

        self.filters = (band_pass_filter).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(x, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)


# ------------------------------
# Residual_block (原始)
# ------------------------------
class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super(Residual_block, self).__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
            self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                                   out_channels=nb_filts[1],
                                   kernel_size=(2, 3),
                                   padding=(1, 1),
                                   stride=1)
        self.selu = nn.SELU(inplace=True)

        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=nb_filts[1],
                                kernel_size=(2, 3),
                                padding=(1, 1),
                                stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False
        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x):
        identity = x

        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
            out = self.conv1(x)
        else:
            x = x
            out = self.conv_1(x)

        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out


# ------------------------------
# RawGAT_ST with TreeFus-ST only
# ------------------------------
class RawGAT_ST(nn.Module):
    def __init__(self, d_args, device, use_treefus_st=True):
        """
        d_args: model config dict (keep original expectations)
        device: torch device
        use_treefus_st: if True, enable TreeFus-ST for cross-branch fusion
        """
        super(RawGAT_ST, self).__init__()
        self.device = device
        self.use_treefus_st = use_treefus_st

        # ---------------- Sinc conv ----------------
        self.conv_time = CONV(device=self.device,
                              out_channels=d_args['out_channels'],
                              kernel_size=d_args['first_conv'],
                              in_channels=d_args['in_channels']
                              )
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        # ---------------- Encoders (original) ----------------
        self.encoder1 = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=d_args['filts'][1], first=True)),
            nn.Sequential(Residual_block(nb_filts=d_args['filts'][1])),
            nn.Sequential(Residual_block(nb_filts=d_args['filts'][2])),
            nn.Sequential(Residual_block(nb_filts=d_args['filts'][3])),
            nn.Sequential(Residual_block(nb_filts=d_args['filts'][3])),
            nn.Sequential(Residual_block(nb_filts=d_args['filts'][3]))
        )

        self.encoder2 = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=d_args['filts'][1], first=True)),
            nn.Sequential(Residual_block(nb_filts=d_args['filts'][1])),
            nn.Sequential(Residual_block(nb_filts=d_args['filts'][2])),
            nn.Sequential(Residual_block(nb_filts=d_args['filts'][3])),
            nn.Sequential(Residual_block(nb_filts=d_args['filts'][3])),
            nn.Sequential(Residual_block(nb_filts=d_args['filts'][3]))
        )

        # ---------------- GAT + Pool for spectral/temporal (original) ----------------
        self.GAT_layer1 = GraphAttentionLayer(d_args['filts'][-1][-1], 32)
        self.pool1 = Pool(0.64, 32, 0.3)

        self.GAT_layer2 = GraphAttentionLayer(d_args['filts'][-1][-1], 32)
        self.pool2 = Pool(0.81, 32, 0.3)


        # ---------------- TreeFus-ST as final spectral-temporal fusion ----------------
        # It is applied after spectral and temporal node features are aligned and concatenated
        if self.use_treefus_st:
            self.treefus_st = TreeFusSTLayer(in_dim=64, out_dim=16)  # maps to 16-d features like original GAT3 out
            # Original third GAT and pool (operate on TreeFus-ST output dim 16)
            self.GAT_layer3 = GraphAttentionLayer(16, 16)
            self.pool3 = Pool(0.64, 16, 0.3)
        else:
            # If TreeFus-ST is disabled, use a simple projection from 32 to 16
            self.treefus_st = None
            self.proj_no_tree = nn.Linear(64, 16)
            self.GAT_layer3 = GraphAttentionLayer(16, 16)
            self.pool3 = Pool(0.64, 16, 0.3)

        # Projection layers (keep original shapes as much as possible)
        self.proj1 = nn.Linear(14, 12)
        self.proj2 = nn.Linear(23, 12)
        self.proj = nn.Linear(16, 1)


        # classifier
        self.proj_node = nn.Linear(7, 2)

    def forward(self, x, Freq_aug=False):
        """
        x: (#bs, samples)
        """
        nb_samp = x.shape[0]
        len_seq = x.shape[1]

        x = x.view(nb_samp, 1, len_seq)

        # Freq masking during training only
        if (Freq_aug == True):
            x = self.conv_time(x, mask=True)
        else:
            x = self.conv_time(x, mask=False)

        # treat as 2-D image
        x = x.unsqueeze(dim=1)  # 2-D (#bs,1,F,T)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # ---------------- spectral branch ----------------
        e1 = self.encoder1(x)  # [#bs, C(64), F(23), T(29)]
        x_max, _ = torch.max(torch.abs(e1), dim=3)  # [#bs, C(64), F(23)]
        x_gat1 = self.GAT_layer1(x_max.transpose(1, 2))  # [#bs, N(F)=23, 32]

        x_gat1_enh = x_gat1

        x_pool1 = self.pool1(x_gat1_enh)
        out1 = self.proj1(x_pool1.transpose(1, 3))
        out1 = out1.view(out1.shape[0], out1.shape[1], out1.shape[3])  # [#bs, 32, #node1]

        # ---------------- temporal branch ----------------
        e2 = self.encoder2(x)
        x_max2, _ = torch.max(torch.abs(e2), dim=2)  # [#bs, C(64), T(29)]
        x_gat2 = self.GAT_layer2(x_max2.transpose(1, 2))  # [#bs, N(T)=29, 32]

        x_gat2_enh = x_gat2

        x_pool2 = self.pool2(x_gat2_enh)
        out2 = self.proj2(x_pool2.transpose(1, 3))
        out2 = out2.view(out2.shape[0], out2.shape[1], out2.shape[3])  # [#bs, 32, #node2]


        # ---------------- align node counts and produce node-wise features for TreeFus-ST ----------------
        # choose minimal node count to align (we will fuse features at node-level)
        N1 = out1.shape[-1]
        N2 = out2.shape[-1]
        Nmin = min(N1, N2)
        out1_trim = out1[:, :, :Nmin]  # [B, 32, Nmin]
        out2_trim = out2[:, :, :Nmin]  # [B, 32, Nmin]

        # permute to node-first representation for TreeFus-ST: [B, N, d]
        out1_nodes = out1_trim.permute(0, 2, 1).contiguous()
        out2_nodes = out2_trim.permute(0, 2, 1).contiguous()

        # concatenate spectral and temporal node features before TreeFus-ST
        out_concat = torch.cat([out1_nodes, out2_nodes], dim=-1)  # [B, Nmin, 64]

        # ---------------- TreeFus-ST fusion ----------------
        if self.use_treefus_st:
            tree_out = self.treefus_st(out_concat)  # [B, Nmin, 16]
        else:
            tree_out = self.proj_no_tree(out_concat)  # [B, Nmin, 16]

        # ---------------- optionally pass through GAT_layer3 and pool3 (保持原 pipeline) ----------------
        x_gat3 = self.GAT_layer3(tree_out)  # [B, Nmin, 16]
        x_pool3 = self.pool3(x_gat3)
        out_proj = self.proj(x_pool3).flatten(1)  # [B, 7] (保持原形状假设)
        output = self.proj_node(out_proj)  # [B, 2]

        return output

    # keep original helper functions
    def _make_layer(self, nb_blocks, nb_filts, first=False):
        layers = []
        for i in range(nb_blocks):
            first = first if i == 0 else False
            layers.append(Residual_block(nb_filts=nb_filts, first=first))
            if i == 0: nb_filts[0] = nb_filts[1]
        return nn.Sequential(*layers)

    def summary(self, input_size, batch_size=-1, device="cuda", print_fn=None):
        if print_fn is None:
            print_fn = print
        model = self

        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)

                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                try:
                    summary[m_key]["input_shape"] = list(input[0].size())
                except:
                    summary[m_key]["input_shape"] = str(type(input))
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    try:
                        summary[m_key]["output_shape"] = list(output.size())
                        summary[m_key]["output_shape"][0] = batch_size
                    except:
                        summary[m_key]["output_shape"] = str(type(output))

                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params

            if (
                    not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
                    and not (module == model)
            ):
                hooks.append(module.register_forward_hook(hook))

        device = device.lower()
        assert device in [
            "cuda",
            "cpu",
        ], "Input device is not valid, please specify 'cuda' or 'cpu'"

        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        if isinstance(input_size, tuple):
            input_size = [input_size]
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
        summary = OrderedDict()
        hooks = []
        model.apply(register_hook)
        model(*x)
        for h in hooks:
            h.remove()

        print_fn("----------------------------------------------------------------")
        line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        print_fn(line_new)
        print_fn("================================================================")
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = "{:>20}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
            total_params += summary[layer]["nb_params"]
            try:
                total_output += np.prod(summary[layer]["output_shape"])
            except:
                total_output += 0
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]
            print_fn(line_new)
        print_fn("================================================================")
        print_fn("Total params: %0.2fM" % (total_params / 1e6))
        print_fn("Trainable params: %0.2fM" % (trainable_params / 1e6))

# End of model.py
