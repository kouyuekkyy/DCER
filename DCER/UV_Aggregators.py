import torch
import torch.nn as nn
import torch.nn.functional as F

# from MTGE.Attention import Attention
# from MTGE.New_TextCnn import TextCNN
from Attention import Attention
from New_TextCnn import TextCNN

class UV_Aggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """
    # def __init__(self, v2e, r2e, u2e, embed_dim, word_dim, vocab_size, filters_num, filter_sizes, seq_len, cuda, uv=True):
    def __init__(self, v2e, r2e, u2e, initW, opts, cuda, uv=True):
        super(UV_Aggregator, self).__init__()
        self.uv = uv
        self.v2e = v2e
        self.r2e = r2e
        self.u2e = u2e
        self.device = cuda
        self.embed_dim = opts.embed_dim
        self.word_dim = opts.word_dim
        self.vocab_size = opts.vocab_size_u if uv == True else opts.vocab_size_v
        self.filters_num = opts.num_filters
        self.filter_sizes = opts.filter_sizes
        self.seq_len = opts.seq_len
        self.opts = opts
        self.ourl = opts.ourl
        self.murl = opts.murl

        if self.ourl == 'all':
            self.w_r1 = nn.Linear(self.embed_dim * 2 + 300, self.embed_dim)

        self.w_r1 = nn.Linear(self.embed_dim + 300, self.embed_dim)

        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att = Attention(self.embed_dim)
        self.textCnn = TextCNN(self.word_dim, self.vocab_size, self.filters_num, self.filter_sizes, self.seq_len, initW, cuda)

    def forward(self, nodes, history_uv, history_r, history_w):
        embed_matrix = torch.empty(len(history_uv), self.embed_dim, dtype=torch.float).to(self.device)

        for i in range(len(history_uv)):
            history = history_uv[i]  # 邻居
            num_histroy_item = len(history)
            tmp_label = history_r[i]
            tmp_review = history_w[i]

            if self.uv == True:
                # user component
                e_uv = self.v2e.weight[history]      # 邻居i的嵌入
                uv_rep = self.u2e.weight[nodes[i]]     # 当前节点的嵌入
            else:
                # item component
                e_uv = self.u2e.weight[history]
                uv_rep = self.v2e.weight[nodes[i]]

            e_r = self.r2e.weight[tmp_label]
            e_w = self.textCnn(tmp_review)

            if self.ourl == 'all':
                x = torch.cat((e_uv, e_r, e_w), 1)

            x = F.relu(self.w_r1(x))
            o_history = F.relu(self.w_r2(x))  # 意见交互向量

            if self.opts.murl == 'all':
                att_w = self.att(o_history, uv_rep, num_histroy_item)  # 注意力分数
                att_history = torch.mm(o_history.t(), att_w)  # 加权求和
                att_history = att_history.t()

            embed_matrix[i] = att_history
        to_feats = embed_matrix
        return to_feats