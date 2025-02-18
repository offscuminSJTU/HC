import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn.utils import constant_init

class TransformerPerceptor(nn.Module):

    def __init__(self, d_in=128, n_head=4, d_out=128, dropout=0.1, if_proj=False, use_xyz=True, embd='learned'):
        super(TransformerPerceptor, self).__init__()

        self.if_proj = if_proj
        self.use_xyz = use_xyz
        self.embd = embd

        if self.if_proj:
            self.proposal_proj = nn.Conv1d(128, 128, kernel_size=1) # 其他参数怎么配置?
        self.seed_proj = nn.Conv1d(256, 128, kernel_size=1)
        self.self_attn = nn.MultiheadAttention(d_in, n_head, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_in, n_head, dropout=dropout)
        self.self_dropout = nn.Dropout(dropout)
        self.cross_dropout = nn.Dropout(dropout)
        self.self_norm = nn.LayerNorm(d_in)
        self.cross_norm = nn.LayerNorm(d_in)
        self.hidden_dropout = nn.Dropout(dropout)
        self.final_dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_in)

        if self.embd == 'learned':
            self.pos_embd = PositionEmbeddingLearned(3, 128)

        self.linear1 = ConvModule(
            128,
            128,
            kernel_size=1,
            padding=0,
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU'),
            bias=True,
            inplace=True)

        self.linear2 = ConvModule(
            128,
            128,
            kernel_size=1,
            padding=0,
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU'),
            bias=True,
            inplace=True)

        self.init_weights()
        # nn.init.xavier_uniform_(self.proposal_proj.weight)
        # nn.init.xavier_uniform_(self.seed_proj.weight)

        # 这些nn.MultiHeadAttention自带，似乎自己就不用写了
        # self.self_proj = Proj(128, 128, 128, 128)
        # self.cross_proj = Proj(128, 128, 256, 128)

    def init_weights(self):
        for m in self.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        constant_init(self.self_norm, 1, bias=0)
        constant_init(self.cross_norm, 1, bias=0)

    def forward(self, proposal, seed, proposal_xyz, seed_xyz):
        """
        proposal: features of object proposals (16,128,256)
        seed: features of seed points [16, 256, 1024]
        proposal_xyz: base_xyz of proposals [16, 256, 3]
        seed_xyz: xyz of seed pts [16, 1024, 3]
        """
        # proposal = proposal.transpose(1, 2).contiguous() # (16, 256, 128)
        # seed = seed.transpose(1, 2).contiguous() # (16, 1024, 256)

        query, key = self.projection(proposal, seed)

        if self.use_xyz:
            if self.embd == 'learned':
                proposal_pos = self.pos_embd(proposal_xyz)
                seed_pos = self.pos_embd(seed_xyz)
            query = query + proposal_pos
            key = key + seed_pos

        # print('query: ', query.shape) ## [16, 128, 256]
        # print('key: ', key.shape) ## [16, 128, 1024]
        query = query.permute(2, 0, 1).contiguous()
        key = key.permute(2, 0, 1).contiguous()

        self_query = self.self_attn(query, query, value=query)[0] # 具体的怎么写?
        query = query + self.self_dropout(self_query)
        query = self.self_norm(query)
        # print('query: ', query.shape) ## [16, 256, 128]

        cross_query = self.cross_attn(query, key, value=key)[0]
        query = query + self.cross_dropout(cross_query)
        query = self.cross_norm(query)
        # print('query: ', query.shape) ## [256, 16, 128]

        query = query.permute(1, 2, 0).contiguous()

        query_ = self.linear2(self.hidden_dropout(self.linear1(query))) # 缺一个dropout和残差连接
        # query = query.transpose(1, 2).contiguous()
        query = query + self.final_dropout(query_)
        query = query.transpose(1, 2).contiguous()
        query = self.final_norm(query)
        query = query.transpose(1, 2).contiguous()

        return query

    def pos_embedding_cos(self, xyz):
        """

        :param xyz:
        :return: position embedding (cosine)
        """

        return

    def projection(self, proposal, seed):
        """
        Project features of proposals and seeds, keeping their d the same
        :param proposal:
        :param seed:
        :return: projected proposal features (query) & seed features (key)
        """
        key = self.seed_proj(seed)
        if self.if_proj:
            query = self.proposal_proj(proposal)
        else:
            query = proposal
        return query, key


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel=3, num_pos_feats=128):
        super(PositionEmbeddingLearned, self).__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class Proj(nn.Module):

    def __init__(self, q_d_in, q_d_out, kv_d_in, kv_d_out):
        super(Proj, self).__init__()
        self.proj_q = nn.Linear(q_d_in, q_d_out, bias=False)
        self.proj_k = nn.Linear(kv_d_in, kv_d_out, bias=False)
        self.proj_v = nn.Linear(kv_d_in, kv_d_out, bias=False)

        nn.init.xavier_uniform_(self.proj_q.weight)
        nn.init.xavier_uniform_(self.proj_k.weight)
        nn.init.xavier_uniform_(self.proj_v.ewight)

    def forward(self, q, k, v):
        q = self.proj_q(q)
        k = self.proj_k(k)
        v = self.proj_v(v)
        return q, k, v

