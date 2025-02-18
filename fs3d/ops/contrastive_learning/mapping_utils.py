import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MappingLayer(nn.Module):
    def __init__(self, in_c=128, out_c=None, if_proj=True):
        super(MappingLayer, self).__init__()
        if out_c is None:
            out_c = in_c // 2
        self.q_proj = nn.Linear(in_c, out_c)
        self.k_proj = nn.Linear(in_c, out_c)
        self.v_proj = nn.Linear(in_c, out_c)
        self.p = if_proj

    def forward(self, features):
        # TODO: return vi|j, vj|i
        #  (先按照不进行projection的方式实现
        #  to decrease calculation amount, perform projection
        # features:(16*2,128,256)
        b = features.shape[0]
        features = features.transpose(1, 2).contiguous() # features(16,256,128)
        a = features.shape[1]  # number of anchors
        if self.p:
            q_f = self.q_proj(features) # (16,256,out_c)
            k_f = self.k_proj(features)
            v_f = self.v_proj(features) # (16,256,out_c)
        else:
            q_f, k_f, v_f = features, features, features
        d = q_f.shape[2] # 64
        # feat = features[:b//2, ...]
        # feat_t = features[b:, ...]
        # q_f, k_f, v_f = F.normalize(q_f, dim=2), F.normalize(k_f, dim=2), F.normalize(v_f, dim=2)
        q_f, k_f, v_f = (q_f.reshape((-1, d)).contiguous(), k_f.reshape((-1, d)).contiguous(),
                         v_f.reshape((-1, d)).contiguous()) # (b*a, d)
        a_f = torch.mm(q_f, k_f.transpose(0, 1).contiguous()) # (b*a, b*a)
        a_f = F.softmax((a_f / math.sqrt(d)), dim=1)
        qkv_f = torch.mm(a_f, v_f) # (b*a,b*a),(b*a,d) -> (b*a,d) # 相似度整合过的value
        qkv_f, v_f = F.normalize(qkv_f), F.normalize(v_f) # l2_normalized value
        # sim = sim.reshape((b, a, b, a)).transpose(1, 2).contiguous() # (b,b,a,a)
        # A_ot = torch.matmul(feat, feat_t.transpose(1, 2).contiguous()) # (256,128)*(128,256) -> (256,256)
        # A_to = A_ot.transpose(1, 2).contiguous()
        # value_t = l2_normalize(torch.matmul(A_ot, feat_t))
        # value_o = l2_normalize(torch.matmul(A_to, feat))
        sim = torch.mm(v_f, qkv_f.transpose(0, 1).contiguous()) # (b*a, b*a)
        sim = sim.reshape((b, a, b, a)).transpose(1, 2).contiguous() # (b,b,a,a)
        sim_s = torch.zeros((b, b)).cuda(0)
        for i in range(a):
            # print('anchor sim: ', sim[..., i, i].shape)
            sim_s += sim[..., i, i] # (16,16)
        sim_s /= a # (b,b)
        return sim_s


class ProjectionLayer(nn.Module):

    def __init__(self, in_c=None, out_c=None, if_proj=False, eps=1e-08):
        super(ProjectionLayer, self).__init__()
        self.p = if_proj
        if self.p:
            self.global_proj = nn.Linear(in_c, out_c)
        self.eps = torch.tensor(eps)
        self.dist2 = nn.PairwiseDistance(p=2)

    def forward(self, features):
        # features:(16*2,128,256)
        b = features.shape[0] # 32
        eps = self.eps.repeat(1, b)
        global_feats = torch.mean(features, dim=2) # (16*2,128)
        # print('mean features: ', features.shape)
        if self.p:
            global_feats = self.global_proj(global_feats)
        global_feats = F.normalize(global_feats)  # default: p=2,dim=1 # (16,128)
        sim = torch.mm(global_feats, global_feats.transpose(0, 1).contiguous()) # sim:(32,32)
        dist = self.dist2(global_feats, torch.tensor(0)).unsqueeze(0) # (1,16)
        # print('2dist: ', dist.shape)
        deno = torch.ones((b, b)).cuda(0) # (32,32)
        for i in range(b):
            comparison = torch.cat([dist[0, i].repeat(1, b), dist, eps.cuda(0)], dim=0) # (3,16)
            max_dist = torch.max(comparison, dim=0)[0].reshape((1, -1))
            deno[i] = max_dist
        sim = sim / deno # (16,16)
        # print('deno: ', torch.min(deno))
        # sim = F.cosine_similarity(global_feats[:b, ...], global_feats[b:, ...], dim=1)
        # print('global_feats: ', global_feats.shape)
        # print('similarity: ', sim.shape)
        return sim


class PrototypeMapping(nn.Module):

    def __init__(self, in_c=128, out_c=None, if_proj=False, eps=1e-08):
        super(PrototypeMapping, self).__init__()
        self.p = if_proj
        if self.p:
            if out_c is None:
                out_c = in_c // 2
            self.proto_proj = nn.Linear(in_c, out_c)
        self.eps = torch.tensor(eps)
        # self.dist2 = nn.PairwiseDistance(p=2)

    def forward(self, features):
        # features: (2b, way(4), 128)
        # b = features.shape[0]
        # way = features.shape[1]
        if self.p:
            features = self.proto_proj(features)
        features = F.normalize(features, dim=2)  # default: p=2, dim=1 # (2b, way(4), 128)
        # print('features: ', features.shape)
        sim = torch.einsum('bik,ljk->blij', features, features) # (2b, 2b, way, way)
        # print('sim of features: ', sim.shape)
        # print('sim: ', sim[0,4,...], sim[0,5,...])
        # deno = torch.ones((b, b, way, way)).cuda(0) # (32,32)
        # deno[torch.where(sim == 0)] = self.eps
        # sim = sim / deno # (16,16)
        return sim


class PrimitiveProj(nn.Module):

    def __init__(self, in_c=256, out_c=None, if_proj=False, eps=1e-08):
        super(PrimitiveProj, self).__init__()
        self.p = if_proj
        if self.p:
            if out_c is None:
                out_c = in_c // 2
            self.proto_proj = nn.Linear(in_c, out_c)
        self.eps = torch.tensor(eps)
        # self.dist2 = nn.PairwiseDistance(p=2)

    def forward(self, features, prototypes, memory_bank, hardest_num=2):
        # features: [120, (n,256)]
        # prototypes: (120,256)
        # memory_bank: (120,256)
        # sim between prototypes and memory_bank
        if self.p:
            prototypes = self.proto_proj(prototypes)
            memory_bank = self.proto_proj(memory_bank)
            for i in range(len(features)):
                features[i] = self.proto_proj(features[i])
        prototypes = F.normalize(prototypes, dim=1)
        memory_bank = F.normalize(memory_bank, dim=1)
        for i in range(len(features)):
            features[i] = F.normalize(features[i], dim=1)
        sim_prototype = torch.mm(prototypes, memory_bank.transpose(0, 1).contiguous()) # (120,120)

        point_feature = torch.zeros(memory_bank.shape[0], hardest_num, memory_bank.shape[1]) # (120, 2, 256)
        for i in range(len(features)): # 120
            if len(features[i]) == 1:
                point_feature[i][0], point_feature[i][1] = features[i][0], memory_bank[i]
            else:
                sim_tmp = torch.mm(features[i], features[i].transpose(0, 1).contiguous())
                min_pos = torch.argmin(sim_tmp)
                min_row = min_pos % sim_tmp.shape[1]
                min_col = min_pos // sim_tmp.shape[1]
                point_feature[i][0], point_feature[i][1] = features[i][min_row], features[i][min_col]
        sim_point = torch.einsum('bik,ljk->blij', point_feature, point_feature) # (120, 120, 2, 2)
        # print('sim_point: ', sim_point)
        # print(pause)
        return sim_prototype, sim_point


class ProposalProj(nn.Module):
    def __init__(self, in_c=128, out_c=None, if_proj=False):
        super(ProposalProj, self).__init__()
        self.p = if_proj
        if self.p:
            if out_c is None:
                out_c = in_c // 2
            self.proposal_proj = nn.Linear(in_c, out_c)

    def forward(self, x):
        if self.p:
            return F.normalize(self.proposal_proj(x), dim=1)
        else:
            return F.normalize(x, dim=1)
