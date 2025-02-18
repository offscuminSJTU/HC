import torch
import torch.nn as nn
import numpy as np
from .mapping_utils import ProjectionLayer, MappingLayer, PrototypeMapping, PrimitiveProj, ProposalProj


class scene2scene_infoNCEloss(nn.Module):

    def __init__(self, temperature=0.2, in_c=None, out_c=None, if_proj=False):
        super(scene2scene_infoNCEloss, self).__init__()
        self.T = temperature
        self.p = if_proj
        self.proj = ProjectionLayer(in_c, out_c, if_proj)

    def forward(self, features):
        sim = self.proj(features)
        loss = self.infoNCEloss(sim)# .requires_grad()
        return loss

    def infoNCEloss(self, sim):
        b = sim.shape[0]
        sim = torch.exp(sim / self.T)
        sum_2N = torch.sum(sim, dim=1)
        l = 0
        for i in range(b):
            l -= torch.log(sim[i, (i + b//2) % b] / (sum_2N[i] - sim[i, i]))
        l /= (b-1)
        return l


class anchor2anchor_SimLoss(nn.Module):

    def __init__(self, temperature=0.2, in_c=128, out_c=None, if_proj=True):
        super(anchor2anchor_SimLoss, self).__init__()
        self.T = temperature
        self.mapping = MappingLayer(in_c, out_c, if_proj)

    def forward(self, anchor_features):
        sims = self.mapping(anchor_features) # len(sims): 2*b
        # print('a2asims: ', sims.shape)
        loss = self.simLoss(sims)
        return loss

    def simLoss(self, sim):
        b = sim.shape[0]
        l = 0
        sim /= self.T
        for i in range(b):
            top = torch.exp(sim[i, (i + b//2) % b] + sim[(i + b//2) % b, i])
            bottom = 0
            for j in range(b):
                if j == i:
                    continue
                bottom += torch.exp(sim[i, j] + sim[j, i])
            l -= torch.log(top / bottom)
        l /= (b-1)
        return l


class cls2cls_CLloss(nn.Module):
    def __init__(self, temperature=0.2, in_c=128, out_c=None, if_proj=False):
        super(cls2cls_CLloss, self).__init__()
        self.T = temperature
        self.mapping = PrototypeMapping(in_c, out_c, if_proj)

    def forward(self, features, way=None, shot=None):
        sim = self.mapping(features)
        loss = self.clsLoss_mm(sim)
        return loss

    def clsLoss_firstConly(self, sim):
        # sim:(8,8,4,4)
        b = sim.shape[0]
        way = sim.shape[2]
        sim /= self.T
        l = 0
        # positive: [:,:,i,i] for i in range(way)
        # negative: [:,:,i,not i] for i in range(way)
        use_sim = sim[..., 0, :] # (8,8,4) dim2: positive prototype only
        all_sim = torch.sum(use_sim, dim=1) / b # (8,4)
        pos_sim = torch.sum(use_sim[..., 0], dim=1) / b # (8,)
        self_allsim = torch.diagonal(use_sim, dim1=0, dim2=1).transpose(0,1).contiguous()
        self_possim = torch.diagonal(use_sim[..., 0], dim1=0, dim2=1)
        all_sim -= self_allsim
        pos_sim -= self_possim
        all_sim = torch.exp(all_sim)
        pos_sim = torch.exp(pos_sim)
        for i in range(b):
            l -= torch.log(pos_sim[i] / torch.sum(all_sim[i]))
        l /= b
        l *= 0.1 # lambda: 调整权重
        return l

    def clsLoss_mm(self, sim):
        """
        calculate similarity between way w and way v
        """
        # sim:(b,b,way,way)
        b = sim.shape[0]
        way = sim.shape[2]
        sim /= self.T
        """
        if way == 1:
        """
            # TODO: add context_component to sim
        # print('sim: ', sim.shape) # sim与3-shot无异
        # positive: [:,:,w,w] for w in range(way)
        # negative: [:,:,w,not w] for w in range(way)
        pos_sim = torch.sum(sim, dim=1) # (8,4,4) sim: ijwv, dim1:j
        self_sim = torch.diagonal(sim, dim1=0, dim2=1).permute(2, 0, 1).contiguous() # transpose之前:(way,way,b)
        # print('self_sim: ', self_sim.shape)
        pos_sim -= self_sim # j |= i
        pos_sim = torch.diagonal(pos_sim, dim1=1, dim2=2) / (b-1) # (b,way) (i,w) for i=1,2,...,b, for w=1,2,...,way
        pos_sim = torch.exp(pos_sim)
        # print('pos_sim: ', pos_sim.shape)
        neg_sim = torch.sum(sim, dim=1) / b # (i,w,v)
        neg_sim = torch.exp(neg_sim)
        sum_negsim = torch.sum(neg_sim, dim=2) - torch.diagonal(neg_sim, dim1=1, dim2=2) # (i,w)
        term = -torch.log(pos_sim / (pos_sim + sum_negsim)) # (i,w)
        # print('each term: ', term.shape)
        l = torch.sum(term) / b / way
        # print('l: ', l)
        l *= 0.1  # lambda: 调整权重
        return l

    def clsLoss_mf(self, sim, way=4, shot=5):
        """
        intra-sample inter-class contrastive loss
        ablation study: m&f
        """
        # sim: (b,b,feature_num, feature_num)
        # print('sim: ', sim.shape)
        sim /= self.T
        b = sim.shape[0]
        feature_num = sim.shape[2] # prototype num one batch
        proto_num = shot + 1 # prototype num for each class
        class_num = feature_num // proto_num # actual class num in one batch
        compen_idx = feature_num % proto_num # actual class < way时补足随机向量的个数
        sample_sim = torch.diagonal(sim, dim1=0, dim2=1).permute(2, 0, 1).contiguous() # (16, 24, 24)
        # print('sample_sim: ', sample_sim.shape)
        m_loc = torch.arange(0, class_num) * proto_num
        l = 0
        for w in range(class_num):
            pos_sim = sample_sim[:, m_loc[w], m_loc[w]+1 : m_loc[w]+proto_num] # (16, 5)
            # print('pos_sim: ', pos_sim.shape)
            pos_sim = torch.sum(pos_sim, dim=1) # (16)
            # print('pos_sim: ', pos_sim.shape)
            pos_sim = torch.exp(pos_sim / shot)
            neg_sim = 0
            for v in range(class_num):
                if v == w:
                    continue
                neg_sim += torch.exp(torch.sum(sample_sim[:, m_loc[w], m_loc[v]+1: m_loc[v]+proto_num], dim=1) / shot)
            if compen_idx != 0:
                neg_sim += torch.exp(sample_sim[:, m_loc[w], -compen_idx]) # (16)
                # print('neg_sim: ', torch.exp(sample_sim[:, m_loc[w], -compen_idx]).shape)
            term = -torch.log(pos_sim / (pos_sim + neg_sim)) # (16)
            # print('term: ', term.shape)
            l += torch.sum(term) / b
        deno = way if compen_idx == 0 else (class_num + 1)
        l /= deno
        # print('class_num: ', deno)
        l *= 0.1
        return l


class primitiveCLloss(nn.Module):

    def __init__(self, temperature=0.2, in_c=256, out_c=None, if_proj=False, weight=0.1, eps=1e-08):
        super(primitiveCLloss, self).__init__()
        self.T = temperature
        self.w = weight
        self.proj = PrimitiveProj(in_c, out_c, if_proj)
        self.eps = eps

    def forward(self, primlabel, features, prototype):
        prototype = prototype.detach()  # (120,256)
        prim_feature, prim_prototype = self.prim_feature_extractor(primlabel, features, prototype)
        # print('prim_prototype: ', prim_prototype)
        # print('prim_feature: ', prim_feature)
        sim_prototype, sim_point = self.proj(prim_feature, prim_prototype, prototype)
        primitive_protoNCEloss = self.class_agnostic_clLoss(sim_prototype, sim_point)
        return primitive_protoNCEloss

    def class_agnostic_clLoss(self, sim_prototype, sim_point, hardest_num=2):
        """
        primitive_protoNCEloss
        primitive_pointNCEloss
        """
        sim_prototype /= self.T
        proto_num = sim_prototype.shape[0]
        sim_prototype = torch.exp(sim_prototype)
        pos_sim = torch.diagonal(sim_prototype)
        all_sim = torch.sum(sim_prototype, dim=1)
        term = -torch.log(pos_sim / all_sim)
        primitive_protoloss = torch.sum(term) / proto_num
        primitive_protoloss *= self.w
        # print('sim_point: ', sim_point)

        sim_point /= self.T
        pos_sim = torch.diagonal(sim_point, dim1=0, dim2=1) .permute(2, 0, 1) # (120, 2, 2)
        # print("pos_sim: ", pos_sim.shape)
        pos_sim = torch.exp(pos_sim[:, 0, 1]) # (120,)
        # print('pos_sim: ', pos_sim)
        neg_sim = torch.sum(sim_point, dim=(2, 3)) / 4 # (120,120)
        neg_sim = torch.exp(neg_sim)
        neg_sim = torch.sum(neg_sim, dim=1) - torch.diagonal(neg_sim) # (120,)
        term = -torch.log(pos_sim / (pos_sim + neg_sim))
        primitive_pointloss = torch.sum(term) / proto_num
        primitive_pointloss *= self.w

        primitive_clloss = primitive_protoloss# primitive_pointloss # + primitive_protoloss
        # print('primitive_point_clloss: ', primitive_pointloss)
        return primitive_clloss

    def prim_feature_extractor(self, primlabel, features, prototype):
        """
        sim of primitive features
        """
        # extract prim_features from features
        # prim_feature matrix should be (120, feature_num, 256)
        # primlabel: ([bs]: (prim_ind, prim_feature_num)): feature_ind
        # features: (16, 1024, 256)
        # print('prototype: ', prototype.shape)
        features = features.permute(1, 0, 2).contiguous()
        # print('feature: ', features.shape)
        bs = len(primlabel)
        prim_num = primlabel[0].shape[0]
        prim_feature = [[] for i in range(prim_num)] # [120: (prim_feature_num, 256)] # prim_feature of whole batch
        prim_prototype = torch.zeros(prim_num, features.shape[-1]).cuda(0)
        for p in range(prim_num): # prim_num: 120
            for b in range(bs): # bs: 16
                prim_idx = primlabel[b][p]
                for i in prim_idx:
                    if i != -1:
                        feature = features[b, i].unsqueeze(0) # (1,256)
                        feature_norm = torch.norm(feature, p=2, dim=1)
                        feature = feature.div(feature_norm)
                        prim_feature[p].append(feature)
            # empty_prim = False
            if len(prim_feature[p]) == 0:
                prim_feature[p] = prototype[p].unsqueeze(0) # (1,256)
                # empty_prim = True
            else:
                prim_feature[p] = torch.cat(prim_feature[p], dim=0)
            one_prim = torch.mean(prim_feature[p], dim=0) # (256)
            features_norm = torch.norm(one_prim, p=2, dim=0) # if not empty_prim else self.eps
            one_prim = one_prim.div(features_norm)
            prim_prototype[p] = one_prim
        return prim_feature, prim_prototype


class proposal_clloss(nn.Module):

    def __init__(self, temperature=0.2, in_c=128, out_c=None, if_proj=False, weight=0.1):
        super(proposal_clloss, self).__init__()
        self.T = temperature
        self.w = weight
        self.proj = ProposalProj(in_c, out_c, if_proj)

    def forward(self, proposal_instance_mask, grouped_features, grouped_indices, context_compen):
        """

        :param proposal_instance_mask: label for proposal_contrast, [16, 256, 16]
        :param grouped_features: voted group features of seeds, [16, 128, 256, 16](now)/[16, 259, 256, 16](try)
        :param grouped_indices: prevent repeated seeds, [16, 256, 16]
        :param context_compen: random vector when background pts not enough, [1, 128]
        :return: proposal-wise contrast loss
        """
        # 按照group来计算, 共有bs*group_num个group
        bs, feature_dim, group_num, sample_num = grouped_features.shape
        grouped_features = grouped_features.permute(0, 2, 3, 1).contiguous() # (16, 256, 16, 128)
        grouped_indices = torch.reshape(grouped_indices, (-1, sample_num))
        proposal_instance_mask = torch.reshape(proposal_instance_mask, (-1, sample_num))
        grouped_features = torch.reshape(grouped_features, (-1, sample_num, feature_dim))
        total_group = bs * group_num # maximum of 4096
        loss = 0
        for g in range(bs * group_num):
            proposal_indices = grouped_indices[g] # [16]
            proposal_features = grouped_features[g] # [16, 128]
            proposal_labels = proposal_instance_mask[g] # [16]
            unique_indices, position = torch.unique(proposal_indices, return_inverse=True)
            # print('proposal_indices: ', proposal_indices.dtype) # int64
            unique_num = torch.max(position) + 1 # how many unique seed points in one group of grouped_indices
            # valid proposal information
            # proposal_indices = proposal_indices[:unique_num] # proposal seed indices # 这个用不到
            proposal_features = proposal_features[:unique_num, :] # proposal seed features
            proposal_labels = proposal_labels[:unique_num] # proposal seed instance labels

            unique_instance = torch.unique(proposal_labels)
            # print('proposal_labels: ', proposal_labels.dtype) # int64
            instance_num = len(unique_instance) # contains background
            if instance_num == 1 and unique_instance[0] == -1: # only background
                total_group -= 1
                continue
            elif torch.min(unique_instance) != -1: # no background
                bg_context = context_compen
                instance_num += 1
            else: # contains background and foreground instances
                bg_context = None

            one_group_features = self.feature_gather(unique_instance, proposal_labels, proposal_features, bg_context)
            one_group_loss = self.cal_loss(one_group_features, instance_num)
            # print('one_group_loss: ', one_group_loss)
            loss = loss + one_group_loss
        # print('loss: ', loss)

        loss = loss / total_group
        loss = loss * self.w
        return loss

    def feature_gather(self, unique_instance, proposal_labels, proposal_features, bg_context=None):
        """

        :param instance_num: number of different instances in one proposal, contains background
        :param unique_instance: unique instance mask in one proposal
        :param proposal_labels: instance masks of seed points in one proposal
        :param proposal_features: seed features in one proposal
        :param bg_context: background features
        :return: gathered proposal features, [bg_features, fg1_features, fg2_features, ...]
        """
        instance_features = []
        # first element of instance_features is background features
        if bg_context != None: # no background in current proposal
            instance_features.append(bg_context)
        else:
            bg_context = []
            bg_index = torch.where(proposal_labels == -1)
            for index in bg_index:
                bg_context.append(proposal_features[index, :])
            bg_context = torch.cat(bg_context, dim=0)
            instance_features.append(bg_context)

        for i in range(len(unique_instance)):
            fg_context = []
            if unique_instance[i] == -1:
                continue
            instance_index = torch.where(proposal_labels == unique_instance[i])
            for index in instance_index:
                fg_context.append(proposal_features[index, :])
            fg_context = torch.cat(fg_context, dim=0)
            instance_features.append(fg_context)

        return instance_features

    def cal_loss(self, instance_features, instance_num):
        """
        calculate contrastive loss of one proposal
        :param instance_features:
        :param instance_num: num of instances in one proposal, contains background
        :return:
        """
        for i in range(instance_num):
            instance_features[i] = self.proj(instance_features[i])

        term = 0 # contrastive loss of one proposal

        for i in range(1, instance_num): # foreground
            fg = instance_features[i] # contrastive loss of one instance
            fg_num = fg.shape[0]
            pos_sim = torch.mm(fg, fg.transpose(0, 1).contiguous())
            # print('pos_sim: ', pos_sim)
            pos_sim = torch.sum(pos_sim) / (fg_num * fg_num)
            pos_sim = pos_sim / self.T
            pos_sim = torch.exp(pos_sim)
            neg_sim = 0
            for j in range(instance_num):
                if j == i:
                    continue
                other = instance_features[j]
                other_num = other.shape[0]
                sim = torch.mm(fg, other.transpose(0, 1).contiguous())
                # print('neg_sim: ', sim)
                sim = torch.sum(sim) / (fg_num * other_num)
                sim = sim / self.T
                sim = torch.exp(sim)
                neg_sim = neg_sim + sim
            term = term - torch.log(pos_sim / (pos_sim + neg_sim))
            # print('pos_sim: ', pos_sim, 'neg_sim: ', neg_sim)

        term = term / (instance_num - 1)
        # print('term: ', term)
        return term
