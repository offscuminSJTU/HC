# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import ConvModule
from torch import nn as nn
from torch.nn import functional as F

from fs3d.ops import (GroupAll, Points_Sampler, QueryAndGroup,
                         gather_points, grouping_operation)
from .builder import SA_MODULES

from mmcv.cnn import xavier_init
from mmcv.cnn.utils import constant_init, kaiming_init

import copy
from functools import partial

from fs3d.prototypical_vote_info import cls_gather, cls_prototype, cls_gather_grad, cls_prototype_grad
from mmcv.cnn.bricks.norm import build_norm_layer

from fs3d.ops import geo_knn, knn
from fs3d.ops.geo_utils.geodesic_utils import cal_geodesic_vectorize, cal_geodesic_protonet
import numpy as np
from fs3d.ops import scene2scene_infoNCEloss, anchor2anchor_SimLoss, cls2cls_CLloss, proposal_pairs_fb, proposal_clloss

from .cross_refinement import TransformerPerceptor

class BasePointSAModule(nn.Module):
    """Base module for point set abstraction module used in PointNets.

    Args:
        num_point (int): Number of points.
        radii (list[float]): List of radius in each ball query.
        sample_nums (list[int]): Number of samples in each ball query.
        mlp_channels (list[list[int]]): Specify of the pointnet before
            the global pooling for each scale.
        fps_mod (list[str]: Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
            F-FPS: using feature distances for FPS.
            D-FPS: using Euclidean distances of points for FPS.
            FS: using F-FPS and D-FPS simultaneously.
        fps_sample_range_list (list[int]): Range of points to apply FPS.
            Default: [-1].
        dilated_group (bool): Whether to use dilated ball query.
            Default: False.
        use_xyz (bool): Whether to use xyz.
            Default: True.
        pool_mod (str): Type of pooling method.
            Default: 'max_pool'.
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Default: False.
        grouper_return_grouped_xyz (bool): Whether to return grouped xyz in
            `QueryAndGroup`. Defaults to False.
        grouper_return_grouped_idx (bool): Whether to return grouped idx in
            `QueryAndGroup`. Defaults to False.
    """

    def __init__(self,
                 num_point,
                 radii,
                 sample_nums,
                 mlp_channels,
                 fps_mod=['D-FPS'],
                 fps_sample_range_list=[-1],
                 dilated_group=False,
                 use_xyz=True,
                 pool_mod='max',
                 normalize_xyz=False,
                 grouper_return_grouped_xyz=False,
                 grouper_return_grouped_idx=False,
                 use_cls_refine=False,
                 transformer_nhead=4,
                 transformer_dropout=0.1):
        super(BasePointSAModule, self).__init__()

        assert len(radii) == len(sample_nums) == len(mlp_channels)
        assert pool_mod in ['max', 'avg']
        assert isinstance(fps_mod, list) or isinstance(fps_mod, tuple)
        assert isinstance(fps_sample_range_list, list) or isinstance(
            fps_sample_range_list, tuple)
        assert len(fps_mod) == len(fps_sample_range_list)

        if isinstance(mlp_channels, tuple):
            mlp_channels = list(map(list, mlp_channels))
        self.mlp_channels = mlp_channels

        if isinstance(num_point, int):
            self.num_point = [num_point]
        elif isinstance(num_point, list) or isinstance(num_point, tuple):
            self.num_point = num_point
        else:
            raise NotImplementedError('Error type of num_point!')

        self.pool_mod = pool_mod
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.fps_mod_list = fps_mod
        self.fps_sample_range_list = fps_sample_range_list

        self.points_sampler = Points_Sampler(self.num_point, self.fps_mod_list,
                                             self.fps_sample_range_list)

        for i in range(len(radii)):
            radius = radii[i]
            sample_num = sample_nums[i]
            if num_point is not None:
                if dilated_group and i != 0:
                    min_radius = radii[i - 1]
                else:
                    min_radius = 0
                grouper = QueryAndGroup(
                    radius,
                    sample_num,
                    min_radius=min_radius,
                    use_xyz=use_xyz,
                    normalize_xyz=normalize_xyz,
                    return_grouped_xyz=grouper_return_grouped_xyz,
                    return_grouped_idx=grouper_return_grouped_idx)
            else:
                grouper = GroupAll(use_xyz)
            self.groupers.append(grouper)

        self.use_cls_refine = use_cls_refine
        if self.use_cls_refine:
            self.cls_point_refinement = CLS_TransformerLayer(
                d_model=self.mlp_channels[0][-1],
                nhead=transformer_nhead,
                dropout=transformer_dropout,
            )

            self.fuse = Fuse()

            """
            contrastive loss modules
            comments: 先尝试放在PVM之后
            """
            self.s2sloss = scene2scene_infoNCEloss()
            self.a2aloss = anchor2anchor_SimLoss()
            self.c2closs = cls2cls_CLloss(if_proj=True)
            self.attn_perceptor = TransformerPerceptor()
            self.pfbloss = proposal_clloss(if_proj=True)

        self.transformer_nhead = transformer_nhead
        self.transformer_dropout = transformer_dropout

    def _sample_points(self, points_xyz, features, indices, target_xyz):
        """Perform point sampling based on inputs.

        If `indices` is specified, directly sample corresponding points.
        Else if `target_xyz` is specified, use is as sampled points.
        Otherwise sample points using `self.points_sampler`.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor): (B, C, N) features of each point.
                Default: None.
            indices (Tensor): (B, num_point) Index of the features.
                Default: None.
            target_xyz (Tensor): (B, M, 3) new_xyz coordinates of the outputs.

        Returns:
            Tensor: (B, num_point, 3) sampled xyz coordinates of points.
            Tensor: (B, num_point) sampled points' index.
        """


        xyz_flipped = points_xyz.transpose(1, 2).contiguous()
        if indices is not None:
            assert (indices.shape[1] == self.num_point[0])
            new_xyz = gather_points(xyz_flipped, indices).transpose(
                1, 2).contiguous() if self.num_point is not None else None
        elif target_xyz is not None:
            new_xyz = target_xyz.contiguous()
        else:
            indices = self.points_sampler(points_xyz, features)
            new_xyz = gather_points(xyz_flipped, indices).transpose(
                1, 2).contiguous() if self.num_point is not None else None

        return new_xyz, indices

    def feature_norm(self, feature):
        features_norm = torch.norm(feature, p=2, dim=1)
        feature = feature.div(features_norm.unsqueeze(1))
        return feature

    def _pool_features(self, features):
        """Perform feature aggregation using pooling operation.

        Args:
            features (torch.Tensor): (B, C, N, K)
                Features of locally grouped points before pooling.

        Returns:
            torch.Tensor: (B, C, N)
                Pooled features aggregating local information.
        """
        if self.pool_mod == 'max':
            # (B, C, N, 1)
            new_features = F.max_pool2d(
                features, kernel_size=[1, features.size(3)])
        elif self.pool_mod == 'avg':
            # (B, C, N, 1)
            new_features = F.avg_pool2d(
                features, kernel_size=[1, features.size(3)])
        else:
            raise NotImplementedError

        return new_features.squeeze(-1).contiguous()

    """
        Modification:
        Self-defined function: geo_dist_embedding
        Calculate geo_dists embeddings for seed points
    """

    def cal_geo_embedding_fourier(self, geo_dists, feature_dim=256, normalize=True):
        # get_fourier_embeddings(self, xyz, num_channels=None, input_range=None):  # input_range = pc_dims
        # Follows - https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

        geo_dists = geo_dists.unsqueeze(3).repeat(1,1,1,3).contiguous()
        geo_dists = geo_dists.reshape(geo_dists.shape[0], -1, 3).contiguous()
        bsize, npoints = geo_dists.shape[0], geo_dists.shape[1] # 每个anchor都接收16个seed pts
        assert feature_dim > 0 and feature_dim % 2 == 0
        d_in, max_d_out = 3, feature_dim
        d_out = feature_dim // 2
        assert d_out <= max_d_out
        assert d_in == geo_dists.shape[-1]

        gauss_scale = 1.0
        d_pos = feature_dim
        gauss_B = torch.empty((d_in, d_pos // 2)).normal_()
        gauss_B *= gauss_scale

        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = geo_dists
        geo_dists = orig_xyz.clone()

        if normalize:
            for b in range(bsize):
                one_sample_dist = geo_dists[b, :, 0] # (4096,)
                # print('one_sample_dist', one_sample_dist.shape)
                neg_dist_idx = torch.where(one_sample_dist < 0)[0]
                # print('neg_dist_idx: ', neg_dist_idx)
                max_dist = torch.max(one_sample_dist) * 2
                # print('geo_dists:', geo_dists[b, neg_dist_idx])
                geo_dists[b, neg_dist_idx] = max_dist
                # print('geo_dists:', geo_dists[b, neg_dist_idx])
                geo_dists[b] = geo_dists[b] / max_dist
                # print('max_dist[b]:', max_dist)
                # print('geo_dists[b]:', torch.max(geo_dists[b]))
            # geo_dists = shift_scale_points(geo_dists, src_range=input_range)  # 相当于对geo_dist进行归一化
            # geo_dists = geo_dists / torch.max(torch.max(torch.max(geo_dists)))

        geo_dists *= 2 * np.pi
        geo_dists = geo_dists.float()
        xyz_proj = torch.mm(geo_dists.view(-1, d_in), gauss_B[:, :d_out]).view(bsize, npoints, d_out) # gauss_B
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

        # return batch x d_pos x npoints embedding
        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    """
    def cal_geo_embedding_sine(self, geo_dists, feature_dim=256, normalize=False):
        geo_dists = geo_dists.unsqueeze(3).repeat(1, 1, 1, 3)
        geo_dists = geo_dists.reshape(geo_dists.shape[0], -1, 3)
        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = geo_dists
        xyz = orig_xyz.clone()

        if self.normalize:
            # xyz = shift_scale_points(xyz, src_range=input_range)
            pass

        ndim = num_channels // xyz.shape[2]
        if ndim % 2 != 0:
            ndim -= 1
        # automatically handle remainder by assiging it to the first dim
        rems = num_channels - (ndim * xyz.shape[2])

        assert (
                ndim % 2 == 0
        ), f"Cannot handle odd sized ndim={ndim} where num_channels={num_channels} and xyz={xyz.shape}"

        final_embeds = []
        prev_dim = 0

        for d in range(xyz.shape[2]):
            cdim = ndim
            if rems > 0:
                # add remainder in increments of two to maintain even size
                cdim += 2
                rems -= 2

            if cdim != prev_dim:
                dim_t = torch.arange(cdim, dtype=torch.float32, device=xyz.device)
                dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)

            # create batch x cdim x nccords embedding
            raw_pos = xyz[:, :, d]
            if self.scale:
                raw_pos *= self.scale
            pos = raw_pos[:, :, None] / dim_t
            pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
            final_embeds.append(pos)
            prev_dim = cdim

        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds
    """
    def forward_with_geo_embedding(self, seed_xyz, seed_indices, grouped_seed_indices, indices, grouped_results):
        idx, dist2 = geo_knn(16, seed_xyz)  # seed_xyz: ordered by seed_indices, 不是从小到大排列的
        # 16 neighbors, 64 steps, 1.2 radius
        idx = idx.transpose(1, 2).contiguous()
        graph = (idx, dist2)
        del idx, dist2
        # print('neighbor metrix:', graph[0].shape)
        group_seed_idx_exp = grouped_seed_indices.reshape((grouped_seed_indices.shape[0], -1)).contiguous()
        group_idx_inseed = [torch.nonzero(torch.eq(group_seed_idx_exp[b][:, None], seed_indices[b]), as_tuple=True)[1] \
                                .reshape((grouped_seed_indices.shape[1], -1)).contiguous() for b in range(seed_indices.shape[0])]
        group_idx_inseed = torch.stack(group_idx_inseed, dim=0)  # (16,256,16)
        # print('group_idx_inseed', group_idx_inseed.shape, group_idx_inseed[0,1,...])
        geo_dists = cal_geodesic_protonet(graph=graph, seed_indices=indices, idx_inseed=group_idx_inseed, \
                                          max_step=32, radius=0.6)  # seed_indices(anchor indices)
        del graph
        geo_dists = torch.stack(geo_dists, dim=0)  # (16,256,1024) # 将group_idx_inseed传入，提前中断以加速计算
        # print('geo_dists:', geo_dists.shape)
        # seed_idx_sort = torch.sort(seed_indices, dim=1)[0].contiguous()
        # print('seed_idx_sort', seed_idx_sort)
        sct_idx = torch.arange(0, 256).unsqueeze(1).repeat(1, 16).reshape(-1).contiguous()
        using_dist = torch.zeros(group_idx_inseed.shape)
        for b in range(group_idx_inseed.shape[0]):
            using_dist[b] = geo_dists[b][sct_idx, group_idx_inseed[b].reshape(-1)].reshape(
                group_idx_inseed.shape[1:]).contiguous()
        # print('using_dist', using_dist[0,0,:])
        del geo_dists
        # calculate geo_dist embeddings
        geo_embedding = self.cal_geo_embedding_fourier(using_dist)  # torch.Size([16, 256, 4096]) (16,256,256*16)
        geo_embedding = geo_embedding.reshape((geo_embedding.shape[0], 256, 256, 16)).transpose(2, 1).contiguous().cuda(0)
        # print('geo_embedding:', torch.min(geo_embedding))
        # print('grouped_results:', torch.min(grouped_results))
        """
        cat geo_embedding & grouped_features, use self.fuse to fuse them
        (let net decide how to fuse embedding & feature)
        """
        grouped_results_ = torch.cat([grouped_results, geo_embedding], dim=1)
        # print('concat embededing & features: ', grouped_results.shape)
        # grouped_results[:, 3:, ...] = grouped_results[:, 3:, ...] + geo_embedding
        grouped_results = grouped_results + self.fuse(grouped_results_)
        # print('fused grouped_results: ', grouped_results.shape)
        return grouped_results

    def forward(
        self,
        points_xyz,
        features=None,
        indices=None,
        target_xyz=None,
        seed_xyz=None,
        cls_prototypes=None,
        seed_indices=None,
        stage="train"):
        """forward.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor): (B, C, N) features of each point.
                Default: None.
            indices (Tensor): (B, num_point) Index of the features.
                Default: None.
            target_xyz (Tensor): (B, M, 3) new_xyz coordinates of the outputs.

        Returns:
            Tensor: (B, M, 3) where M is the number of points.
                New features xyz.
            Tensor: (B, M, sum_k(mlps[k][-1])) where M is the number
                of points. New feature descriptors.
            Tensor: (B, M) where M is the number of points.
                Index of the features.
        """
        new_features_list = []

        # sample points, (B, num_point, 3), (B, num_point)
        new_xyz, indices = self._sample_points(points_xyz, features, indices,
                                               target_xyz) # indices are anchor indices in seed pts
        """
        new_xyz: voted & sampled pts coordinates (cluster center)
        points_xyz: voted pts coordinates
        seed_xyz: original pts coordinates
        features: voted pts feats
        """

        for i in range(len(self.groupers)):
            if seed_xyz != None:
                grouped_results, grouped_seeds_xyz, grouped_seed_indices = self.groupers[i](points_xyz, new_xyz, features, seeds_xyz=seed_xyz, seed_indices=seed_indices)
                # grouped_results: (16,259,256,16)
                # print('training stage, seed xyz && use_cls_refine: ', self.use_cls_refine)
                # print('grouped_seed_indices: ', grouped_seed_indices.shape) ## [16, 256, 16]
                # print('cls_prototypes: ', cls_prototypes.keys()) ## ['pts_semantic_mask', 'pts_instance_mask', 'context_compen', 'class_names', 'K_shot', 'N_way']
                """
                forward_with_geo_embedding
                """
                # grouped_results = self.forward_with_geo_embedding(seed_xyz, seed_indices, grouped_seed_indices, indices, grouped_results)
                # TODO:
                #  v检查geo_embedding和grouped_results的数量级，确认geo_embedding的融入是否会过分影响到grouped_results
                #  v确认geo_dists归一化的作用维度
                # Consider:
                # v use MLP to fuse geo_embedding and grouped_results
                # v残差连接(无优化)
                # fine-grind dist calculate
                # calculate geo_dists between seed pts, use geo_dist to guide votelayer
                # support为什么也要经过vote layer, votelayer的细节是怎样的, 有没有什么办法改进或者替换它
            else:
                grouped_results = self.groupers[i](points_xyz, new_xyz, features)

            # (B, mlp[-1], num_point, nsample)
            # print('use_cls_refine and seed_xyz: ', self.use_cls_refine, (seed_xyz != None)) ## 2 cases: True & True or False & False
            new_features = self.mlps[i](grouped_results) # (16,128,256,16)

            if self.use_cls_refine and isinstance(cls_prototypes, dict):
                proposal_fb_label = proposal_pairs_fb(grouped_seed_indices, cls_prototypes["pts_instance_mask"])
                # print('proposal_fb_label: ', proposal_fb_label.shape) # [16, 256, 16]
                pfb_clloss = self.pfbloss(proposal_fb_label, new_features, grouped_seed_indices, cls_prototypes['context_compen'])
                # print('pfb_clloss: ', pfb_clloss)
            # TODO:
            #  if CL loss used here, 目的是使相同场景的特征尽可能相似
            use_c2c_loss = False

            if self.use_cls_refine:
                group_num = new_features.shape[2]
                sample_num = new_features.shape[3]
                new_features = new_features.reshape(new_features.shape[0], new_features.shape[1], -1) # (16,128,4096)
                grouped_seeds_xyz = grouped_seeds_xyz.reshape(grouped_seeds_xyz.shape[0], grouped_seeds_xyz.shape[1], -1).permute(0,2,1).contiguous() # (16,4096,3)

                if isinstance(cls_prototypes, dict):
                    input_feature = {}
                    input_feature['fp_xyz'] = [grouped_seeds_xyz]
                    input_feature['fp_features'] = [new_features]
                    grouped_seed_indices = grouped_seed_indices.reshape(grouped_seed_indices.shape[0], -1) # (16,4096)?
                    input_feature['fp_indices'] = [grouped_seed_indices]

                    batch_list = cls_gather(input_feature, cls_prototypes["pts_semantic_mask"], cls_prototypes["pts_instance_mask"], points_xyz, seed_indices, class_names=cls_prototypes["class_names"])
                    batch_list_grad = cls_gather_grad(input_feature, cls_prototypes["pts_semantic_mask"], cls_prototypes["pts_instance_mask"], points_xyz, seed_indices, class_names=cls_prototypes["class_names"])
                    # print('batch_list: ', batch_list[0].shape)
                    # print('batch_list_grad: ', batch_list_grad[0].shape)
                    cls_prototypes_grad = cls_prototype_grad(batch_list_grad, cls_prototypes["context_compen"], num=cls_prototypes["K_shot"], way=cls_prototypes["N_way"])
                    # print('cls_prototypes_grad', cls_prototypes_grad.shape) # always 4-way
                    use_c2c_loss = True
                    c2c_loss = self.c2closs(cls_prototypes_grad, way=cls_prototypes["N_way"], shot=cls_prototypes["K_shot"])
                    cls_prototypes = cls_prototype(batch_list, cls_prototypes["context_compen"], num=cls_prototypes["K_shot"], way=cls_prototypes["N_way"])
                    # print('cls_prototypes: ', cls_prototypes.shape)
                    # print('c2c_loss: ', c2c_loss)
                    # TODO:
                    #  这里也可以使用cl loss
                    #  考虑temperature对不同标签的样本的区分度的影响, 希望相近的类别能保留相似之处, 比如softlabel相似类别与相异类别的差别
                    #  实现一个不切断梯度的版本; inference期间不调用这个版本
                else:
                    c2c_loss = None

                new_features = new_features.reshape(new_features.shape[0], new_features.shape[1], group_num, sample_num)

            new_features = self._pool_features(new_features) # (16,128,256)

            # if self.use_cls_refine:
            #     new_features = self.attn_perceptor(new_features, features, new_xyz, seed_xyz)

            """
            Calculate Contrastive Loss
            """
            """
            if self.use_cls_refine:
                s2s_loss = self.s2sloss(new_features)
                a2a_loss = self.a2aloss(new_features)
                cl_loss = [s2s_loss, a2a_loss]
                # print('s2sloss: ', s2s_loss, 'a2aloss: ', a2a_loss)
            else:
                cl_loss = None
            """

            # PHM
            if self.use_cls_refine:
                new_features, _ = self.cls_point_refinement(new_features, xyz=None, prototypes=cls_prototypes, stage=stage)
                # refined new_features:(16,128,256)
                # TODO:
                #  如果在这里使用对比损失，可以操作使同类proposal和prototype距离更近，可以增加一个相似性网络帮助判断
                #  另一方面，new_features也只是点云的特征而已，也可以使相同场景的特征尽可能相似
                #  是否loss返回的位置越往后，能训练到的网络权重越多，就越好
                if use_c2c_loss:
                    cl_loss = 0.1 * c2c_loss + pfb_clloss # c2c_loss +
                else:
                    cl_loss = None

            new_features_list.append(new_features)

        if self.use_cls_refine:
            # print('new_xyz: ', new_xyz.shape) ## [16, 256, 3]
            # print('seeds: ', seed_xyz.shape) ## [16, 1024, 3]
            # print('features: ', features.shape) ## [16, 256, 1024]
            return new_xyz, torch.cat(new_features_list, dim=1), indices, cl_loss # c2c_loss
        else:
            return new_xyz, torch.cat(new_features_list, dim=1), indices

@SA_MODULES.register_module()
class PointSAModuleMSG(BasePointSAModule):
    """Point set abstraction module with multi-scale grouping (MSG) used in
    PointNets.

    Args:
        num_point (int): Number of points.
        radii (list[float]): List of radius in each ball query.
        sample_nums (list[int]): Number of samples in each ball query.
        mlp_channels (list[list[int]]): Specify of the pointnet before
            the global pooling for each scale.
        fps_mod (list[str]: Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
            F-FPS: using feature distances for FPS.
            D-FPS: using Euclidean distances of points for FPS.
            FS: using F-FPS and D-FPS simultaneously.
        fps_sample_range_list (list[int]): Range of points to apply FPS.
            Default: [-1].
        dilated_group (bool): Whether to use dilated ball query.
            Default: False.
        norm_cfg (dict): Type of normalization method.
            Default: dict(type='BN2d').
        use_xyz (bool): Whether to use xyz.
            Default: True.
        pool_mod (str): Type of pooling method.
            Default: 'max_pool'.
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Default: False.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
    """

    def __init__(self,
                 num_point,
                 radii,
                 sample_nums,
                 mlp_channels,
                 fps_mod=['D-FPS'],
                 fps_sample_range_list=[-1],
                 dilated_group=False,
                 norm_cfg=dict(type='BN2d'),
                 use_xyz=True,
                 pool_mod='max',
                 normalize_xyz=False,
                 bias='auto',
                 use_cls_refine=False,
                 transformer_nhead=4,
                 transformer_dropout=0.1):
        super(PointSAModuleMSG, self).__init__(
            num_point=num_point,
            radii=radii,
            sample_nums=sample_nums,
            mlp_channels=mlp_channels,
            fps_mod=fps_mod,
            fps_sample_range_list=fps_sample_range_list,
            dilated_group=dilated_group,
            use_xyz=use_xyz,
            pool_mod=pool_mod,
            normalize_xyz=normalize_xyz,
            use_cls_refine=use_cls_refine,
            transformer_nhead=transformer_nhead,
            transformer_dropout=transformer_dropout)

        for i in range(len(self.mlp_channels)):
            mlp_channel = self.mlp_channels[i]
            if use_xyz:
                mlp_channel[0] += 3

            mlp = nn.Sequential()
            for i in range(len(mlp_channel) - 1):
                mlp.add_module(
                    f'layer{i}',
                    ConvModule(
                        mlp_channel[i],
                        mlp_channel[i + 1],
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=norm_cfg,
                        bias=bias))
            self.mlps.append(mlp)


@SA_MODULES.register_module()
class PointSAModule(PointSAModuleMSG):
    """Point set abstraction module with single-scale grouping (SSG) used in
    PointNets.

    Args:
        mlp_channels (list[int]): Specify of the pointnet before
            the global pooling for each scale.
        num_point (int): Number of points.
            Default: None.
        radius (float): Radius to group with.
            Default: None.
        num_sample (int): Number of samples in each ball query.
            Default: None.
        norm_cfg (dict): Type of normalization method.
            Default: dict(type='BN2d').
        use_xyz (bool): Whether to use xyz.
            Default: True.
        pool_mod (str): Type of pooling method.
            Default: 'max_pool'.
        fps_mod (list[str]: Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
        fps_sample_range_list (list[int]): Range of points to apply FPS.
            Default: [-1].
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Default: False.
    """

    def __init__(self,
                 mlp_channels,
                 num_point=None,
                 radius=None,
                 num_sample=None,
                 norm_cfg=dict(type='BN2d'),
                 use_xyz=True,
                 pool_mod='max',
                 fps_mod=['D-FPS'],
                 fps_sample_range_list=[-1],
                 normalize_xyz=False,
                 use_cls_refine=False,
                 transformer_nhead=4,
                 transformer_dropout=0.1):
        super(PointSAModule, self).__init__(
            mlp_channels=[mlp_channels],
            num_point=num_point,
            radii=[radius],
            sample_nums=[num_sample],
            norm_cfg=norm_cfg,
            use_xyz=use_xyz,
            pool_mod=pool_mod,
            fps_mod=fps_mod,
            fps_sample_range_list=fps_sample_range_list,
            normalize_xyz=normalize_xyz,
            use_cls_refine=use_cls_refine,
            transformer_nhead=transformer_nhead,
            transformer_dropout=transformer_dropout)


class CLS_TransformerLayer(nn.Module):

    def __init__(self, d_model, nhead=4, dim_feedforward=256,
                 dropout=0.1, dropout_attn=None,
                 activation="relu", normalize_before=True,
                 norm_fn_name="ln"):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout, inplace=True)

        _, self.norm_last = build_norm_layer(dict(type='BN1d'), d_model)

        self.linear_1 = ConvModule(
            d_model,
            d_model,
            kernel_size=1,
            padding=0,
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU'),
            bias=True,
            inplace=True)

        self.linear_2 = ConvModule(
            d_model,
            d_model,
            kernel_size=1,
            padding=0,
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU'),
            bias=True,
            inplace=True)
        self.init_weights()


    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def init_weights(self):
        for m in self.parameters():
            if m.dim() > 1:
                xavier_init(m, distribution='uniform')
        constant_init(self.norm_last, 1, bias=0)

    def feature_norm(self, feature):
        features_norm = torch.norm(feature, p=2, dim=1)
        feature = feature.div(features_norm.unsqueeze(1))
        return feature

    def forward_pre(self, tgt, memory=None,
                    tgt_mask = None,
                    memory_mask = None,
                    tgt_key_padding_mask = None,
                    memory_key_padding_mask = None,
                    pos = None,
                    query_pos = None,
                    return_attn_weights = False,
                    xyz = None,
                    prototypes=None,
                    stage="train"):

        # print('tgt_mask: ', tgt_mask) # None
        # print('tgt_key_padding_mask: ', tgt_key_padding_mask) # None

        tgt = tgt.permute(2, 0, 1)
        # print('tgt: ', tgt.shape) ## [256, 16, 128]
        q = tgt

        k = v = prototypes.permute(1, 0, 2)
        # print('k, v: ', k.shape, v.shape) ## [13, 16, 128]

        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)

        tgt = tgt.permute(1, 2, 0)

        tgt = self.norm_last(tgt)

        tgt = self.linear_2(self.linear_1(tgt))

        return tgt, None


    def forward(self, tgt, memory=None,
                tgt_mask = None,
                memory_mask = None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None,
                return_attn_weights = False,
                xyz=None,
                prototypes=None,
                stage="train"):

        return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, return_attn_weights, xyz=xyz, prototypes=prototypes, stage=stage)


"""
self-defined MLP to fuse geo_embedding and grouped_features
"""


class Fuse(nn.Module):

    def __init__(self):
        super(Fuse, self).__init__()
        self.fuse = nn.Linear(512, 256)
        self.bn = nn.BatchNorm1d(num_features=4096)
        self.act = nn.ReLU()
        """
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0.01, 0, 1)
            m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        """
        nn.init.normal_(self.bn.weight.data, 1.0, 0.02)
        nn.init.constant_(self.bn.bias.data, 0.0)
        nn.init.kaiming_normal_(self.fuse.weight.data, a=0, mode='fan_in')
        nn.init.constant_(self.fuse.bias.data, 0.0)

    def forward(self, x):
        b, _, anchor_num, k = x.shape
        x = x.reshape((b, x.shape[1], -1)).permute(0, 2, 1).contiguous()
        features = x[..., 3:]
        features = self.fuse(features)
        # print('fused features: ', features.shape)
        features = self.act(self.bn(features))
        # x[..., 3:] = features
        outs = torch.cat([x[..., :3], features], dim=-1)
        outs = outs.permute(0, 2, 1).reshape((b, -1, anchor_num, k)).contiguous()
        return outs

class BatchNormDim1Swap(nn.BatchNorm1d):
    """
    Used for nn.Transformer that uses a HW x N x C rep
    """

    def forward(self, x):
        """
        x: HW x N x C
        permute to N x C x HW
        Apply BN on C
        permute back
        """
        hw, n, c = x.shape
        x = x.permute(1, 2, 0)
        x = super(BatchNormDim1Swap, self).forward(x)
        # x: n x c x hw -> hw x n x c
        x = x.permute(2, 0, 1)
        return x


NORM_DICT = {
    "bn": BatchNormDim1Swap,
    "bn1d": nn.BatchNorm1d,
    "id": nn.Identity,
    "ln": nn.LayerNorm,
}

ACTIVATION_DICT = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leakyrelu": partial(nn.LeakyReLU, negative_slope=0.1),
}

WEIGHT_INIT_DICT = {
    "xavier_uniform": nn.init.xavier_uniform_,
}

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
