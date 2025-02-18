# Copyright (c) OpenMMLab. All rights reserved.
import torch
from fs3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from .single_stage import SingleStage3DDetector
import pickle
from mmcv import Config
from fs3d.datasets import build_dataset
from mmdet.datasets import build_dataloader
import os

from fs3d.ops import furthest_point_sample
from fs3d.prototypical_vote_info import cls_gather, cls_prototype_support

# from fs3d.ops import geo_knn
# from fs3d.ops.geo_utils.geodesic_utils import cal_geodesic_vectorize, cal_geodesic_protonet
from fs3d.ops import scene2scene_infoNCEloss, anchor2anchor_SimLoss


@DETECTORS.register_module()
class Prototypical_Votenet(SingleStage3DDetector):

    def __init__(self,
                 backbone,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(Prototypical_Votenet, self).__init__(
            backbone=backbone,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=None,
            pretrained=pretrained)

        self.stage = train_cfg["stage"]
        self.support_cfg = train_cfg["support_cfg"]
        self.support_anno = train_cfg["support_anno"]

        self.K_shot = train_cfg["K_shot"]
        self.N_way = train_cfg["N_way"]

        self.test_prototype = None
        self.test_cls_prototype = None

        self.save_index = 0

        self.instance_repeat_num = 3
        self.class_repeat_num = 3

        self.compen_context = torch.rand(1, 128).cuda()

        self.class_names = train_cfg["class_names"]
        self.few_shot_class = train_cfg["few_shot_class"]

        self.s2sloss = scene2scene_infoNCEloss(in_c=256, out_c=64, if_proj=False)
        self.a2aloss = anchor2anchor_SimLoss(in_c=256, out_c=32)

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      pts_semantic_mask=None,
                      pts_instance_mask=None,
                      gt_bboxes_ignore=None):
        """Forward of training.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            img_metas (list): Image metas.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): gt bboxes of each batch. (list)
            gt_labels_3d (list[torch.Tensor]): gt class labels of each batch.
            pts_semantic_mask (None | list[torch.Tensor]): point-wise semantic
                label of each batch.
            pts_instance_mask (None | list[torch.Tensor]): point-wise instance
                label of each batch.
            gt_bboxes_ignore (None | list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict: Losses.
        """

        self.test_prototype = None
        self.test_cls_prototype = None

        points_cat = torch.stack(points)
        # print('points: ', torch.sum(points_cat[0,:,3]>=0)) # 按照论文描述, PointNet++的输入为P->N*3
        # print('points: ', type(points))

        # points_cat, _ = self.aug_branch(points_cat, gt_bboxes_3d, rotate=True)
        """
        points = [points_cat[i] for i in range(points_cat.shape[0])]
        # print('aug_points: ', points[0].device)
        gt_labels_3d += gt_labels_3d
        pts_semantic_mask = pts_semantic_mask + pts_semantic_mask if pts_semantic_mask is not None else None
        pts_instance_mask = pts_instance_mask + pts_instance_mask if pts_instance_mask is not None else None
        gt_bboxes_ignore = gt_bboxes_ignore + gt_bboxes_ignore if gt_bboxes_ignore is not None else None
        # print('gt_labels_3d: ', len(gt_labels_3d))
        # gt_bboxes_3d is a list; length is batch_size
        # print('gt_bboxes_3d: ', len(gt_bboxes_3d)) # 有7个元素
        # print('points_cat: ', points_cat.shape)
        """

        x = self.extract_feat(points_cat)

        # s2s_loss = self.s2sloss(x['fp_features'][-1]) # (16,256,1024)
        # a2a_loss = self.a2aloss(x['fp_features'][-1])
        # cl_loss = [s2s_loss, a2a_loss]
        # cl_loss = s2s_loss
        """
        for key in x.keys():
             for p in range(len(x[key])):
                 x[key][p] = x[key][p][: x[key][p].shape[0]//2]
        """
        # idx, dist2 = geo_knn(64, points_cat) # idx(16,64,20000), dist2(16,20000,64)
        # idx = idx.transpose(1, 2)
        # graph = (idx, dist2)
        # del idx, dist2

        # seed_idx = x['fp_indices'][-1]
        # geo_dists = cal_geodesic_protonet(graph=graph, seed_indices=seed_idx, max_step=128, radius=0.05)
        # print('geo')

        if self.stage == "pretraining":
            cls_prototypes = {}
            cls_prototypes["pts_semantic_mask"] = pts_semantic_mask
            cls_prototypes["pts_instance_mask"] = pts_instance_mask
            cls_prototypes["context_compen"] = self.compen_context
            cls_prototypes["class_names"] = self.class_names

            cls_prototypes["K_shot"] = self.K_shot
            cls_prototypes["N_way"] = self.N_way
        else:
            vote_prototypes, cls_prototypes = self.support_prototype(len(points_cat), phase="train")

        vote_prototypes = {}
        vote_prototypes["pts_semantic_mask"] = pts_semantic_mask
        vote_prototypes["pts_instance_mask"] = pts_instance_mask
        vote_prototypes["fp_xyz"] = x['fp_xyz'][-1]
        vote_prototypes["fp_indices"] = x['fp_indices'][-1]

        bbox_preds = self.bbox_head(x, self.train_cfg.sample_mod, prototypes=vote_prototypes, cls_prototypes=cls_prototypes)
        loss_inputs = (points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask,
                       pts_instance_mask, img_metas)

        # bbox_preds['contrastive_loss'] = cl_loss

        losses = self.bbox_head.loss(
            bbox_preds, *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        # print('losses: ', losses.keys())

        return losses

    def aug_branch(self, points, gt_bboxes, rotate=True, angle_sigma=0.06, angle_clip=0.18):
        if rotate:
            rotated_data = points.clone().detach() # 切断内存和梯度
            for k in range(points.shape[0]):
                # 对xyz三个轴方向随机生成一个旋转角度
                angles = torch.clip(angle_sigma * torch.randn(3), -angle_clip, angle_clip)
                # 根据公式构建三个轴方向的旋转矩阵
                Rx = torch.tensor([[1, 0, 0],
                               [0, torch.cos(angles[0]), -torch.sin(angles[0])],
                               [0, torch.sin(angles[0]), torch.cos(angles[0])]])
                Ry = torch.tensor([[torch.cos(angles[1]), 0, torch.sin(angles[1])],
                               [0, 1, 0],
                               [-torch.sin(angles[1]), 0, torch.cos(angles[1])]])
                Rz = torch.tensor([[torch.cos(angles[2]), -torch.sin(angles[2]), 0],
                               [torch.sin(angles[2]), torch.cos(angles[2]), 0],
                               [0, 0, 1]])
                # 按照内旋方式:Z-Y-X旋转顺序获得整体的旋转矩阵
                R = torch.mm(Rz, torch.mm(Ry, Rx))
                shape_pc = points[k, :, 0:3].clone().detach()
                # shape_normal = points[k, :, 3:]
                # 分别对坐标与法向量进行旋转,整体公式应该为: Pt = (Rz * Ry * Rx) * P
                rotated_data[k, :, 0:3] = torch.mm(shape_pc.reshape((-1, 3)), R.cuda(0)).detach() # 切断梯度
                # rotated_data[k, :, 3:6] = torch.mm(shape_normal.reshape((-1, 3)), R)
                rotated_box = gt_bboxes[k].clone() # 切断内存
                # print('before rotate: ', rotated_box)
                rotated_box.rotate(R)
                # print('after rotate: ', rotated_box) # before rotate和after rotate的值确实不一样
                gt_bboxes.append(rotated_box)
                # print('compare boxes: ', rotated_box == gt_bboxes[k])
            points = torch.cat([points, rotated_data], dim=0)

        return points, gt_bboxes

    def simple_test(self, points, img_metas, pts_semantic_mask=None,
                      pts_instance_mask=None, imgs=None, rescale=False):
        """Forward of testing.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list): Image metas.
            rescale (bool): Whether to rescale results.

        Returns:
            list: Predicted 3d boxes.
        """

        points_cat = torch.stack(points)
        x = self.extract_feat(points_cat)

        if self.test_cls_prototype == None or self.test_cls_prototype == None:
            self.test_prototype, self.test_cls_prototype = self.support_prototype(len(points_cat), phase="test")
            prototypes = self.test_prototype
            cls_prototypes = self.test_cls_prototype
        else:
            prototypes = self.test_prototype
            cls_prototypes = self.test_cls_prototype

        bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod,  prototypes=prototypes, cls_prototypes=cls_prototypes, stage="test")
        bbox_preds.pop('contrastive_loss')
        bbox_list = self.bbox_head.get_bboxes(
            points_cat, bbox_preds, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False): # 这个函数有用到吗?
        """Test with augmentation."""

        points_cat = [torch.stack(pts) for pts in points]
        feats = self.extract_feats(points_cat, img_metas)

        aug_bboxes = []
        for x, pts_cat, img_meta in zip(feats, points_cat, img_metas):
            bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod)
            bbox_list = self.bbox_head.get_bboxes(
                pts_cat, bbox_preds, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]

    def support_image(self):

        anno_path = self.support_anno
        with open(anno_path, 'rb') as f:
            contents = pickle.load(f)

        config_path = self.support_cfg

        cfg = Config.fromfile(config_path)

        datasets = build_dataset(cfg.data.train)
        data_loader = build_dataloader(
            datasets,
            1,
            1,
            num_gpus=1,
            dist=False,
            # seed=cfg.seed,
            shuffle=False,
            runner_type='EpochBasedRunner',
            persistent_workers=cfg.data.get('persistent_workers', False))

        return data_loader, len(contents)

    def support_prototype(self, batch_size, phase="train"):

        support_image_loader, data_num = self.support_image()

        support_point_features = []
        support_indices = []
        support_xyz = []
        support_semantic_masks = []
        support_instance_masks = []

        support_cls_point_features = []
        support_cls_indices = []
        support_cls_xyz = []

        support_cls_point_xyz = []
        support_seed_indices = []

        prototypes = None

        for i, data_batch in enumerate(support_image_loader):
            points_n = data_batch["points"].data[0][0].unsqueeze(0).cuda()
            points_n = self.extract_feat(points_n)

            support_semantic_mask = data_batch["pts_semantic_mask"].data[0][0].cuda()
            support_instance_mask = data_batch["pts_instance_mask"].data[0][0].cuda()

            support_semantic_masks.append(support_semantic_mask)
            support_instance_masks.append(support_instance_mask)

            #########
            seed_points = points_n['fp_xyz'][-1]
            seed_features = points_n['fp_features'][-1]
            seed_indices = points_n['fp_indices'][-1]

            #########
            support_point_features.append(points_n['fp_features'][-1])
            support_indices.append(points_n['fp_indices'][-1])
            support_xyz.append(points_n['fp_xyz'][-1])
            #########

            vote_points, vote_features, vote_offset, middle_feature, before_attention_features, prim_clloss = self.bbox_head.PrototypicalVoteModule(
                seed_points, seed_features, prototypes=None, stage="train")

            sample_indices = furthest_point_sample(seed_points, 256)
            aggregation_inputs = dict(
                points_xyz=vote_points,
                features=vote_features,
                indices=sample_indices,
                seed_xyz=seed_points)
            new_xyz, indices = self.bbox_head.aggregation_ph_refinement._sample_points(aggregation_inputs["points_xyz"], aggregation_inputs["features"],
                                                   aggregation_inputs["indices"], None)
            grouped_results, grouped_seeds_xyz, grouped_seed_indices = self.bbox_head.aggregation_ph_refinement.groupers[0](aggregation_inputs["points_xyz"], new_xyz, aggregation_inputs["features"],
                                                                                seeds_xyz=aggregation_inputs["seed_xyz"], seed_indices=seed_indices)
            """forward_with_geo_embedding"""
            # grouped_results = self.bbox_head.aggregation_ph_refinement.forward_with_geo_embedding(seed_points, seed_indices, grouped_seed_indices, indices, grouped_results)
            # forward_with_geo_embedding(seed_xyz, seed_indices, grouped_seed_indices, indices, grouped_results)
            # print('support_prototype, use_cls_refine: ', self.bbox_head.aggregation_ph_refinement.use_cls_refine)

            new_features = self.bbox_head.aggregation_ph_refinement.mlps[0](grouped_results)

            new_features = new_features.reshape(new_features.shape[0], new_features.shape[1], -1)
            grouped_seed_indices = grouped_seed_indices.reshape(grouped_seed_indices.shape[0], -1)

            grouped_seeds_xyz = grouped_seeds_xyz.reshape(grouped_seeds_xyz.shape[0], grouped_seeds_xyz.shape[1], -1).permute(0, 2, 1)

            support_cls_point_features.append(new_features)
            support_cls_indices.append(grouped_seed_indices)
            support_cls_xyz.append(grouped_seeds_xyz)

            support_cls_point_xyz.append(aggregation_inputs["points_xyz"])
            support_seed_indices.append(seed_indices)

            if i == data_num - 1:
                break

        support_point_features = torch.cat(support_point_features, 0)
        support_indices = torch.cat(support_indices, 0)
        support_xyz = torch.cat(support_xyz, 0)

        support_x = {}
        support_x['fp_xyz'] = [support_xyz]
        support_x['fp_indices'] = [support_indices]
        support_x['fp_features'] = [support_point_features]

        support_cls_point_features = torch.cat(support_cls_point_features, 0)
        support_x['fp_features'] = [support_cls_point_features]
        support_cls_indices = torch.cat(support_cls_indices, 0)
        support_x['fp_indices'] = [support_cls_indices]
        support_cls_xyz = torch.cat(support_cls_xyz, 0)
        support_x['fp_xyz'] = [support_cls_xyz]

        support_cls_point_xyz = torch.cat(support_cls_point_xyz, 0)
        support_seed_indices = torch.cat(support_seed_indices, 0)

        batch_list = cls_gather(support_x, support_semantic_masks, support_instance_masks, support_cls_point_xyz, support_seed_indices, class_names=self.class_names)
        cls_prototypes = cls_prototype_support(batch_list, batch_size, num=self.K_shot, way=self.N_way, compen_context=self.compen_context, few_shot_class=self.few_shot_class)

        return prototypes, cls_prototypes

def check_path(pts_folder):
    if not os.path.exists(pts_folder):
        os.makedirs(pts_folder)



