# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import force_fp32

from mmdet.core import bbox_overlaps, multi_apply, reduce_mean
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import YOLOXHead
from mmcv.ops.nms import batched_nms

@HEADS.register_module()
class XRAYHead(YOLOXHead):
    """Localization distillation Head. (Short description)

    It utilizes the learned bbox distributions to transfer the localization
    dark knowledge from teacher to student. Original paper: `Localization
    Distillation for Object Detection. <https://arxiv.org/abs/2102.12252>`_

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss_ld (dict): Config of Localization Distillation Loss (LD),
            T is the temperature for distillation.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_kl=dict(
                     type='KnowledgeDistillationKLDivLoss',
                     loss_weight=0.25,
                     T=10),
                 **kwargs):

        super(XRAYHead, self).__init__(num_classes, in_channels, **kwargs)
        #self.d_list = [5, 32, 38, 42, 43, 74, 55, 82]
        self.loss_bbox_kl = build_loss(loss_kl)
        self.loss_cls_kl = build_loss(loss_kl)
        self.loss_objetness_kl = build_loss(loss_kl)
        #self.weight = nn.Parameter(torch.tensor((1/3,1/3,1/3)))

    def _bboxes_nms(self, cls_scores, bboxes, score_factor, cfg):
        threshold = 0.01
        unk_mask = cls_scores.max(dim=1).values < threshold
        cls_scores[:,83] = torch.clamp(cls_scores[:,83] + unk_mask, max=1)

        max_scores, labels = torch.max(cls_scores, 1)
        valid_mask = score_factor * max_scores >= cfg.score_thr

        bboxes = bboxes[valid_mask]
        scores = max_scores[valid_mask] * score_factor[valid_mask]
        labels = labels[valid_mask]

        if labels.numel() == 0:
            return bboxes, labels
        else:
            dets, keep = batched_nms(bboxes, scores, labels, cfg.nms)
            return dets, labels[keep]

    def forward_train(self,
                      x,
                      out_teacher,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple[dict, list]: The loss components and proposals of each image.

            - losses (dict[str, Tensor]): A dictionary of loss components.
            - proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, out_teacher, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             objectnesses,
             gt_bboxes,
             gt_labels,
             out_teacher,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """
        # gt_mask = []
        # for gt_label in gt_labels:
        #     tmp_tensor = torch.zeros_like(gt_label)
        #     for d_id in self.d_list:
        #         tmp_tensor += (gt_label==d_id).long()
        #     gt_mask.append(tmp_tensor.bool().clone().detach())

        soft_clses, soft_bboxes, soft_objs = out_teacher
        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels)
            for cls_pred in cls_scores
        ]
        flatten_soft_cls = [
            soft_cls.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels)
            for soft_cls in soft_clses
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_soft_bboxes = [
            soft_bbox.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for soft_bbox in soft_bboxes
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]
        flatten_soft_objs = [
            soft_obj.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for soft_obj in soft_objs
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)

        flatten_soft_cls = torch.cat(flatten_soft_cls, dim=1)
        flatten_soft_bboxes = torch.cat(flatten_soft_bboxes, dim=1)
        flatten_soft_objs = torch.cat(flatten_soft_objs, dim=1)

        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)
        flatten_soft_bboxes_ = self._bbox_decode(flatten_priors, flatten_soft_bboxes)

        (pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets,
         num_fg_imgs) = multi_apply(
             self._get_target_single, flatten_cls_preds.detach(),
             flatten_objectness.detach(),
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_bboxes.detach(), gt_bboxes, gt_labels)

        # The experimental results show that ‘reduce_mean’ can improve
        # performance on the COCO dataset.
        num_pos = torch.tensor(
            sum(num_fg_imgs),
            dtype=torch.float,
            device=flatten_cls_preds.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        loss_bbox = self.loss_bbox(
            flatten_bboxes.view(-1, 4)[pos_masks],
            bbox_targets) / num_total_samples
        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1),
                                 obj_targets) / num_total_samples
        loss_cls = self.loss_cls(
            flatten_cls_preds.view(-1, self.num_classes)[pos_masks],
            cls_targets) / num_total_samples

        loss_bbox_soft = self.loss_bbox_kl(
            flatten_bboxes.view(-1, 4)[pos_masks],
            flatten_soft_bboxes_.view(-1, 4)[pos_masks]) / num_total_samples
        loss_obj_soft = self.loss_objetness_kl(
            flatten_objectness.view(-1, 1),
            flatten_soft_objs.view(-1, 1)) / num_total_samples
        loss_cls_soft = self.loss_cls_kl(
            flatten_cls_preds.view(-1, self.num_classes)[pos_masks],
            flatten_soft_cls.view(-1, self.num_classes)[pos_masks]) / num_total_samples
        
        loss_kl = 0.3*loss_obj_soft + 0.4*loss_bbox_soft + 0.3*loss_cls_soft

        loss_dict = dict(
            loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj, loss_kl=loss_kl)
        
        if self.use_l1:
            loss_l1 = self.loss_l1(
                flatten_bbox_preds.view(-1, 4)[pos_masks],
                l1_targets) / num_total_samples
            loss_dict.update(loss_l1=loss_l1)

        return loss_dict
