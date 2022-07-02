import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.core import add_prefix
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.losses import accuracy
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.ops import resize



@SEGMENTORS.register_module()
class HierarchicalSegmentor(EncoderDecoder):
    """
    Hierarchical three level decoder heads with shared encoder.
    """

    def __init__(
            self,
            backbone,
            levels_class_mapping,
            heads_hierarchy,
            decode_head,
            neck=None,
            auxiliary_head=None,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
            init_cfg=None,
    ):
        self.heads_hierarchy = heads_hierarchy
        self.levels_class_mapping = levels_class_mapping
        self.heads_hierarchy_flat = []
        self.level_1_heads = []
        # level_2_heads and level_3_heads maintains the order of heads
        # 0 index level 3 heads are subclassifiers of  0th index level 2 heads ... and ...so on
        self.level_2_heads = []
        self.level_3_heads = []

        for indx, each_head in enumerate(self.heads_hierarchy):
            if indx == 0:
                self.level_1_heads.append(each_head[0])
                self.heads_hierarchy_flat.append(each_head[0])
                continue
            else:
                self.level_2_heads.append(each_head[0])

            self.heads_hierarchy_flat.append(each_head[0])
            if len(each_head) == 2:
                self.heads_hierarchy_flat.extend(each_head[1])
                self.level_3_heads.append(each_head[1])
            else:
                raise NotImplementedError

        super(HierarchicalSegmentor, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )

        # Todo: verify class_hierarchy_heads order of heads and its classes are preserved.
        if list(self.levels_class_mapping.keys()) != self.heads_hierarchy_flat:
            raise ValueError("Incorrect order of classifier in configuration.")
            # key = random.choice(list(self.levels_class_mapping.keys()))
            # if list(self.levels_class_mapping[key]) !=

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = nn.ModuleList()
        for each_head_name in self.heads_hierarchy_flat:
            self.decode_head.append(builder.build_head(decode_head[each_head_name]))
        self.align_corners = self.decode_head[0].align_corners
        # self.num_classes = self.decode_head[-1].num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            self.auxiliary_head = nn.ModuleList()
            for each_head_name in self.heads_hierarchy_flat:
                self.auxiliary_head.append(builder.build_head(auxiliary_head[each_head_name]))

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
                input=out,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        heads_gt_semantic_seg = self.remapped_gt_semantic_seg(gt_semantic_seg)
        losses = dict()
        if isinstance(self.decode_head, nn.ModuleList):
            for idx, decode_head in enumerate(self.decode_head):
                loss_decode = decode_head.forward_train(
                    x, img_metas, heads_gt_semantic_seg[idx], self.train_cfg)

                losses.update(add_prefix(loss_decode, f'decode_{idx}'))
        else:
            raise NotImplementedError
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logit = self.decode_head[self.current_decode_head_indx].forward_test(x, img_metas, self.test_cfg)
        return seg_logit

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        heads_gt_semantic_seg = self.remapped_gt_semantic_seg(gt_semantic_seg)
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  heads_gt_semantic_seg[idx],
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            raise NotImplementedError

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    def aug_test(self, imgs, img_metas, rescale=True):
        raise NotImplementedError

    def slide_inference(self, img, img_meta, rescale):
        raise NotImplementedError


    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        # seg_preds: prediction from all heads
        seg_preds = []
        for idx, heads in enumerate(self.heads_hierarchy_flat):
            self.current_decode_head_indx = idx
            seg_logit = self.inference(img, img_meta, rescale)
            seg_preds.append(seg_logit.argmax(dim=1))
        self.current_decode_head_indx=None
        # combine seg_preds and replace with original label ids
        seg_pred = self.seg_preds_to_original(seg_preds)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def seg_preds_to_original(self, seg_preds):
        """
        Combine preds and replace super classes by fined classes.
        :param seg_preds:
        :return: seg_pred
        """
        # from PIL import Image
        # import numpy as np
        # import seaborn
        # import colorcet as cc
        # palette = (np.array(seaborn.color_palette(cc.glasbey, 50)) * 255).astype(np.uint8)
        #
        # for i in range(len(seg_preds)):
        #     try:
        #         # org_image = org_image.squeeze().permute(1, 2, 0)
        #         # processed_img = org_image.cpu().detach().numpy()
        #         # img = processed_img
        #         #img_arr = Image.fromarray(img)
        #         #img_arr = Image.fromarray((img * 255).astype(np.uint8))
        #         #img_arr.save(f"/netscratch/gautam/semseg/exp_results/hierarchical_deeplabv3plus_191c/evaluation/heads_output/input_image.png")
        #         img_label = seg_preds[i].squeeze().cpu().detach().numpy()
        #         img_label = img_label.astype(np.uint8)
        #         img_label = palette[img_label]
        #         img_label_arr = Image.fromarray(img_label)
        #         img_label_arr.save(f"/netscratch/gautam/semseg/exp_results/temp/heads_output/{i}.png")
        #     except Exception as e:
        #         e.args+=(type(img_label), img_label.shape)
        #         raise
        seg_pred = torch.zeros_like(seg_preds[0])

        l1_seg_pred = seg_preds[0]
        for l2_indx, (l2_h_name, l3_h_list) in enumerate(zip(self.level_2_heads, self.level_3_heads)):
            indx_l2_h_name = self.heads_hierarchy_flat.index(l2_h_name)
            final_l2_seg_pred = torch.zeros_like(seg_preds[indx_l2_h_name])
            l2_seg_pred = seg_preds[indx_l2_h_name]

            for l3_h_indx, l3_h_name in enumerate(l3_h_list):
                # index of l3 seg_pred in seg_preds
                converted_l3_pred = None
                indx_l3_h_name = self.heads_hierarchy_flat.index(l3_h_name)
                mask = l2_seg_pred == l3_h_indx+1
                org_cls = torch.tensor([[0]] + list(self.levels_class_mapping[l3_h_name].values()), device=mask.device).flatten()

                converted_l3_pred = org_cls[seg_preds[indx_l3_h_name]]
                final_l2_seg_pred[mask] = converted_l3_pred[mask]



            l1_mask = l1_seg_pred == l2_indx+1
            seg_pred[l1_mask] = final_l2_seg_pred[l1_mask]
        return seg_pred

    def remapped_gt_semantic_seg(self, ground_truth_map):
        """
        Map the gt_semantic_seg according to all heads
        :param gt_semantic_seg: shape torch.size([2, 1, 512, 512])
        :return: list of gt_semantic_segs for all heads
        """
        gt_semantic_segs = []

        for each_head in self.heads_hierarchy_flat:
            # torch.zeros :- rest unknown sub_cls_ids will be assigned to zero
            gt_semantic_mapped = torch.zeros_like(ground_truth_map)
            each_head_class_maps = self.levels_class_mapping[each_head]
            for cls_indx, (cls_name, sub_cls_ids) in enumerate(each_head_class_maps.items()):
                for sub_cls_id in sub_cls_ids:
                    # sub_cls_id+1 :- introduced 0 label_id to every head
                    gt_semantic_mapped += torch.where(ground_truth_map == sub_cls_id, 1, 0) * (cls_indx+1)
            gt_semantic_segs.append(gt_semantic_mapped)
        return gt_semantic_segs


