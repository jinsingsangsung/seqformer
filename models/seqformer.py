# ------------------------------------------------------------------------
# SeqFormer model and criterion classes.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, accuracy_sigmoid, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (SeqFormer, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
                           
from .deformable_transformer import build_deforamble_transformer
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_frames, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
        """
        super().__init__()
        self.num_frames = num_frames
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        

    def forward(self, samples: NestedTensor, targets, criterion, train=True):
        """??The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [num_frames x 3 x H x W]
               - samples.mask: a binary mask of shape [num_frames x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        # print("samples.tensors.shape: ", samples.tensors.shape)
        # print("samples.mask.shape: ", samples.mask.shape)
        # print("len(features): ", len(features))           
        features, pos = self.backbone(samples)
        # print("mask.shape: ", features[2].mask.shape)
        # print("features.shape: ", features[2].tensors.shape)
        # print("len(features): ", len(features))
        srcs = []
        masks = []
        poses = []
        for l, feat in enumerate(features[1:]):
            # src: [nf*N, _C, Hi, Wi],
            # mask: [nf*N, Hi, Wi],
            # pos: [nf*N, C, H_p, W_p]
            src, mask = feat.decompose() 
            src_proj_l = self.input_proj[l](src)    # src_proj_l: [nf*N, C, Hi, Wi]

            # src_proj_l -> [nf, N, C, Hi, Wi]
            n,c,h,w = src_proj_l.shape
            src_proj_l = src_proj_l.reshape(n//self.num_frames, self.num_frames, c, h, w)
            # src_proj_l = src_proj_l.reshape(n//self.num_frames, self.num_frames, c, h, w).permute(1,0,2,3,4)
            # mask -> [nf, N, Hi, Wi]
            mask = mask.reshape(n//self.num_frames, self.num_frames, h, w)
            # mask = mask.reshape(n//self.num_frames, self.num_frames, h, w).permute(1,0,2,3)
            # pos -> [nf, N, Hi, Wi]
            np, cp, hp, wp = pos[l+1].shape
            pos_l = pos[l+1].reshape(np//self.num_frames, self.num_frames, cp, hp, wp)
            # pos_l = pos[l+1].reshape(np//self.num_frames, self.num_frames, cp, hp, wp).permute(1,0,2,3,4)
            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos_l)
            # for n_f in range(self.num_frames):
            #     srcs.append(src_proj_l[n_f])
            #     masks.append(mask[n_f])
            #     poses.append(pos_l[n_f])
            #     assert mask is not None

        if self.num_feature_levels > (len(features) - 1): # the last feature map is a projection of the previous map
            _len_srcs = len(features) - 1
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask    # [nf*N, H, W]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                mask = mask.unsqueeze(1).repeat(1,samples.tensors.shape[2],1,1) # for ava dataset
                mask = mask.permute(1,0,2,3).flatten(0,1)
                # print("src.shape: ", src.shape)
                # print("mask.shape: ", mask.shape)
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                # src -> [nf, N, C, H, W]
                n, c, h, w = src.shape
                src = src.reshape(n//self.num_frames, self.num_frames, c, h, w)
                # src = src.reshape(n//self.num_frames, self.num_frames, c, h, w).permute(1,0,2,3,4)
                mask = mask.reshape(n//self.num_frames, self.num_frames, h, w)
                # mask = mask.reshape(n//self.num_frames, self.num_frames, h, w).permute(1,0,2,3)
                np, cp, hp, wp = pos_l.shape
                pos_l = pos_l.reshape(np//self.num_frames, self.num_frames, cp, hp, wp)
                # pos_l = pos_l.reshape(np//self.num_frames, self.num_frames, cp, hp, wp).permute(1,0,2,3,4)
                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)
                # for n_f in range(self.num_frames):
                #     srcs.append(src[n_f])
                #     masks.append(mask[n_f])
                #     poses.append(pos_l[n_f])

        query_embeds = None
        query_embeds = self.query_embed.weight
        hs, hs_box, memory, init_reference, inter_references, inter_samples, enc_outputs_class, valid_ratios = self.transformer(srcs, masks, poses, query_embeds)
        valid_ratios = valid_ratios[: 0]

        outputs = {}
        outputs_classes = []
        outputs_coords = []
        indices_list = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs_box[lvl])
            # tmp = self.bbox_embed[lvl](hs[lvl])
            # tmp.shape: bs, 32, 300, 4
            # reference: bs, 32, 300, 4
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            #TODO: temporally aggregate outputs_coord to output a single frame
            # Now, naively average them
            tmp = tmp.mean(dim=1)
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_layer = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
            try:
                indices = criterion.matcher(outputs_layer, targets, self.num_frames, valid_ratios)
            except:
                indices = criterion.matcher(outputs_layer, targets)
            indices_list.append(indices)

            reference_points, num_insts = [], []
            for i, indice in enumerate(indices):
                pred_i, tgt_j = indice
                num_insts.append(len(pred_i))

                # This is the image size after data augmentation (so as the gt boxes & masks)
                
                orig_h, orig_w = targets[i]['size']
                scale_f = torch.stack([orig_w, orig_h], dim=0)
                
                ref_cur_f = reference[i].sigmoid()
                ref_cur_f = ref_cur_f[..., :2]
                ref_cur_f = ref_cur_f * scale_f[None,None, :] 
                
                reference_points.append(ref_cur_f[:,pred_i].unsqueeze(0))
            reference_points = torch.cat(reference_points, dim=2)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        # print("outputs_class.shape: ", outputs_class.shape)
        # print("outputs_coord.shape: ", outputs_coord.shape)

        outputs["pred_logits"] = outputs_class[-1]
        outputs["pred_boxes"] = outputs_coord[-1]
        
        if self.aux_loss:
            outputs['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        try:
            loss_dict = criterion(outputs, targets, indices_list, valid_ratios)
        except:
            loss_dict = criterion(outputs, targets)

        if not train:
            outputs['reference_points'] = inter_references[-2, :, :, :, :2]

            bs, num_queries = outputs_class.size(1), outputs_class.size(3)
            num_insts = [num_queries for i in range(bs)]

            reference_points = []
            for i, target in enumerate(targets):
                orig_h, orig_w = target['size']
                scale_f = torch.stack([orig_w, orig_h], dim=0)
                ref_cur_f = outputs['reference_points'][i] * scale_f[None,None, :] 
                reference_points.append(ref_cur_f.unsqueeze(0))
            # import pdb;pdb.set_trace()
            # reference_points: [1, N * num_queries, 2]
            # mask_head_params: [1, N * num_queries, num_params]
           
            reference_points = torch.cat(reference_points, dim=2)

            # outputs['pred_masks']: [bs, num_queries, num_frames, H/4, W/4]
            
            outputs['pred_boxes'] = outputs['pred_boxes'][:,0] 
            outputs['reference_points'] = outputs['reference_points'][:,0]

        return outputs, loss_dict

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class SetCriterionAVA(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, weight, num_classes, num_queries, matcher, weight_dict, eos_coef, losses, data_file,
                 evaluation=False):
        """ Create the criterion.
        Parameters
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight = weight
        self.evaluation = evaluation
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.data_file = data_file
        empty_weight = torch.ones(3)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        # src_logits_b = outputs['pred_logits_b']
        # target_classes_b = torch.full(src_logits_b.shape[:2], 2,
        #                               dtype=torch.int64, device=src_logits.device)
        # target_classes_b[idx] = 1

        # loss_ce_b = F.cross_entropy(src_logits_b.transpose(1, 2), target_classes_b, self.empty_weight.to(src_logits.device))
        src_logits_sig = src_logits.sigmoid()

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        target_classes = torch.full(src_logits.shape, 0,
                                    dtype=torch.float32, device=src_logits.device)
        # rebalance way 1:
        weights = torch.full(src_logits.shape[:2], 1,
                             dtype=torch.float32, device=src_logits.device)
        weights[idx] = self.weight
        #
        weights = weights.view(weights.shape[0], weights.shape[1], 1)  # [:,:,None]
        target_classes[idx] = target_classes_o
        if self.evaluation:
            loss_ce = F.binary_cross_entropy(src_logits_sig, target_classes)
        else:
            loss_ce = F.binary_cross_entropy(src_logits_sig, target_classes, weight=weights)

        losses = {'loss_ce': loss_ce}
        # losses['loss_ce_b'] = loss_ce_b
        if log:
            # docs this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy_sigmoid(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_boxes = target_boxes[:, 1:]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # docs use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        # _, sidx = self._get_src_permutation_idx(indices)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs_without_aux, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, mask_out_stride=4, num_frames=1):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.mask_out_stride = mask_out_stride
        self.num_frames = num_frames
        self.valid_ratios = None

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        
  
        valid_ratios = self.valid_ratios

        src_boxes = outputs['pred_boxes'].permute(0,2,1,3)[idx]  # [selected_inst, nf, 4]
        num_insts,nf = src_boxes.shape[:2]
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
       
        tgt_bbox = tgt_bbox.reshape(num_insts,nf,4)
        sizes = [len(v["labels"]) for v in targets]
        target_boxes = list(tgt_bbox.split(sizes,dim=0))


        target_boxes = torch.cat([t[i] for t, (_, i) in zip(target_boxes, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes.flatten(1,2), target_boxes.flatten(1,2), reduction='none')
        loss_giou = 0
        for i in range(nf):
            loss_giou = loss_giou + 1 - torch.diag(box_ops.generalized_box_iou(
                                        box_ops.box_cxcywh_to_xyxy(src_boxes[:,i]),
                                        box_ops.box_cxcywh_to_xyxy(target_boxes[:,i])))
        loss_bbox = loss_bbox/nf
        loss_giou = loss_giou/nf

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        

        return losses



    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
  

        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]
        bs = len(targets)
        # src_masks: bs x [1, num_inst, num_frames, H/4, W/4] or [bs, num_inst, num_frames, H/4, W/4]
        # src_masks: [num_insts, num_frames, H/4, M/4]
        # src_masks = src_masks[src_idx]
        if type(src_masks) == list:
            src_masks = torch.cat(src_masks, dim=1)[0]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets], 
                                                             size_divisibility=32,
                                                             split=False).decompose()
        target_masks = target_masks.to(src_masks)
   
        # downsample ground truth masks with ratio mask_out_stride
        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride]
        assert target_masks.size(2) * self.mask_out_stride == im_h
        assert target_masks.size(3) * self.mask_out_stride == im_w
        num_frames = src_masks.shape[1]

        # # upsample predictions to the target size
        # src_masks = interpolate(src_masks, size=target_masks.shape[-2:],
        #                         mode="bilinear", align_corners=False)
        src_masks = src_masks.flatten(1)
        target_masks = target_masks.reshape(bs, -1, num_frames, target_masks.shape[-2], target_masks.shape[-1])
        target_masks = target_masks[tgt_idx].flatten(1)
        # src_masks/target_masks: [n_targets, num_frames* H * W]

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses



    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, indices_list, valid_ratios):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        self.valid_ratios = valid_ratios
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices_list[-1], num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                indices = indices_list[i]
                for loss in self.losses:
                    # if loss == 'masks':
                    #     # Intermediate masks losses are too costly to compute, we ignore them.
                    #     continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, num_frames=1):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
            num_frames: output frame num
        """
        # output single / multi frames
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        # out_logits: [N, num_queries, num_classes]
        # out_bbox: [N, num_queries, num_frames, 4]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        bs, num_q = out_logits.shape[:2]

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).unsqueeze(-1).repeat(1,1,boxes.shape[-2],boxes.shape[-1]))
        
        # samples = torch.gather(out_samples, 1, topk_boxes.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, out_samples.shape[2], 2))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, None, :]

        # samples = samples * scale_fct[:, None, None, :2]
        
        # all_scores = torch.cat([scores.unsqueeze(0) for scores in all_scores], dim=0).permute(1,2,0)
        # all_labels = torch.cat([labels.unsqueeze(0) for labels in all_labels], dim=0).permute(1,2,0)
        # all_boxes = torch.cat([boxes.unsqueeze(0) for boxes in all_boxes], dim=0).permute(1,2,0,3)

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        #   scores: [num_ins]
        #   labels: [num_ins]
        #   boxes: [num_ins, num_frames, 4]
        # import pdb;pdb.set_trace()
        return results

class PostProcessAVA(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """

        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        # print("out_logits shape: ", out_logits.shape)
        # print("target_sizes shape: ", target_sizes.shape)
        ## TODO: need to uncomment below
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2


        # prob = out_logits.sigmoid() * out_logits_b.softmax(-1)[:,:,1:2]


        # prob_binary = out_logits_b.softmax(-1)[:, :, 1:2]
        # prob_bbox = (prob_binary > 0.8).float() * prob_binary
        prob = out_logits.sigmoid() #* prob_bbox

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        scores = prob.detach().cpu().numpy()
        boxes = boxes.detach().cpu().numpy()
        # output_b = out_logits_b.softmax(-1).detach().cpu().numpy()[..., 1:2]

        return scores, boxes #, output_b


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):    
    if args.dataset_file == 'ava':
        num_classes = 80
    elif args.dataset_file == 'coco':
        num_classes = 91
    else:
        num_classes = 20
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    if args.dataset_file == 'YoutubeVIS' or args.dataset_file == 'jointcoco' or args.dataset_file == 'Seq_coco':
        num_classes = 42

    device = torch.device(args.device)

    if 'swin' in args.backbone:
        from .swin_transformer import build_swin_backbone
        backbone = build_swin_backbone(args) 
    else:
        backbone = build_backbone(args)


    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_frames=args.num_frames,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
    )
    if args.masks:
        model = SeqFormer(model, freeze_detr=False, rel_coord=args.rel_coord)
    
    matcher = build_matcher(args)
    
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    # losses = ['labels', 'cardinality']
    if args.masks:
        losses += ["masks"]

    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    if "ava" in args.dataset_file:
        criterion = SetCriterionAVA(weight=10,
                                    num_classes=num_classes,
                                    num_queries=args.num_queries,
                                    matcher=matcher,
                                    weight_dict=weight_dict,
                                    eos_coef=0.1,
                                    losses=losses,
                                    data_file="ava",
                                    evaluation=args.eval_only)
    else:
        criterion = SetCriterion(num_classes, matcher, weight_dict, losses, 
                                mask_out_stride=args.mask_out_stride,
                                focal_alpha=args.focal_alpha,
                                num_frames = args.num_frames)
    criterion.to(device)

    postprocessors = {'bbox': PostProcessAVA() if "ava" in args.dataset_file else PostProcess()}
    # postprocessors = {}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
       

    return model, criterion, postprocessors



