# ------------------------------------------------------------------------
# Train and eval functions used in main.py
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import math
import os
import sys
from typing import Iterable
import cv2
import numpy as np
import json
import copy
import requests
import traceback

import glob
import time

import torch
import util.misc as utils
from util.misc import NestedTensor
from vad_utils.evaluate_ava import STDetectionEvaluater
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from PIL import Image, ImageDraw


from scipy.optimize import linear_sum_assignment

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 4000

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)

    samples, targets = prefetcher.next()

    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):

 
        outputs, loss_dict = model(samples, targets, criterion, train=True)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced = loss_dict
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
    
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        
        optimizer.step()
   
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        samples, targets = prefetcher.next()

    metrics_data = json.dumps({
        '@epoch': epoch,
        '@step': epoch, # actually epoch
        '@loss': metric_logger.loss.value,
        '@class_error': metric_logger.class_error.value,
        '@lr': metric_logger.lr.value,
        '@grad_norm': metric_logger.grad_norm.value,
    })

    try:
        # Report JSON data to the NSML metric API server with a simple HTTP POST request.
        requests.post(os.environ['NSML_METRIC_API'], data=metrics_data)
    except requests.exceptions.RequestException:
        # Sometimes, the HTTP request might fail, but the training process should not be stopped.
        traceback.print_exc()    

    torch.cuda.empty_cache()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}







@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args):

    num_frames = args.num_frames 
    eval_types = args.eval_types

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    coco_iou_types = [k for k in ['bbox', 'segm'] if k in postprocessors.keys()]

    coco_evaluator = None
    if 'coco' in eval_types:
        coco_evaluator = CocoEvaluator(base_ds['coco'], coco_iou_types)
        # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    

    for samples, targets in metric_logger.log_every(data_loader, 1000, header):

        samples = samples.to(device)
        all_outputs, loss_dict = model(samples, targets, criterion, train=False)

        #### reduce losses over all GPUs for logging purposes ####
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        weight_dict = criterion.weight_dict

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                            **loss_dict_reduced_scaled,
                            **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        #### reduce losses over all GPUs for logging purposes ####
        ##### single clip input ######
    
        if all_outputs['pred_boxes'].dim() == 3:
            all_outputs['pred_boxes'] = all_outputs['pred_boxes'].unsqueeze(2)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = [{} for i in range(len(targets))]
        if 'bbox' in postprocessors.keys():
            results = postprocessors['bbox'](all_outputs, orig_target_sizes, num_frames=num_frames)
            #   scores: [num_ins]
            #   labels: [num_ins]
            #   boxes: [num_ins, num_frames, 4]

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, all_outputs, orig_target_sizes, target_sizes)
        
        res_img = {}

        # evaluate results
        if 'coco' in eval_types:
            for target, output in zip(targets, results):
                for fid in range(num_frames):
                    res_img[target['image_id'][fid].item()] = {}
                    for k, v in output.items():
                        if k == 'masks':
                            res_img[target['image_id'][fid].item()][k] = v[:,fid].unsqueeze(1)
                        elif k == 'boxes':
                            res_img[target['image_id'][fid].item()][k] = v[:,fid]
                        else:
                            res_img[target['image_id'][fid].item()][k] = v



        if coco_evaluator is not None:
            coco_evaluator.update(res_img)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    return stats, coco_evaluator


@torch.no_grad()
def validate_ava_detection(args, model, criterion, postprocessors, data_loader, epoch, device):

    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    class_err = utils.AverageMeter()
    losses_box = utils.AverageMeter()
    losses_giou = utils.AverageMeter()
    losses_ce = utils.AverageMeter()
    losses_avg = utils.AverageMeter()
    losses_ce_b = utils.AverageMeter()

    end = time.time()
    model.eval()
    criterion.eval()

    buff_output = []
    buff_anno = []
    buff_id = []
    # buff_binary = []

    buff_GT_label = []
    buff_GT_anno = []
    buff_GT_id = []

    if utils.get_local_rank() == 0:
        tmp_path = "{}/{}".format(args.output_dir, "results")
        if not os.path.exists(tmp_path): os.makedirs(tmp_path)
        tmp_dirs_ = glob.glob("{}/{}/*.txt".format(args.output_dir, "results"))
        for tmp_dir in tmp_dirs_:
            os.remove(tmp_dir)
            print("remove {}".format(tmp_dir))
        print("all tmp files removed")

    for idx, data in enumerate(data_loader):
        data_time.update(time.time() - end)
        # for samples, targets in metric_logger.log_every(data_loader, print_freq, epoch, ddp_params, writer, header):
        # device = "cuda:" + str(utils.get_local_rank())
        samples = data[0]
        targets = data[1]

        samples = samples.to(device)

        batch_id = [t["image_id"] for t in targets]

        for t in targets:
            del t["image_id"]

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # outputs = model(samples)
        outputs, loss_dict = model(samples, targets, criterion, train=False)
        # import pdb; pdb.set_trace()
        # loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict

        orig_target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        scores, boxes = postprocessors['bbox'](outputs, orig_target_sizes)
        for bidx in range(scores.shape[0]):
            frame_id = batch_id[bidx][0]
            key_pos = batch_id[bidx][1]

            buff_output.append(scores[bidx])
            buff_anno.append(boxes[bidx])

            for l in range(args.num_queries):
                buff_id.extend([frame_id])

            # raw_idx = (targets[bidx]["raw_boxes"][:, 1] == key_pos).nonzero().squeeze()
            raw_idx = torch.nonzero((targets[bidx]["raw_boxes"][:, 1] == key_pos), as_tuple=False).squeeze()

            val_label = targets[bidx]["labels"][raw_idx]
            val_label = val_label.reshape(-1, val_label.shape[-1])
            raw_boxes = targets[bidx]["raw_boxes"][raw_idx]
            raw_boxes = raw_boxes.reshape(-1, raw_boxes.shape[-1])
            # print('raw_boxes',raw_boxes.shape)

            buff_GT_label.append(val_label.detach().cpu().numpy())
            buff_GT_anno.append(raw_boxes.detach().cpu().numpy())


            # print(buff_anno, buff_GT_anno)

            img_id_item = [batch_id[int(raw_boxes[x, 0] - targets[0]["raw_boxes"][0, 0])][0] for x in
                           range(len(raw_boxes))]

            buff_GT_id.extend(img_id_item)

        batch_time.update(time.time() - end)
        end = time.time()
        if utils.get_local_rank() == 0:
            if idx % args.log_display_freq == 0 or idx == len(data_loader) - 1:
                print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(data_loader))
                print(print_string)
                print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val,
                    batch_time=batch_time.val)
                print(print_string)

            # reduce losses over all GPUs for logging purposes
            # loss_dict_reduced = utils.reduce_dict(loss_dict)

            # reduce on single GPU
            loss_dict_reduced = loss_dict
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                          for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            losses_avg.update(loss_value, len(targets))
            losses_box.update(loss_dict_reduced['loss_bbox'].item(), len(targets))
            losses_giou.update(loss_dict_reduced['loss_giou'].item(), len(targets))
            losses_ce.update(loss_dict_reduced['loss_ce'].item(), len(targets))
            class_err.update(loss_dict_reduced['class_error'], len(targets))

            # if cfg.CONFIG.MATCHER.BNY_LOSS:
            #     losses_ce_b.update(loss_dict_reduced['loss_ce_b'].item(), len(targets))

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping eval".format(loss_value))
                print(loss_dict_reduced)
                exit(1)
            if idx % args.log_display_freq == 0:
                print_string = 'class_error: {class_error:.3f}, loss: {loss:.3f}, loss_bbox: {loss_bbox:.3f}, loss_giou: {loss_giou:.3f}, loss_ce: {loss_ce:.3f}'.format(
                    class_error=class_err.avg,
                    loss=losses_avg.avg,
                    loss_bbox=losses_box.avg,
                    loss_giou=losses_giou.avg,
                    loss_ce=losses_ce.avg,
                    # cardinality_error=loss_dict_reduced['cardinality_error']
                )
                print(print_string)
            # print("len(buff_id): ", len(buff_id))
            # print("len(buff_binary): ", len(np.concatenate(buff_binary, axis=0)))
            # print("len(buff_anno): ", len(np.concatenate(buff_anno, axis=0)))
            # print("len(buff_output): ", len(np.concatenate(buff_output, axis=0)))
            # print("len(buff_GT_id): ", len(buff_GT_id))
            # print("len(buff_GT_label): ", len(np.concatenate(buff_GT_label, axis=0)))
            # print("len(buff_GT_anno): ", len(np.concatenate(buff_GT_anno, axis=0)))

    # if utils.get_local_rank() == 0:
    #     writer.add_scalar('val/class_error', class_err.avg, epoch)
    #     writer.add_scalar('val/total_loss', losses_avg.avg, epoch)
    #     writer.add_scalar('val/loss_bbox', losses_box.avg, epoch)
    #     writer.add_scalar('val/loss_giou', losses_giou.avg, epoch)
    #     writer.add_scalar('val/loss_ce', losses_ce.avg, epoch)

    buff_output = np.concatenate(buff_output, axis=0)
    buff_anno = np.concatenate(buff_anno, axis=0)
    # buff_binary = np.concatenate(buff_binary, axis=0)
    buff_GT_label = np.concatenate(buff_GT_label, axis=0)
    buff_GT_anno = np.concatenate(buff_GT_anno, axis=0)
    print(buff_output.shape, buff_anno.shape, len(buff_id), buff_GT_anno.shape, buff_GT_label.shape, len(buff_GT_id))
    
    tmp_path = '{}/{}/{}.txt'
    with open(tmp_path.format(args.output_dir, "results", utils.get_local_rank()), 'w') as f:
        for x in range(len(buff_id)):
            data = np.concatenate([buff_anno[x], buff_output[x]])
            f.write("{} {}\n".format(buff_id[x], data.tolist()))
    tmp_GT_path = '{}/{}/GT_{}.txt'
    with open(tmp_path.format(args.output_dir, "results", utils.get_local_rank()), 'w') as f:
        for x in range(len(buff_GT_id)):
            data = np.concatenate([buff_GT_anno[x], buff_GT_label[x]])
            f.write("{} {}\n".format(buff_GT_id[x], data.tolist()))

    # write files and align all workers
    torch.distributed.barrier()
    # aggregate files
    Map_ = 0
    # aggregate files
    if utils.get_local_rank() == 0:
        # read results
        evaluater = STDetectionEvaluater(args.output_dir, tiou_thresholds=[0.5], class_num=args.ava_num_classes)
        file_path_lst = [tmp_GT_path.format(args.output_dir, "results", x) for x in range(utils.get_world_size)]
        evaluater.load_GT_from_path(file_path_lst)
        file_path_lst = [tmp_path.format(args.output_dir, "results", x) for x in range(utils.get_world_size)]
        evaluater.load_detection_from_path(file_path_lst)
        mAP, metrics = evaluater.evaluate()
        print(metrics)
        print_string = 'mAP: {mAP:.5f}'.format(mAP=mAP[0])
        print(print_string)
        print(mAP)
        # writer.add_scalar('val/val_mAP_epoch', mAP[0], epoch)
        Map_ = mAP[0]
    metrics_data = json.dumps({
        '@epoch': epoch,
        '@step': epoch, # actually epoch
        'val_class_error': class_err.avg,
        'val_loss': losses_avg.avg,
        'val_loss_giou': losses_giou.avg,
        'val_loss_ce': losses_ce.avg,
        'val_mAP': Map_        
    })
    try:
        # Report JSON data to the NSML metric API server with a simple HTTP POST request.
        requests.post(os.environ['NSML_METRIC_API'], data=metrics_data)
    except requests.exceptions.RequestException:
        # Sometimes, the HTTP request might fail, but the training process should not be stopped.
        traceback.print_exc()    

    torch.distributed.barrier()
    time.sleep(30)
    return Map_
