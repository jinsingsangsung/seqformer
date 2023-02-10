import pandas as pd
import cv2
import torch.utils.data as data
from glob import glob
import numpy as np
from util.misc import collate_fn
import torch
import random
from PIL import Image
import torch.nn.functional as F
import datasets.ava_datasets.video_transforms as T
import json


class VideoDataset(data.Dataset):

    def __init__(self, frame_path, video_frame_bbox, frame_keys_list, clip_len, frame_sample_rate,
                 transforms, crop_size=224, resize_size=256, mode="train", class_num=80):
        self.video_frame_bbox = video_frame_bbox
        self.video_frame_list = frame_keys_list
        self.frame_path = frame_path

        self.video_frame_list = self.video_frame_list

        self.crop_size = crop_size
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.class_num = class_num
        self.resize_size = resize_size

        self.index_cnt = 0
        self._transforms = transforms
        self.mode = mode

        print("rescale size: {}, crop size: {}".format(resize_size, crop_size))

    def __getitem__(self, index):

        frame_key = self.video_frame_list[index]
        vid, frame_second = frame_key.split(",")
        timef = int(frame_second) - 900

        start_img = np.max((timef * 30 - self.clip_len // 2 * self.frame_sample_rate, 0))

        imgs, target = self.loadvideo(start_img, vid, frame_key)

        if len(target) == 0 or target['boxes'].shape[0] == 0:
            pass
        else:
            if self._transforms is not None:
                imgs, target = self._transforms(imgs, target)

        while len(target) == 0 or target['boxes'].shape[0] == 0:
            print('resample.')
            self.index_cnt -= 1
            index = np.random.randint(len(self.video_frame_list))
            frame_key = self.video_frame_list[index]
            vid, frame_second = frame_key.split(",")
            timef = int(frame_second) - 900

            start_img = np.max((timef * 30 - self.clip_len // 2 * self.frame_sample_rate, 0))

            imgs, target = self.loadvideo(start_img, vid, frame_key) # ex) frame_key: '_eBah6c5kyA,1024' 

            if len(target)==0 or target['boxes'].shape[0] == 0:
                pass
            else:
                if self._transforms is not None:
                    imgs, target = self._transforms(imgs, target)

        imgs = torch.stack(imgs, dim=0)
        imgs = imgs.permute(1, 0, 2, 3)

        return imgs, target

    def load_annotation(self, sample_id, video_frame_list): # (val 혹은 train의 key frame을 표시해놓은 list)

        num_classes = self.class_num
        boxes, classes = [], []
        target = {}

        first_img = cv2.imread(video_frame_list[0])

        oh = first_img.shape[0]
        ow = first_img.shape[1]
        if oh <= ow:
            nh = self.resize_size
            nw = self.resize_size * (ow / oh)
        else:
            nw = self.resize_size
            nh = self.resize_size * (oh / ow)

        p_t = int(self.clip_len // 2)
        key_pos = p_t
        anno_entity = self.video_frame_bbox[sample_id]

        for i, bbox in enumerate(anno_entity["bboxes"]):
            label_tmp = np.zeros((num_classes, ))
            acts_p = anno_entity["acts"][i]
            for l in acts_p:
                label_tmp[l] = 1

            if np.sum(label_tmp) == 0: continue
            p_x = np.int_(bbox[0] * nw)
            p_y = np.int_(bbox[1] * nh)
            p_w = np.int_(bbox[2] * nw)
            p_h = np.int_(bbox[3] * nh)

            boxes.append([p_t, p_x, p_y, p_w, p_h])
            classes.append(label_tmp)

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 5)
        boxes[:, 1::3].clamp_(min=0, max=int(nw))
        boxes[:, 2::3].clamp_(min=0, max=nh)

        if boxes.shape[0]:
            raw_boxes = F.pad(boxes, (1, 0, 0, 0), value=self.index_cnt)
        else:
            raw_boxes = boxes
        classes = np.array(classes)
        classes = torch.as_tensor(classes, dtype=torch.float32).reshape(-1, num_classes)

        target["image_id"] = [str(sample_id).replace(",", "_"), key_pos]
        target['boxes'] = boxes
        target['raw_boxes'] = raw_boxes
        target["labels"] = classes
        target["orig_size"] = torch.as_tensor([int(nh), int(nw)])
        target["size"] = torch.as_tensor([int(nh), int(nw)])
        self.index_cnt = self.index_cnt + 1

        return target

    def loadvideo(self, start_img, vid, frame_key):
        video_frame_path = self.frame_path.format(vid)
        video_frame_list = sorted(glob(video_frame_path + '/*.jpg'))

        if len(video_frame_list) == 0:
            print("path doesnt exist", video_frame_path)
            return [], []

        target = self.load_annotation(frame_key, video_frame_list)

        start_img = np.max(start_img, 0)
        end_img = start_img + self.clip_len * self.frame_sample_rate
        indx_img = list(np.clip(range(start_img, end_img, self.frame_sample_rate), 0, len(video_frame_list) - 1))
        buffer = []
        for frame_idx in indx_img:
            tmp = Image.open(video_frame_list[frame_idx])
            tmp = tmp.resize((target['orig_size'][1], target['orig_size'][0]))
            buffer.append(tmp)

        return buffer, target

    def __len__(self):
        return len(self.video_frame_list)


def make_transforms(image_set, args):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print("transform image crop: {}".format(args.ava_img_size))
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSizeCrop_Custom(args.ava_img_size),
            T.ColorJitter(),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.Resize_Custom(args.ava_img_size),
            normalize,
        ])

    if image_set == 'visual':
        return T.Compose([
            T.Resize_Custom(args.ava_img_size),
            normalize,
        ])
    raise ValueError(f'unknown {image_set}')

def obtain_generated_bboxes_training(input_csv="../assets/ava_{}_v2.2.csv",
                                     eval_only=False,
                                     frame_root="/mnt/video-nfs5/datasets/ava/frames",
                                     mode="train"):
    import os
    from glob import glob
    used=[]
    input_csv = input_csv.format(mode)
    # frame_root = frame_root.format(mode)

    video_frame_bbox = {}
    gt_sheet = pd.read_csv(input_csv, header=None)
    count = 0
    frame_keys_list = set()
    missed_videos = set()

    for index, row in gt_sheet.iterrows():
        vid = row[0]
        if not os.path.isdir(frame_root + "/" + vid + ""):
            missed_videos.add(vid)
            continue

        frame_second = row[1]

        bbox_conf = row[7]
        if bbox_conf < 0.8:
            continue
        frame_key = "{},{}".format(vid, str(frame_second).zfill(4))

        frame_keys_list.add(frame_key)

        count += 1
        bbox = [row[2], row[3], row[4], row[5]]
        gt = int(row[6])

        if frame_key not in video_frame_bbox.keys():
            video_frame_bbox[frame_key] = {}
            video_frame_bbox[frame_key]["bboxes"] = [bbox]
            video_frame_bbox[frame_key]["acts"] = [[gt - 1]]
        else:
            if bbox not in video_frame_bbox[frame_key]["bboxes"]:
                video_frame_bbox[frame_key]["bboxes"].append(bbox)
                video_frame_bbox[frame_key]["acts"].append([gt - 1])
            else:
                idx = video_frame_bbox[frame_key]["bboxes"].index(bbox)
                video_frame_bbox[frame_key]["acts"][idx].append(gt - 1)

    print("missed vids:")
    print(missed_videos)
    return video_frame_bbox, list(frame_keys_list)


def make_image_key(video_id, timestamp):
    """Returns a unique identifier for a video id & timestamp."""
    return "%s,%04d" % (video_id, int(timestamp))


def build_dataloader(image_set, args):
    if image_set == 'train':
        train_bbox_json = json.load(open(args.ava_anno_path.format("train")))
        train_video_frame_bbox, train_frame_keys_list = train_bbox_json["video_frame_bbox"], train_bbox_json["frame_keys_list"]

        train_dataset = VideoDataset(args.ava_path,
                                    train_video_frame_bbox,
                                    train_frame_keys_list,
                                    transforms=make_transforms("train", args),
                                    frame_sample_rate=args.ava_frame_rate,
                                    clip_len=args.ava_temp_len,
                                    resize_size=args.ava_img_reshape_size,
                                    crop_size=args.ava_img_size,
                                    mode="train")
    elif image_set == 'val':
        val_bbox_json = json.load(open(args.ava_anno_path.format("val")))
        val_video_frame_bbox, val_frame_keys_list = val_bbox_json["video_frame_bbox"], val_bbox_json["frame_keys_list"]

        val_dataset = VideoDataset(args.ava_path,
                                val_video_frame_bbox,
                                val_frame_keys_list,
                                transforms=make_transforms("val", args),
                                frame_sample_rate=args.ava_frame_rate,
                                clip_len=args.ava_temp_len,
                                resize_size=args.ava_img_reshape_size,
                                crop_size=args.ava_img_size,
                                mode="val")
    else:
        raise AssertionError("image_set value is either train or val")

    try:
        return train_dataset
    except:
        return val_dataset

def reverse_norm(imgs):
    img = imgs
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img = (img * std + mean) * 255.0
    img = img.transpose((1, 2, 0))[..., ::-1].astype(np.uint8)
    return img