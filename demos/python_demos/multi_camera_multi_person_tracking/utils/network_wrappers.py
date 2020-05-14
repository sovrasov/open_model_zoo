"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import json
import os
import logging as log
from abc import ABC, abstractmethod

import cv2
import numpy as np

from utils.ie_tools import load_ie_model
from .segm_postprocess import postprocess


class DetectorInterface(ABC):
    @abstractmethod
    def run_async(self, frames, index):
        pass

    @abstractmethod
    def wait_and_grab(self):
        pass


class Detector(DetectorInterface):
    """Wrapper class for detector"""

    def __init__(self, ie, model_path, conf=.6, device='CPU', ext_path='', max_num_frames=1):
        self.net = load_ie_model(ie, model_path, device, None, ext_path, num_reqs=max_num_frames)
        self.confidence = conf
        self.expand_ratio = (1., 1.)
        self.max_num_frames = max_num_frames

    def run_async(self, frames, index):
        assert len(frames) <= self.max_num_frames
        self.shapes = []
        for i in range(len(frames)):
            self.shapes.append(frames[i].shape)
            self.net.forward_async(frames[i])

    def wait_and_grab(self):
        all_detections = []
        outputs = self.net.grab_all_async()
        for i, out in enumerate(outputs):
            detections = self.__decode_detections(out, self.shapes[i])
            all_detections.append(detections)
        return all_detections

    def get_detections(self, frames):
        """Returns all detections on frames"""
        self.run_async(frames)
        return self.wait_and_grab()

    def __decode_detections(self, out, frame_shape):
        """Decodes raw SSD output"""
        detections = []

        for detection in out[0, 0]:
            confidence = detection[2]
            if confidence > self.confidence:
                left = int(max(detection[3], 0) * frame_shape[1])
                top = int(max(detection[4], 0) * frame_shape[0])
                right = int(max(detection[5], 0) * frame_shape[1])
                bottom = int(max(detection[6], 0) * frame_shape[0])
                if self.expand_ratio != (1., 1.):
                    w = (right - left)
                    h = (bottom - top)
                    dw = w * (self.expand_ratio[0] - 1.) / 2
                    dh = h * (self.expand_ratio[1] - 1.) / 2
                    left = max(int(left - dw), 0)
                    right = int(right + dw)
                    top = max(int(top - dh), 0)
                    bottom = int(bottom + dh)

                detections.append(((left, top, right, bottom), confidence))

        if len(detections) > 1:
            detections.sort(key=lambda x: x[1], reverse=True)

        return detections


class VectorCNN:
    """Wrapper class for a network returning a vector"""

    def __init__(self, ie, model_path, device='CPU', ext_path='', max_reqs=100):
        self.max_reqs = max_reqs
        self.net = load_ie_model(ie, model_path, device, None, ext_path, num_reqs=self.max_reqs)

    def forward(self, batch):
        """Performs forward of the underlying network on a given batch"""
        assert len(batch) <= self.max_reqs
        for frame in batch:
            self.net.forward_async(frame)
        outputs = self.net.grab_all_async()
        return outputs

    def forward_async(self, batch):
        """Performs async forward of the underlying network on a given batch"""
        assert len(batch) <= self.max_reqs
        for frame in batch:
            self.net.forward_async(frame)

    def wait_and_grab(self):
        outputs = self.net.grab_all_async()
        return outputs


class MaskRCNN(DetectorInterface):
    """Wrapper class for a network returning masks of objects"""

    def __init__(self, ie, model_path, conf=.6, device='CPU', ext_path='',
                 max_reqs=100):
        self.max_reqs = max_reqs
        self.confidence = conf
        self.net = load_ie_model(ie, model_path, device, None, ext_path, num_reqs=self.max_reqs)

        required_input_keys = [{'im_info', 'im_data'}, {'im_data', 'im_info'}]
        current_input_keys = set(self.net.inputs_info.keys())
        assert current_input_keys in required_input_keys
        required_output_keys = {'boxes', 'scores', 'classes', 'raw_masks'}
        assert required_output_keys.issubset(self.net.net.outputs)

        self.n, self.c, self.h, self.w = self.net.inputs_info['im_data'].shape
        assert self.n == 1, 'Only batch 1 is supported.'

    def preprocess(self, frame):
        image_height, image_width = frame.shape[:2]
        scale = min(self.h / image_height, self.w / image_width)
        processed_image = cv2.resize(frame, None, fx=scale, fy=scale)
        processed_image = processed_image.astype('float32').transpose(2, 0, 1)

        sample = dict(original_image=frame,
                      meta=dict(original_size=frame.shape[:2],
                                processed_size=processed_image.shape[1:3]),
                      im_data=processed_image,
                      im_info=np.array([processed_image.shape[1], processed_image.shape[2], 1.0], dtype='float32'))
        return sample

    def forward(self, im_data, im_info):
        if (self.h - im_data.shape[1] < 0) or (self.w - im_data.shape[2] < 0):
            raise ValueError('Input image should resolution of {}x{} or less, '
                             'got {}x{}.'.format(self.w, self.h, im_data.shape[2], im_data.shape[1]))
        im_data = np.pad(im_data, ((0, 0),
                                   (0, self.h - im_data.shape[1]),
                                   (0, self.w - im_data.shape[2])),
                         mode='constant', constant_values=0).reshape(1, self.c, self.h, self.w)
        im_info = im_info.reshape(1, *im_info.shape)
        output = self.net.net.infer(dict(im_data=im_data, im_info=im_info))

        classes = output['classes']
        valid_detections_mask = classes > 0
        classes = classes[valid_detections_mask]
        boxes = output['boxes'][valid_detections_mask]
        scores = output['scores'][valid_detections_mask]
        masks = output['raw_masks'][valid_detections_mask]
        return boxes, classes, scores, np.full(len(classes), 0, dtype=np.int32), masks

    def get_detections(self, frames, return_cropped_masks=True, only_class_person=True):
        outputs = []
        for frame in frames:
            data_batch = self.preprocess(frame)
            im_data = data_batch['im_data']
            im_info = data_batch['im_info']
            meta = data_batch['meta']

            boxes, classes, scores, _, masks = self.forward(im_data, im_info)
            scores, classes, boxes, masks = postprocess(scores, classes, boxes, masks,
                                                        im_h=meta['original_size'][0],
                                                        im_w=meta['original_size'][1],
                                                        im_scale_y=meta['processed_size'][0] / meta['original_size'][0],
                                                        im_scale_x=meta['processed_size'][1] / meta['original_size'][1],
                                                        full_image_masks=True, encode_masks=False,
                                                        confidence_threshold=self.confidence)
            frame_output = []
            for i in range(len(scores)):
                if only_class_person and classes[i] != 1:
                    continue
                bbox = [int(value) for value in boxes[i]]
                if return_cropped_masks:
                    left, top, right, bottom = bbox
                    mask = masks[i][top:bottom, left:right]
                else:
                    mask = masks[i]
                frame_output.append([bbox, scores[i], mask])
            outputs.append(frame_output)
        return outputs

    def run_async(self, frames, index):
        self.frames = frames

    def wait_and_grab(self):
        return self.get_detections(self.frames)


class DetectionsFromFileReader(DetectorInterface):
    """Read detection from *.json file.
    Format of the file should be:
    [
        {'frame_id': N,
         'scores': [score0, score1, ...],
         'boxes': [[x0, y0, x1, y1], [x0, y0, x1, y1], ...]},
        ...
    ]
    """

    def __init__(self, input_file, score_thresh):
        self.input_file = input_file
        self.score_thresh = score_thresh
        self.detections = []
        log.info('Loading {}'.format(input_file))
        with open(input_file) as f:
            all_detections = json.load(f)
        for source_detections in all_detections:
            detections_dict = {}
            for det in source_detections:
                detections_dict[det['frame_id']] = {'boxes': det['boxes'], 'scores': det['scores']}
            self.detections.append(detections_dict)

    def run_async(self, frames, index):
        self.last_index = index

    def wait_and_grab(self):
        output = []
        for source in self.detections:
            valid_detections = []
            if self.last_index in source:
                for bbox, score in zip(source[self.last_index]['boxes'], source[self.last_index]['scores']):
                    if score > self.score_thresh:
                        bbox = [int(value) for value in bbox]
                        valid_detections.append((bbox, score))
            output.append(valid_detections)
        return output


class MOTDetectionsReader(DetectorInterface):
    def __init__(self, input_seq_path, score_thresh, half_mode=False):
        self.score_thresh = score_thresh
        self.detections = {}
        log.info('Loading {}'.format(input_seq_path))

        all_detections = []
        with open(os.path.join(input_seq_path, 'det/det.txt')) as f:
            all_detections = f.readlines()
        for line in all_detections:
            numbers = [int(float(x)) for x in line.split(',')]
            frame, id, bb_left, bb_top, bb_width, bb_height, conf = numbers
            if frame not in self.detections:
                self.detections[frame] = []
            self.detections[frame].append(([bb_left, bb_top, bb_left + bb_width, bb_top + bb_height], 1.))

    def run_async(self, frames, index):
        self.last_index = index

    def wait_and_grab(self):
        return [self.detections[self.last_index + 1]]


# for FAIRMOT
import sys
sys.path.append('/home/sovrasov/repositories/FairMOT/src/')
print(sys.path)
from lib.models.decode import mot_decode
from lib.models.model import create_model, load_model
from lib.models.utils import _tranpose_and_gather_feat
from lib.utils.post_process import ctdet_post_process
import torch
import torch.nn.functional as F


def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


class FairMOTWrapper(DetectorInterface):
    def __init__(self, weights_path, score_thresh):
        self.score_thresh = score_thresh
        self.arch = 'dla_34'
        self.num_classes = 1
        self.cat_spec_wh = False
        self.reid_dim = 512
        self.K = 128
        self.down_ratio = 4
        self.reg_offset = True
        self.heads = {'hm': self.num_classes,
                      'wh': 2,
                      'id': self.reid_dim,
                      'reg': 2}
        self.head_conv = 256
        self.model = create_model(self.arch, self.heads, self.head_conv)
        self.model = load_model(self.model, weights_path)
        self.model = self.model.to(0)
        self.model.eval()
        self.width = 1088
        self.height = 608

    def _post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def _pre_process_img(self, img0):
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        return torch.from_numpy(img).cuda().unsqueeze(0), img0

    def _merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.K:
            kth = len(scores) - self.K
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def run_async(self, frames, index):
        im_blob, img0 = self._pre_process_img(frames[0])
        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.down_ratio,
                'out_width': inp_width // self.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg'] if self.reg_offset else None
            dets, inds = mot_decode(hm, wh, reg=reg, cat_spec_wh=self.cat_spec_wh, K=self.K)
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()

        dets = self._post_process(dets, meta)
        dets = self._merge_outputs([dets])[1]

        remain_inds = dets[:, 4] > self.score_thresh
        self.dets = []
        self.id_feature = id_feature[remain_inds]
        for det in dets[remain_inds]:
            det = [int(d) for d in det]
            self.dets.append((det[0:4], 1.))


        # vis
        '''
        for i in range(0, dets.shape[0]):
            bbox = dets[i][0:4]
            cv2.rectangle(img0, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
        cv2.imshow('dets', img0)
        cv2.waitKey(0)
        id0 = id0-1
        '''

    def wait_and_grab(self):
        return [self.dets]
