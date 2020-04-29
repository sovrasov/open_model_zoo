"""
 Copyright (c) 2020 Intel Corporation
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

import argparse
import json
import glob
import os
import logging as log

import motmetrics as mm
import numpy as np
from xml.etree import ElementTree as etree
from tqdm import tqdm

from mc_tracker.sct import TrackedObj
from utils.misc import set_log_config

set_log_config()


def read_gt_tracks(gt_filenames, size_divisor=1, skip_frames=0, skip_heavy_occluded_objects=False):
    min_last_frame_idx = -1
    camera_tracks = [[] for _ in gt_filenames]
    for i, filename in enumerate(gt_filenames):
        last_frame_idx = -1
        tree = etree.parse(filename)
        root = tree.getroot()
        for track_xml_subtree in tqdm(root, desc='Reading ' + filename):
            if track_xml_subtree.tag != 'track':
                continue
            track = {'id': None, 'boxes': [], 'timestamps': []}
            for box_tree in track_xml_subtree.findall('box'):
                if skip_frames > 0 and int(box_tree.get('frame')) % skip_frames == 0:
                    continue
                occlusion = [tag.text for tag in box_tree if tag.attrib['name'] == 'occlusion'][0]
                if skip_heavy_occluded_objects and occlusion == 'heavy_occluded':
                    continue
                x_left = int(float(box_tree.get('xtl'))) // size_divisor
                x_right = int(float(box_tree.get('xbr'))) // size_divisor
                y_top = int(float(box_tree.get('ytl'))) // size_divisor
                y_bottom = int(float(box_tree.get('ybr'))) // size_divisor
                assert x_right > x_left
                assert y_bottom > y_top
                track['boxes'].append([x_left, y_top, x_right, y_bottom])
                track['timestamps'].append(int(box_tree.get('frame')) // size_divisor)
                last_frame_idx = max(last_frame_idx, track['timestamps'][-1])
                id = [int(tag.text) for tag in box_tree if tag.attrib['name'] == 'id'][0]
            track['id'] = id
            camera_tracks[i].append(track)
        if min_last_frame_idx < 0:
            min_last_frame_idx = last_frame_idx
        else:
            min_last_frame_idx = min(min_last_frame_idx, last_frame_idx)

    return camera_tracks, min_last_frame_idx


def get_detections_from_tracks(tracks_history, time):
    active_detections = [[] for _ in tracks_history]
    for i, camera_hist in enumerate(tracks_history):
        for track in camera_hist:
            if time in track['timestamps']:
                idx = track['timestamps'].index(time)
                active_detections[i].append(TrackedObj(track['boxes'][idx], track['id']))
    return active_detections


def check_contain_duplicates(all_detections):
    for detections in all_detections:
        all_labels = [obj.label for obj in detections]
        uniq = set(all_labels)
        if len(all_labels) != len(uniq):
            return True

    return False


def accumulate_results(gt_data, tracking_results, accumulator, seq_name=''):
    for frame_idx in tqdm(sorted(gt_data), 'Processing sequence ' + seq_name):
        time = frame_idx - 1
        active_detections = get_detections_from_tracks(tracking_results, time)
        if check_contain_duplicates(active_detections):
            log.info('Warning: at least one IDs collision has occured at the timestamp ' + str(time))

        gt_boxes = []
        gt_labels = []
        for obj in gt_data[frame_idx]:
            gt_boxes.append([obj.rect[0], obj.rect[1],
                             obj.rect[2],
                             obj.rect[3]])
            gt_labels.append(obj.label)

        ht_boxes = []
        ht_labels = []
        for obj in active_detections[0]:
            ht_boxes.append([obj.rect[0], obj.rect[1],
                             obj.rect[2] - obj.rect[0],
                             obj.rect[3] - obj.rect[1]])
            ht_labels.append(obj.label)

        distances = mm.distances.iou_matrix(np.array(gt_boxes),
                                            np.array(ht_boxes), max_iou=0.5)
        accumulator.update(gt_labels, ht_labels, distances)


def main():
    """Computes MOT metrics for the multi camera multi person tracker"""
    parser = argparse.ArgumentParser(description='Multi camera multi person \
                                                  tracking visualization demo script')
    parser.add_argument('--mot_results_folder', type=str, default='', required=True,
                        help='Folder with detections')
    parser.add_argument('--gt_folder', type=str, required=True,
                        help='Folder with the ground truth annotation')

    args = parser.parse_args()

    results_paths = sorted(glob.glob(os.path.join(args.mot_results_folder, '*.json')))
    tracking_results = {}
    for path in results_paths:
        with open(path) as hist_f:
            seq_name = os.path.basename(path).split('.')[0]
            tracking_results[seq_name] = json.load(hist_f)

    gt_boxes = {}
    gt_paths = glob.glob(os.path.join(args.gt_folder, '*-FRCNN/gt/gt.txt'))
    for gt_path in gt_paths:
        seq_name = gt_path.split('/')[-3]

        seq_detections = {}
        lines = []
        with open(gt_path) as f:
            lines = f.readlines()
        for line in lines:
            numbers = [int(float(x)) for x in line.split(',')][:-2]
            frame, id, bb_left, bb_top, bb_width, bb_height, conf = numbers
            if frame not in seq_detections:
                seq_detections[frame] = []
            seq_detections[frame].append(TrackedObj([bb_left, bb_top, bb_width, bb_height], id))
        gt_boxes[seq_name] = seq_detections


    accs = [mm.MOTAccumulator(auto_id=True) for _ in tracking_results]

    for i, seq in enumerate(sorted(tracking_results)):
        assert seq in gt_boxes
        accumulate_results(gt_boxes[seq], tracking_results[seq], accs[i])

    log.info('Computing MOT metrics...')
    mh = mm.metrics.create()
    summary = mh.compute_many(accs,
                              metrics=mm.metrics.motchallenge_metrics,
                              generate_overall=True,
                              names=[name for name in sorted(tracking_results)])

    strsummary = mm.io.render_summary(summary,
                                      formatters=mh.formatters,
                                      namemap=mm.io.motchallenge_metric_names)
    print(strsummary)


if __name__ == '__main__':
    main()
