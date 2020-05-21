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

import argparse
import json
import logging as log
from os import path as osp

import motmetrics as mm
import numpy as np
from xml.etree import ElementTree as etree
from tqdm import tqdm

from mc_tracker.sct import TrackedObj
from utils.misc import set_log_config

set_log_config()


def read_txt(file_name):
    camera_tracks = []
    with open(file_name, 'r') as f:
        gt_lines = f.readlines()
    meta_file = osp.join(osp.abspath(osp.dirname(file_name)), '../seqinfo.ini')
    if osp.isfile(meta_file):
        with open(meta_file, 'r') as f:
            meta = f.readlines()
        params = {}
        for l in meta:
            l = l.split('=')
            if len(l) > 1:
                params[l[0]] = l[1].replace('\n', '')
        width, height = int(params['imWidth']), int(params['imHeight'])
    else:
        width, height = None, None
    last_frame_num = 0
    for l in gt_lines:
        l = l.split(',')
        frame_num = int(l[0]) - 1
        pid = int(l[1])
        x0, y0, w, h = int(float(l[2])), int(float(l[3])), int(float(l[4])), int(float(l[5]))
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = x0 + w, y0 + h
        if width is not None and height is not None:
            x1, y1 = min(x1, width - 1), min(y1, height - 1)
        conf = float(l[6])
        pid_found = False
        for i in range(len(camera_tracks)):
            if camera_tracks[i]['id'] == pid:
                camera_tracks[i]['boxes'].append([x0, y0, x1, y1])
                camera_tracks[i]['timestamps'].append(frame_num)
                pid_found = True
                break
        if not pid_found:
            camera_tracks.append({'id': pid, 'boxes': [[x0, y0, x1, y1]], 'timestamps': [frame_num]})
    last_frame_num = max(last_frame_num, frame_num)
    return camera_tracks, last_frame_num


def read_gt_tracks(gt_filenames, size_divisor=1, skip_frames=0, skip_heavy_occluded_objects=False):
    camera_tracks = [[] for _ in gt_filenames]
    frame_nums = [[] for _ in gt_filenames]
    for i, filename in tqdm(enumerate(gt_filenames), 'Reading ground truth...'):
        if filename.endswith('.txt'):
            camera_tracks[i], frame_nums[i] = read_txt(filename)
            continue
        last_frame_idx = 0
        tree = etree.parse(filename)
        root = tree.getroot()
        for track_xml_subtree in root:
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
        frame_nums[i] = last_frame_idx
    return camera_tracks, frame_nums


def get_detections_from_tracks(tracks_history, time):
    active_detections = []
    for track in tracks_history:
        if time in track['timestamps']:
            idx = track['timestamps'].index(time)
            active_detections.append(TrackedObj(track['boxes'][idx], track['id']))
    return active_detections


def main():
    """Computes MOT metrics for the multi camera multi person tracker"""
    parser = argparse.ArgumentParser(description='Multi camera multi person \
                                                  tracking visualization demo script')
    parser.add_argument('--history_file', type=str, nargs='+', required=True,
                        help='Files with tracker history')
    parser.add_argument('--gt_files', type=str, nargs='+', required=True,
                        help='Files with ground truth annotation')
    parser.add_argument('--size_divisor', type=int, default=1,
                        help='Scale factor for GT image resolution')
    parser.add_argument('--skip_frames', type=int, default=0,
                        help='Frequency of skipping frames')
    args = parser.parse_args()

    assert len(args.gt_files) == len(args.history_file)
    gt_tracks, last_frame_idx = read_gt_tracks(args.gt_files,
                                               size_divisor=args.size_divisor,
                                               skip_frames=args.skip_frames)
    accs = [mm.MOTAccumulator(auto_id=True) for _ in args.gt_files]
    names = [osp.basename(name).split('.')[0] for name in args.history_file]

    for n in tqdm(range(len(gt_tracks)), 'Processing detections'):
        if args.history_file[n].endswith('.json'):
            with open(args.history_file[n]) as hist_f:
                history = json.load(hist_f)
        elif args.history_file[n].endswith('.txt'):
            history = read_txt(args.history_file[n])
        else:
            raise ValueError('Unexpected results file format!')
        for time in range(last_frame_idx[n] + 1):
            gt_detections = get_detections_from_tracks(gt_tracks[n], time)
            active_detections = get_detections_from_tracks(history[0], time)

            gt_boxes = []
            gt_labels = []
            for i, obj in enumerate(gt_detections):
                gt_boxes.append([obj.rect[0], obj.rect[1],
                                 obj.rect[2] - obj.rect[0],
                                 obj.rect[3] - obj.rect[1]])
                gt_labels.append(obj.label)

            ht_boxes = []
            ht_labels = []
            for obj in active_detections:
                ht_boxes.append([obj.rect[0], obj.rect[1],
                                 obj.rect[2] - obj.rect[0],
                                 obj.rect[3] - obj.rect[1]])
                ht_labels.append(obj.label)

            distances = mm.distances.iou_matrix(np.array(gt_boxes),
                                                np.array(ht_boxes), max_iou=0.5)
            accs[n].update(gt_labels, ht_labels, distances)

    log.info('Computing MOT metrics...')
    mh = mm.metrics.create()
    summary = mh.compute_many(accs,
                              metrics=mm.metrics.motchallenge_metrics,
                              generate_overall=True,
                              names=names)#['video ' + str(i) for i in range(len(accs))])

    strsummary = mm.io.render_summary(summary,
                                      formatters=mh.formatters,
                                      namemap=mm.io.motchallenge_metric_names)
    print(strsummary)


if __name__ == '__main__':
    main()
