import argparse
import cv2
import os
from os import path as osp
from tqdm import tqdm
from xml.etree import ElementTree as etree


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
            pid = track_xml_subtree.attrib['id']
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
            track['id'] = pid
            camera_tracks[i].append(track)
        if min_last_frame_idx < 0:
            min_last_frame_idx = last_frame_idx
        else:
            min_last_frame_idx = min(min_last_frame_idx, last_frame_idx)

    return camera_tracks, min_last_frame_idx


def main():
    parser = argparse.ArgumentParser(description='Crop video and annotation')
    parser.add_argument('--video', type=str, default='', required=True,
                        help='Video file')
    parser.add_argument('--annotation', type=str, default='', required=False,
                        help='Annotation file')
    parser.add_argument('--time', type=float, nargs='+',
                        help='Start and end time in minutes')
    parser.add_argument('--output', type=str, default='',
                        help='Output directory')
    args = parser.parse_args()
    video = cv2.VideoCapture(args.video)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_start  = int(fps * args.time[0] * 60)
    frame_last = int(fps * args.time[1] * 60) if args.time[1] > 0 else length
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out_name = osp.basename(args.video)
    directory = video_out_name.split('.')[0]
    out_dir = osp.join(args.output, directory)
    if not osp.isdir(out_dir):
        os.makedirs(out_dir)
    video_out_name = osp.join(args.output, directory, video_out_name)
    output_video = cv2.VideoWriter(video_out_name, fourcc, fps, (width, height))
    for i in tqdm(range(frame_last + 1), 'Cropping video...'):
        has_frame, frame = video.read()
        if not has_frame:
            break
        if i >= frame_start:
            output_video.write(frame)
    output_video.release()
    print('Result video saved to: {}'.format(video_out_name))
    if not args.annotation:
        return
    tracks, _ = read_gt_tracks([args.annotation])
    pattern = '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'
    output = ''
    txt_out_name = osp.join(args.output, directory, 'gt.txt')
    for track in tqdm(tracks[0], 'Saving annotation...'):
        pid = track['id']
        for i, box in enumerate(track['boxes']):
            frame_id = track['timestamps'][i]
            if frame_start <= frame_id <= frame_last or args.time[1] == -1:
                x0, y0, x1, y1 = box
                w, h = x1 - x0, y1 - y0
                output += pattern.format(frame_id + 1, pid, x0, y0, w, h, 1, -1, -1, -1)
    with open(txt_out_name, 'w') as outf:
        outf.write(output)
    print('Result annotation saved to: {}'.format(txt_out_name))


if __name__ == '__main__':
    main()

