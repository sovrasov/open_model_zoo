#!/bin/bash

mot_root=$1
reid_model=$2
output_folder=$3
detection_model=$4
echo $detection_model

declare -a mot_train_seqs=("02" "04" "05" "09" "10" "11" "13")

for seq in "${mot_train_seqs[@]}"
do
  if [ -z "$detection_model" ]
  then
    detection_command="--detections $mot_root/MOT17-$seq-FRCNN"
  else
    detection_command="-m $detection_model"
  fi
  echo $detection_command
  python3 multi_camera_multi_person_tracking.py --config ./config.py --m_reid $reid_model -i $mot_root/MOT17-$seq-FRCNN/img1/ --mot_eval --history_file $output_folder/MOT17-$seq-FRCNN.json $detection_command
done

python3 run_mot_evaluation.py --mot_results_folder $output_folder --gt_folder $mot_root
