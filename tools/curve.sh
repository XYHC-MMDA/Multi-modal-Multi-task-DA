#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=5 python tools/mmda_test.py configs/merge_aug_weights/fusion_train_day.py checkpoints/merge_aug_weights/fusion_train_day/epoch_35.pth --eval mAP --json checkpoints/merge_aug_weights/fusion_train_day/35_test_night
CUDA_VISIBLE_DEVICES=5 python tools/mmda_test.py configs/merge_aug_weights/fusion_train_day.py checkpoints/merge_aug_weights/fusion_train_day/epoch_34.pth --eval mAP --json checkpoints/merge_aug_weights/fusion_train_day/34_test_night
CUDA_VISIBLE_DEVICES=5 python tools/mmda_test.py configs/merge_aug_weights/fusion_train_day.py checkpoints/merge_aug_weights/fusion_train_day/epoch_33.pth --eval mAP --json checkpoints/merge_aug_weights/fusion_train_day/33_test_night
CUDA_VISIBLE_DEVICES=5 python tools/mmda_test.py configs/merge_aug_weights/fusion_train_day.py checkpoints/merge_aug_weights/fusion_train_day/epoch_32.pth --eval mAP --json checkpoints/merge_aug_weights/fusion_train_day/32_test_night
CUDA_VISIBLE_DEVICES=5 python tools/mmda_test.py configs/merge_aug_weights/fusion_train_day.py checkpoints/merge_aug_weights/fusion_train_day/epoch_31.pth --eval mAP --json checkpoints/merge_aug_weights/fusion_train_day/31_test_night
CUDA_VISIBLE_DEVICES=5 python tools/mmda_test.py configs/merge_aug_weights/fusion_train_day.py checkpoints/merge_aug_weights/fusion_train_day/epoch_30.pth --eval mAP --json checkpoints/merge_aug_weights/fusion_train_day/30_test_night
CUDA_VISIBLE_DEVICES=5 python tools/mmda_test.py configs/merge_aug_weights/fusion_train_day.py checkpoints/merge_aug_weights/fusion_train_day/epoch_29.pth --eval mAP --json checkpoints/merge_aug_weights/fusion_train_day/29_test_night
CUDA_VISIBLE_DEVICES=5 python tools/mmda_test.py configs/merge_aug_weights/fusion_train_day.py checkpoints/merge_aug_weights/fusion_train_day/epoch_28.pth --eval mAP --json checkpoints/merge_aug_weights/fusion_train_day/28_test_night
CUDA_VISIBLE_DEVICES=5 python tools/mmda_test.py configs/merge_aug_weights/fusion_train_day.py checkpoints/merge_aug_weights/fusion_train_day/epoch_27.pth --eval mAP --json checkpoints/merge_aug_weights/fusion_train_day/27_test_night
CUDA_VISIBLE_DEVICES=5 python tools/mmda_test.py configs/merge_aug_weights/fusion_train_day.py checkpoints/merge_aug_weights/fusion_train_day/epoch_26.pth --eval mAP --json checkpoints/merge_aug_weights/fusion_train_day/26_test_night
CUDA_VISIBLE_DEVICES=5 python tools/mmda_test.py configs/merge_aug_weights/fusion_train_day.py checkpoints/merge_aug_weights/fusion_train_day/epoch_25.pth --eval mAP --json checkpoints/merge_aug_weights/fusion_train_day/25_test_night
CUDA_VISIBLE_DEVICES=5 python tools/mmda_test.py configs/merge_aug_weights/fusion_train_day.py checkpoints/merge_aug_weights/fusion_train_day/epoch_24.pth --eval mAP --json checkpoints/merge_aug_weights/fusion_train_day/24_test_night
CUDA_VISIBLE_DEVICES=5 python tools/mmda_test.py configs/merge_aug_weights/fusion_train_day.py checkpoints/merge_aug_weights/fusion_train_day/epoch_23.pth --eval mAP --json checkpoints/merge_aug_weights/fusion_train_day/23_test_night
CUDA_VISIBLE_DEVICES=5 python tools/mmda_test.py configs/merge_aug_weights/fusion_train_day.py checkpoints/merge_aug_weights/fusion_train_day/epoch_22.pth --eval mAP --json checkpoints/merge_aug_weights/fusion_train_day/22_test_night
CUDA_VISIBLE_DEVICES=5 python tools/mmda_test.py configs/merge_aug_weights/fusion_train_day.py checkpoints/merge_aug_weights/fusion_train_day/epoch_21.pth --eval mAP --json checkpoints/merge_aug_weights/fusion_train_day/21_test_night
CUDA_VISIBLE_DEVICES=5 python tools/mmda_test.py configs/merge_aug_weights/fusion_train_day.py checkpoints/merge_aug_weights/fusion_train_day/epoch_20.pth --eval mAP --json checkpoints/merge_aug_weights/fusion_train_day/20_test_night
CUDA_VISIBLE_DEVICES=5 python tools/mmda_test.py configs/merge_aug_weights/fusion_train_day.py checkpoints/merge_aug_weights/fusion_train_day/epoch_19.pth --eval mAP --json checkpoints/merge_aug_weights/fusion_train_day/18_test_night

