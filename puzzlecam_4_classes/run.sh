CUDA_VISIBLE_DEVICES=0,1,2 python3 train_classification_with_puzzle.py --architecture resnest101 \
    --re_loss_option masking --re_loss L1_Loss --alpha_schedule 0.50 --alpha 4.00 \
    --tag ResNeSt101@Puzzle@optimal \
    --data_dir your_data_dir

CUDA_VISIBLE_DEVICES=0 python3 inference_classification.py --architecture resnest101 \
    --tag ResNeSt101@Puzzle@optimal --domain train_aug \
    --data_dir your_data_dir

python3 make_affinity_labels.py --experiment_name ResNeSt101@Puzzle@optimal@train@scale=0.5,1.0,1.5,2.0 \
    --domain train_aug --fg_threshold 0.40 --bg_threshold 0.10 \
    --data_dir your_data_dir

CUDA_VISIBLE_DEVICES=0 python3 train_affinitynet.py --architecture resnest101 \
    --tag AffinityNet@ResNeSt-101@Puzzle \
    --label_name ResNeSt101@Puzzle@optimal@train@scale=0.5,1.0,1.5,2.0@aff_fg=0.40_bg=0.10 \
    --data_dir your_data_dir

CUDA_VISIBLE_DEVICES=0 python3 inference_rw.py --architecture resnest101 \
    --model_name AffinityNet@ResNeSt-101@Puzzle --cam_dir ResNeSt101@Puzzle@optimal@train@scale=0.5,1.0,1.5,2.0 \
    --domain train_aug \
    --data_dir your_data_dir

python3 make_pseudo_labels.py --experiment_name AffinityNet@ResNeSt-101@Puzzle@train@beta=10@exp_times=8@rw \
    --domain train_aug --threshold 0.35 --crf_iteration 1 \
    --data_dir your_data_dir

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_segmentation.py \
    --backbone resnest101 --mode fix --use_gn True --tag DeepLabv3+@ResNeSt-101@Fix@GN \
    --label_name AffinityNet@ResNeSt-101@Puzzle@train@beta=10@exp_times=8@rw@crf=1 \
    --data_dir your_data_dir

CUDA_VISIBLE_DEVICES=0 python3 inference_segmentation.py --backbone resnest101 --mode fix --use_gn True \
    --tag DeepLabv3+@ResNeSt-101@Fix@GN --scale 0.5,1.0,1.5,2.0 \
    --iteration 10 \
    --data_dir your_data_dir

python3 evaluate.py --experiment_name DeepLabv3+@ResNeSt-101@Fix@GN@val@scale=0.5,1.0,1.5,2.0@iteration=10 \
    --domain val \
    --gt_dir your_data_dir/SegmentationClass

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

python3 evaluate.py --experiment_name DeepLabv3+@aug@val@scale=0.5,1.0,1.5,2.0@iteration=10 \
    --domain val \
    --mode png \
    --gt_dir your_data_dir/SegmentationClass

CUDA_VISIBLE_DEVICES=0 python3 inference_segmentation.py --backbone resnest101 --mode fix --use_gn True \
    --tag DeepLabv3+@aug --scale 0.5,1.0,1.5,2.0 \
    --iteration 10 \
    --data_dir your_data_dir

CUDA_VISIBLE_DEVICES=0 python3 visual.py --architecture resnest101 \
    --re_loss_option masking --re_loss L1_Loss --alpha_schedule 0.50 --alpha 4.00 \
    --tag ResNeSt101@Puzzle@optimal \
    --data_dir your_data_dir