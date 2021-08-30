python dataset/pascal_preprocess.py

CUDA_VISIBLE_DEVICES=1 python train_semisup.py --dataset pascal --num_labels 60 --apply_aug classmix

python train_semisup.py --dataset pascal --num_labels 60 --apply_aug classmix --apply_reco


python train_sup.py --dataset pascal --num_labels 0 --seed 1



python test_semi.py --dataset pascal --num_labels 0 --apply_aug classmix --apply_reco

python test_sup.py --dataset pascal --num_labels 0 --seed 1
