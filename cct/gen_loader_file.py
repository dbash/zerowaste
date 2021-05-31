import os
import numpy as np
import tqdm
import argparse

opj = os.path.join
SEED = 123
DATA_ROOT = "/scratch2/dinka/data/recycling/splits_deblurred/"

def main(args):
    sup_name_list = os.listdir(args.sup_root)
    unsup_name_list = os.listdir(args.unsup_root)
    val_name_list = os.listdir(args.val_root)
    test_name_list = os.listdir(args.test_root)
    if args.random:
        sup_idx = np.arange(len(sup_name_list))
        unsup_idx = np.arange(len(unsup_name_list))
        np.random.shuffle(sup_idx, seed=SEED)
        np.random.shuffle(unsup_idx, seed=SEED)
        sup_name_list = sup_name_list[sup_idx]
        unsup_name_list = unsup_name_list[unsup_idx]
    
    sup_fl_name = opj(args.root, "%i_train_supervised.txt" % args.num_train)
    unsup_fl_name = opj(args.root, "%i_train_unsupervised.txt" % args.num_train)
    val_fl_name = opj(args.root, "val.txt")
    test_fl_name = opj(args.root, "test.txt")

    print("Writing the supervised train loader")
    with open(sup_fl_name, "w") as fl:
        for fname in tqdm.tqdm(sup_name_list[:args.num_train]):
            fl.write("%s\n" % fname)
    print("Writing the unsupervised train loader")
    with open(unsup_fl_name, "w") as fl:
        for fname in tqdm.tqdm(unsup_name_list[:args.num_unsup]):
            fl.write("%s\n" % fname)
    print("Writing the val loader")
    with open(val_fl_name, "w")  as fl:
        for fname in tqdm.tqdm(val_name_list):
            fl.write("%s\n" % fname)
    print("Writing the test loader")
    with open(test_fl_name, "w") as fl:
        for fname in tqdm.tqdm(test_name_list):
            fl.write("%s\n" % fname)
    print('finished.')

        

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Generade a loader txt file')
    parser.add_argument('--root', default='./dataloaders/zerowaste_splits/',type=str,
                        help='Path to the the dataloaders root')
    parser.add_argument('--num_train',  default=1245, type=int,
                        help='number of labeled training samples')
    parser.add_argument('--num_unsup', default=-1, type=int,
                        help='number of unlabeled training samples')
    parser.add_argument('--sup_root',  default=opj(DATA_ROOT, "train", "data"), type=str,
                           help='path to the supervised data root')
    parser.add_argument('--val_root',  default=opj(DATA_ROOT, "val", "data"), type=str,
                           help='path to the val data root')
    parser.add_argument('--test_root',  default=opj(DATA_ROOT, "test", "data"), type=str,
                           help='path to the test data root')
    parser.add_argument('--unsup_root',  default=opj(DATA_ROOT, "unlabeled", "data"), type=str,
                           help='path to the supervised data root')
    parser.add_argument('--random', action='store_true', default=False)
    args = parser.parse_args()

    
    main(args)
