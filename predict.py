import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import cv2
import PIL.Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import GradualWarmupSchedulerV2
# import apex
# from apex import amp
from dataset import get_df, get_transforms, MelanomaDataset
from models import Effnet_Melanoma, Resnest_Melanoma, Seresnext_Melanoma
from train import get_trans
from config import config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='/raid/')
    parser.add_argument('--data-folder', type=int, required=True)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--out-dim', type=int, default=9)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--sub-dir', type=str, default='./subs')
    parser.add_argument('--eval', type=str, choices=['best', 'best_20', 'final'], default="best")
    parser.add_argument('--n-test', type=int, default=8)
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--n-meta-dim', type=str, default='512,128')

    args, _ = parser.parse_known_args()
    return args

def predict_image(image_path):
    
    OUTPUTS = []
    n_test = 8
    transforms_train, transforms_val = get_transforms(config.image_size)

    dataset_test = MelanomaDataset(None, 'test', None, transform=transforms_val, image_path=image_path)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, num_workers=0)

    for fold in range(5):
        model_file = os.path.join(config.model_dir, f'{config.kernel_type}_best_o_fold{fold}.pth')

        ModelClass = Effnet_Melanoma
        model = ModelClass(config.enet_type, out_dim=config.out_dim)
        model = model.to(config.device)

        try:  # single GPU model_file
            model.load_state_dict(torch.load(model_file, map_location=config.device), strict=True)
        except:  # multi GPU model_file
            state_dict = torch.load(model_file, map_location=config.device)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=True)

        model.eval()
        
        LOGITS = []
        PROBS = []

        with torch.no_grad():
            for (data) in tqdm(test_loader):
                
                if config.use_meta:
                    data, meta = data
                    data, meta = data.to(config.device), meta.to(config.device)
                    logits = torch.zeros((data.shape[0], config.out_dim)).to(config.device)
                    probs = torch.zeros((data.shape[0], config.out_dim)).to(config.device)
                    for I in range(n_test):
                        l = model(get_trans(data, I), meta)
                        logits += l
                        probs += l.softmax(1)
                else:
                    data = data.to(config.device)
                    logits = torch.zeros((data.shape[0], config.out_dim)).to(config.device)
                    probs = torch.zeros((data.shape[0], config.out_dim)).to(config.device)
                    for I in range(n_test):
                        l = model(get_trans(data, I))
                        logits += l
                        probs += l.softmax(1)
                logits /= n_test
                probs /= n_test
        
                LOGITS.append(logits.detach().cpu())
                PROBS.append(probs.detach().cpu())

        LOGITS = torch.cat(LOGITS).numpy()
        PROBS = torch.cat(PROBS).numpy()
        
        OUTPUTS.append(PROBS[:, config.mel_idx])
    
    #If you are predicting on your own moles, your don't need to rank the probability
    pred = np.zeros(OUTPUTS[0].shape[0])
    for probs in OUTPUTS:
        pred += probs
    pred /= len(OUTPUTS)
    
    return round(pred[0], 8)

def main():

    df, df_test, meta_features, n_meta_features, mel_idx = get_df(
        args.kernel_type,
        args.out_dim,
        args.data_dir,
        args.data_folder,
        args.use_meta
    )

    transforms_train, transforms_val = get_transforms(args.image_size)

    if args.DEBUG:
        df_test = df_test.sample(args.batch_size * 3)
    dataset_test = MelanomaDataset(df_test, 'test', meta_features, transform=transforms_val)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers)

    print(f'\nPredicting test set using {args.enet_type} ...')

    OUTPUTS = []
    for fold in range(5):

        if args.eval == 'best':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold{fold}.pth')
        elif args.eval == 'best_20':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_20_fold{fold}.pth')
        if args.eval == 'final':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold{fold}.pth')

        model = ModelClass(
            args.enet_type,
            n_meta_features=n_meta_features,
            n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
            out_dim=args.out_dim
        )
        model = model.to(device)

        try:  # single GPU model_file
            model.load_state_dict(torch.load(model_file), strict=True)
        except:  # multi GPU model_file
            state_dict = torch.load(model_file)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=True)
        
        if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
            model = torch.nn.DataParallel(model)

        model.eval()

        PROBS = []

        with torch.no_grad():
            for (data) in tqdm(test_loader):

                if args.use_meta:
                    data, meta = data
                    data, meta = data.to(device), meta.to(device)
                    probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    for I in range(args.n_test):
                        l = model(get_trans(data, I), meta)
                        probs += l.softmax(1)
                else:   
                    data = data.to(device)
                    probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    for I in range(args.n_test):
                        l = model(get_trans(data, I))
                        probs += l.softmax(1)

                probs /= args.n_test

                PROBS.append(probs.detach().cpu())

        PROBS = torch.cat(PROBS).numpy()
        OUTPUTS.append(PROBS[:, mel_idx])

    # Rank per fold (If you are predicting on your own moles, your don't need to rank the probability)
    pred = np.zeros(OUTPUTS[0].shape[0])
    for probs in OUTPUTS:
        pred += pd.Series(probs).rank(pct=True).values
    pred /= len(OUTPUTS)

    df_test['target'] = pred
    df_test[['image_name', 'target']].to_csv(os.path.join(args.sub_dir, f'sub_{args.kernel_type}_{args.eval}.csv'), 
                                             index=False)
    print('\nSaved submission in -> ./subs')


if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.sub_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    if args.enet_type == 'resnest101':
        ModelClass = Resnest_Melanoma
    elif args.enet_type == 'seresnext101':
        ModelClass = Seresnext_Melanoma
    elif 'efficientnet' in args.enet_type:
        ModelClass = Effnet_Melanoma
    else:
        raise NotImplementedError()

    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    main()