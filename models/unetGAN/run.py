import pathlib
import sys
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from common.utils import save_reconstructions
from data.mri_data import SliceData, SliceDataDev
from models.unetGAN.model import UNet
from tqdm import tqdm

def create_data_loaders(args):
    data = SliceDataDev(args.val_path, args.acceleration)
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=32,
        pin_memory=True,
    )
    return data_loader

def build_model(args):
    model = UNet(1,1).to(args.device)
    return model

def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    return checkpoint, model

def run(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (input, target, coord, fnames, slices) in tqdm(data_loader):
            input = input.float()
            input = input.unsqueeze(1).to(args.device)
            recons = model(input).to('cpu').squeeze(1)
            for i in range(recons.shape[0]):
                reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }
    return reconstructions


def main(args):
    print('creating data loaders...')
    data_loader = create_data_loaders(args)
    print('loading model...')
    checkpoint, model = load_model(args.checkpoint)
    print('running model...')
    reconstructions = run(args, model, data_loader)
    save_reconstructions(reconstructions, args.out_dir)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-path', type=string, required=True, help='Path to validation data')
    parser.add_argument('--acceleration', type=int, choices=[2, 4, 8], default=4, help='Acceleration factor for undersampled data')
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to') 
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
