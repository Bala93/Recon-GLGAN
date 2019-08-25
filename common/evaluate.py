import argparse
import pathlib
from argparse import ArgumentParser
import h5py
import numpy as np
from runstats import Statistics
from skimage.measure import compare_psnr, compare_ssim

def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return compare_psnr(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return compare_ssim(
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    )


METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        self.metrics = {
            metric: Statistics() for metric in metric_funcs
        }

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {
            metric: stat.mean() for metric, stat in self.metrics.items()
        }

    def stddevs(self):
        return {
            metric: stat.stddev() for metric, stat in self.metrics.items()
        }

    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return " ".join(f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names)


def evaluate(args):
    metrics_global = Metrics(METRIC_FUNCS)
    metrics_local = Metrics(METRIC_FUNCS)

    roi = args.roi_size

    for tgt_file in args.target_path.iterdir():
        with h5py.File(tgt_file) as target, h5py.File(
          args.predictions_path / tgt_file.name) as recons:

            target2 = target['volfs'].value
            target2 = np.transpose(target2, (2, 0, 1))
            center_coord = target['center_coord'].value

            gt_patch = []
            recons_patch = []


            for i in range(len(target2)):
                gt_patch.append(target2[i,int(center_coord[1])-roi:int(center_coord[1])+roi, int(center_coord[0])-roi:int(center_coord[0])+roi])

            recons = recons['reconstruction'].value
            recons = recons[:, 5:155, 5:155] # Predicted data will be in dimension (160 x 160), center cropping to (150x150)

            for i in range(len(target2)):
                recons_patch.append(recons[i,int(center_coord[1])-roi:int(center_coord[1])+roi, int(center_coord[0])-roi:int(center_coord[0])+roi])

            # Code to calculate metrics both whole image and patch wise 
                
            metrics_global.push(target2, recons)
            metrics_local.push(np.asarray(gt_patch) , np.asarray(recons_patch))

    return metrics_local, metrics_global


if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target-path', type=pathlib.Path, required=True,
                        help='Path to the ground truth data')
    parser.add_argument('--predictions-path', type=pathlib.Path, required=True,
                        help='Path to reconstructions')
    parser.add_argument('--roi-size', type=int, default=roi, help='Dimensions of the square ROI')  
    args = parser.parse_args()

    metrics_local, metrics_global = evaluate(args)
    print("metrics_ROI")
    print(metrics_local)
    print("metrics_fullImage")
    print(metrics_global)
