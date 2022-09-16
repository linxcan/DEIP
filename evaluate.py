import os
import cv2
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
import networks
from kitti_dataset import Kitti



STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """
    Computation of error metrics between predicted and ground truth depths

    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3





def readlines(filename):
    """
    Read all the lines in a text file and return as a list

    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def evaluate(opt):
    """
    Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    # opt.load_weights_folder='./model/'

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    filenames = readlines(opt.test_file)
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    dataset = Kitti(opt.data_path, filenames,encoder_dict['height'], encoder_dict['width'],
                                       [0], 4, is_train=False)
    dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=12,
                            pin_memory=True, drop_last=False)

    #load model
    encoder = networks.ResnetEncoder(18, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    pred_disps = []

    print("-> Computing predictions with size {}x{}".format(
        encoder_dict['width'], encoder_dict['height']))

    with torch.no_grad():
        for data in dataloader:
            input_color = data[("color", 0, 0)].cuda()

            output = depth_decoder(encoder(input_color))

            pred_disp, _ = disp_to_depth(output[("disp", 0)], 0.1, 100)
            pred_disp = pred_disp.cpu()[:, 0].numpy()


            pred_disps.append(pred_disp)

    pred_disps = np.concatenate(pred_disps)
    gt_path = os.path.join( "./gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    print("-> Evaluating")

    print("   Stereo evaluation - "
          "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
    disable_median_scaling = True
    pred_depth_scale_factor = STEREO_SCALE_FACTOR


    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp


        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)


        # mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= pred_depth_scale_factor
        if not disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate.')
    parser.add_argument('--load_weights_folder', type=str, help='the path of model',default='./model')
    parser.add_argument('--test_file', type=str, help='the path of file', default="./Eigen_test_files.txt")
    parser.add_argument('--data_path', type=str, help='the path to the images')
    args = parser.parse_args()

    evaluate(args)













