'''
Hyperparameters wrapped in argparse
This file contains most of tuanable parameters for this homework


You can change the values by changing their default fields or by command-line
arguments. For example, "python q2_1_4.py --sigma 0.15 --ratio 0.7"
'''

import argparse

def get_opts():
    parser = argparse.ArgumentParser(description='16-720 HW2: Homography')

    ## Feature detection (requires tuning)
    parser.add_argument('--sigma', type=float, default=0.15,
                        help='threshold for corner detection using FAST feature detector')
    parser.add_argument('--ratio', type=float, default=0.7,
                        help='ratio for BRIEF feature descriptor')

    ## Ransac (requires tuning)
    parser.add_argument('--max_iters', type=int, default=500,
                        help='the number of iterations to run RANSAC for')
    parser.add_argument('--inlier_tol', type=float, default=2.0,
                        help='the tolerance value for considering a point to be an inlier')

    ## Additional options (add your own hyperparameters here)
    # whether to show the warpped image
    parser.add_argument('--show_warpped_img', action='store_true', default=False,
                        help='whether to show the warpped image from HarryPotterize.py')
    
    # Select the best feature point representation
    parser.add_argument('--match_method', type=str, choices=['BRIEF_FAST', 'SIFT', 'ORB'], default='BRIEF_FAST',
                        help='the method to match features across images')

    ## Q2.2.5 RANSAC parameters ablation study
    parser.add_argument('--store_Q225_img', action='store_true', default=False,
                        help='whether to store img at ../Figures/composite_{opts.max_iters}_{opts.inlier_tol}.png')

    ## Q3.1 AR
    parser.add_argument('--crop_video', action='store_true', default=False,
                        help='whether to generate cropped video')
    parser.add_argument('--ar_crop_vid_path', type=str, default='../data/cropped_ar_source.avi',
                        help='save path to the cropped source video')
    parser.add_argument('--ar_src_vid_path', type=str, default='../data/ar_source.mov',
                        help='path to the source video')
    parser.add_argument('--ar_tgt_vid_path', type=str, default='../data/book.mov',
                        help='path to the target video')
    parser.add_argument('--ar_img_path', type=str, default='../data/cv_cover.jpg',
                        help='path to the reference image')
    parser.add_argument('--ar_out_dir', type=str, default='../result/',
                        help='dir to the output video and frames')
    parser.add_argument('--ar_out_vid', type=str, default='ar.avi',
                        help='output video name')

    ##
    opts = parser.parse_args()

    return opts
