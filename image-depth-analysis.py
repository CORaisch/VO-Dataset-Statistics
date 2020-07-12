#!/home/claudio/Apps/anaconda3/envs/PyTorch/bin/python

import os, argparse
import numpy as np
from PIL import Image
from glob import glob
from pathlib import Path

# parse arguments
argparser = argparse.ArgumentParser(description="computes several statistics over the input images and depth data of the dataset.")
argparser.add_argument('out', type=str, help="path where results will be saved")
argparser.add_argument('dataset', type=str, help="dataset base directory")
argparser.add_argument('sequences', type=str, nargs='+', help="video indices to transform")
args = argparser.parse_args()

if __name__ == '__main__':
    # set dataset to test on
    image_dir = os.path.join(args.dataset, 'images')

    # prepare directory structure
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # loop over sequences
    seq_mean = []
    for seq in args.sequences:
        # prepare sequence
        im_paths = glob(os.path.join(image_dir, seq, '*.png'))
        n_images = len(im_paths)

        # compute mean image of sequence
        print('compute mean image:')
        im_size = np.asarray(Image.open(im_paths[0])).shape
        im_mean = np.zeros(im_size, dtype=np.float)
        for i, path in enumerate(im_paths):
            print('\t{} / {}'.format(i+1, n_images), end='\r', flush=True)
            im = np.asarray(Image.open(path))
            im_mean += im
        im_mean /= n_images
        seq_mean.append(im_mean)
        print(' done!')
        # im_mean = np.mean(np.asarray([ np.array(Image.open(path)) for path in im_paths ]), axis=0)

        # compute variance image
        print('compute variance image:')
        im_var = np.zeros_like(im_mean)
        for i, path in enumerate(im_paths):
            print('\t{} / {}'.format(i+1, n_images), end='\r', flush=True)
            im = np.asarray(Image.open(path))
            diff = (im - im_mean)
            im_var += np.multiply(diff, diff)
        im_var /= n_images
        print(' done!')


        ## beg DEBUG
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1,2,1)
        ax.imshow(im_mean.astype('int'))
        im_var = 255.0*(im_var-np.min(im_var))/(np.max(im_var)-np.min(im_var)) # scale to [0,255]
        ax = fig.add_subplot(1,2,2)
        ax.imshow(im_var.astype('int'))
        plt.show()
        ## end DEBUG

