#!/home/claudio/Apps/anaconda3/envs/PyTorch/bin/python
# bultins
import sys, os, glob, argparse
sys.path.append('..')
from math import ceil, pi
from pathlib import Path
# project dependencies
from thirdparty.data_helper import get_data_info, create_pose_data
# external dependencies
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class Params():
    def __init__(self, scanlines, steps, yaw, pitch, dist_th, dir_th, seq_len, batch_size):
        self.scans = scanlines; self.steps = steps; self.yaw = yaw;
        self.pitch = pitch; self.seq_len = seq_len; self.batch_size = batch_size;
        self.dist_th = dist_th; self.dir_th = dir_th;
        self.bins = np.linspace(0.0, 3*(self.seq_len-1), 15)

class AbsolutePoseSequenceDataset(Dataset):
    '''yields sequences of absolute poses in 4x4 matrix format (SE(3))'''

    def __init__(self, info_dataframe):
        self.data_info = info_dataframe
        self.seq_len_list = list(self.data_info.seq_len)
        self.groundtruth_arr = np.asarray(self.data_info.pose)

    def _to_mat(self, arr, R):
        '''arr: [(r_0, r_1, r_2, t_0, t_1, t_2), R], |arr[0]| = 6, R: 3x3 matrix'''
        t = np.matrix(arr[3:]).reshape((3,1))
        T = np.matrix(np.eye(4, dtype=R.dtype))
        T[:3,:3] = R; T[:3,3] = t;
        return T

    def __getitem__(self, index):
        seq_raw = np.hsplit(self.groundtruth_arr[index], np.array([6]))
        seq_len = seq_raw[0].shape[0]
        seq_abs = [ np.asarray(self._to_mat(seq_raw[0][i], seq_raw[1][i].reshape((3,3)))) for i in range(seq_len) ]
        return torch.FloatTensor(seq_abs)

    def __len__(self):
        return len(self.data_info.index)

def inv(T):
    '''T: [R|t] as 4x4 matrix, inv: [R.T|-R.T*t] as 4x4 matrix'''
    Inv = np.matrix(np.eye(4, dtype=T.dtype))
    R = T[:3,:3]; t = T[:3,3]
    Inv[:3,:3] = R.T; Inv[:3,3] = -R.T*t;
    return Inv

def euler_to_mat(theta):
    '''theta: array containing euler angles about x, y and z axis (in that order)'''
    R_x = np.array([[1,         0,                  0               ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0               ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                  1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return np.matrix(R)

def mat_to_euler(R):
        import math
        # NOTE code taken from: https://www.learnopencv.com/rotation-matrix-to-euler-angles
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z]) # returns [pitch, yaw, roll]

def dist(a, b):
    return np.linalg.norm(a-b)

def to_rad(a):
    ''''a: tuple input of floats'''
    return tuple( x * pi/180.0 for x in a )

def plot_dir_error_distributions(f_out, params, e_hist, err_hist, t_hist):
    # normalize histograms at center yaw location
    yid = [params.scans][0]
    err_hist_y = err_hist[yid].copy(); e_hist_y = e_hist[yid].copy(); t_hist_y = t_hist[yid].copy();
    err_hist_y /= e_hist_y; err_hist_y = [ x if np.isfinite(x) else 0.0 for x in err_hist_y ];
    t_hist_y /= np.sum(t_hist_y); e_hist_y /= np.sum(e_hist_y); err_hist_y /= np.sum(err_hist_y);

    # plot pitch==0, yaw[:]
    fig = plt.figure()
    ax = fig.add_subplot(211)
    # plot for pitch==0 -> hist[params.scans]
    xs = np.linspace(params.yaw*params.steps, -params.yaw*params.steps, 2*params.steps+1)
    w = (xs[1]-xs[0])*0.4
    ax.bar(xs-w, t_hist_y, width=w, align='edge', color='#0000FF', alpha=0.8)
    ax.bar(xs, e_hist_y, width=w, align='edge', color='#00FF00', alpha=0.8)
    ax.plot(xs, err_hist_y, '--', color='#FF0000', alpha=0.8)
    ax.set_xlabel('Yaw'); ax.set_ylabel('Relative Frequency');
    # set legend
    import matplotlib.patches as mpatches
    train_patch = mpatches.Patch(color='#0000FF', label='Train Distribution')
    eval_patch = mpatches.Patch(color='#00FF00', label='Eval Distribution')
    error_patch = mpatches.Patch(color='#FF0000', label='Error Distribution')
    plt.legend(handles=[train_patch, eval_patch, error_patch])
    plt.yscale('log')

    # plot yaw==0, pitch[:]
    ax = fig.add_subplot(212)
    # normalize histograms at center pitch location
    pid = [params.steps][0]
    err_hist_p = err_hist[:,pid].copy(); e_hist_p = e_hist[:,pid].copy(); t_hist_p = t_hist[:,pid].copy();
    err_hist_p /= e_hist_p; err_hist_p = [ x if np.isfinite(x) else 0.0 for x in err_hist_p ];
    t_hist_p /= np.sum(t_hist_p); e_hist_p /= np.sum(e_hist_p); err_hist_p /= np.sum(err_hist_p);
    # plot for yaw==0 -> hist[:,steps]
    xs = np.linspace(params.pitch*params.scans, -params.pitch*params.scans, 2*params.scans+1)
    w = (xs[1]-xs[0])*0.4
    ax.bar(xs-w, t_hist_p, width=w, align='edge', color='#0000FF', alpha=0.8)
    ax.bar(xs, e_hist_p, width=w, align='edge', color='#00FF00', alpha=0.8)
    ax.plot(xs, err_hist_p, '--', color='#FF0000', alpha=0.8)
    ax.set_xlabel('Pitch'); ax.set_ylabel('Relative Frequency');
    # set legend
    import matplotlib.patches as mpatches
    train_patch = mpatches.Patch(color='#0000FF', label='Train Distribution')
    eval_patch = mpatches.Patch(color='#00FF00', label='Eval Distribution')
    error_patch = mpatches.Patch(color='#FF0000', label='Error Distribution')
    plt.legend(handles=[train_patch, eval_patch, error_patch])
    plt.yscale('log')

    plt.savefig(f_out + '_dir_hist.png')

def plot_dist_error_distributions(f_out, params, e_dist_hist, err_dist_hist, t_dist_hist):
    # prepare figure
    fig = plt.figure()
    ax = fig.add_subplot()

    # norm hist data
    err_dist_hist /= e_dist_hist; err_dist_hist = [ x if np.isfinite(x) else 0.0 for x in err_dist_hist ];
    t_dist_hist /= np.sum(t_dist_hist); e_dist_hist /= np.sum(e_dist_hist); err_dist_hist /= np.sum(err_dist_hist);

    # plot training histogram
    w = (params.bins[1]-params.bins[0])*0.4
    ax.bar(params.bins-w, t_dist_hist, width=w, align='edge', color='#0000FF', alpha=0.8)
    # plot eval histogram
    ax.bar(params.bins, e_dist_hist, width=w, align='edge', color='#00FF00', alpha=0.8)
    # plot error histogram
    err_dist_hist /= np.sum(err_dist_hist)
    # ax.plot(0.5*(params.bins[1:]+params.bins[:-1]), err_dist_hist[:-1], '--', color='#FF0000', alpha=0.8)
    ax.plot(params.bins, err_dist_hist, '--', color='#FF0000', alpha=0.8)

    # set labels
    ax.set_xlabel('Meter\nKm/h'); ax.set_ylabel('Relative Frequency');
    # labels = np.arange(0,3*(params.seq_len-1),1)
    plt.xticks(params.bins, [ str('{:.1f}'.format(l))+'\n'+str(int(l*3.6/(0.1*(params.seq_len-1)))) for l in params.bins ])

    # set legend
    import matplotlib.patches as mpatches
    train_patch = mpatches.Patch(color='#0000FF', label='Train Distribution')
    eval_patch = mpatches.Patch(color='#00FF00', label='Eval Distribution')
    error_patch = mpatches.Patch(color='#FF0000', label='Error Distribution')
    plt.legend(handles=[train_patch, eval_patch, error_patch])

    # save and show plot
    plt.savefig(f_out + '_dist_hist.png')

def compute_dataset_distributions(ds, sequences, params):
    # prepare dataset
    image_dir = os.path.join(ds, 'images')
    pose_dir = os.path.join(ds, 'poses_gt')
    n_workers = 1; overlap = params.seq_len - 1;
    print('seq_len = {},  overlap = {}'.format(params.seq_len, overlap))

    # prepare histograms
    v_forward = np.matrix([[0.0],[0.0],[1.0]], dtype=np.float)
    ref_eulers = [ [ (p,y) for y in np.linspace(params.yaw*params.steps, -params.yaw*params.steps, 2*params.steps+1)] for p in np.linspace(params.pitch*params.scans, -params.pitch*params.scans, 2*params.scans+1) ]
    ref_dirs = [ [ euler_to_mat(to_rad(x+(0,)))*v_forward for x in sub_eulers ] for sub_eulers in ref_eulers ]
    histogram = np.zeros((2*params.scans+1, 2*params.steps+1), dtype=float)
    dist_hist = np.zeros(params.bins.shape, dtype=float)
    accum_euler = np.array([0,0,0], dtype=np.float64); accum_coord = np.array([0,0,0], dtype=np.float64);

    # loop over sequences
    seq_directions = []
    for seq in sequences:
        n_poses = len(glob.glob(os.path.join(eval_image_dir, seq, '*.png')))
        print('exp. #sub-sequences = {}, exp. #batches = {}'.format(n_poses-overlap, ceil((n_poses-overlap)/params.batch_size)))

        # create sub-sequenced dataset
        df = get_data_info(image_dir, pose_dir, folder_list=[seq], seq_len_range=[params.seq_len, params.seq_len], overlap=overlap, sample_times=1, shuffle=False, sort=False)
        dataset = AbsolutePoseSequenceDataset(df)
        dataloader = DataLoader(dataset, batch_size=params.batch_size, drop_last=False, shuffle=False, num_workers=n_workers)

        # loop over sequence
        n_batch = len(dataloader)

        for i, batch in enumerate(dataloader):
            # NOTE batch: tensor of rank Bx(S-1)x6
            print('{} / {}'.format(i+1, n_batch), end='\r', flush=True)
            # for all further predictions only integrate the last pose, since overlap=seq_len-1
            batch = batch.numpy()
            for j, seq in enumerate(batch):
                # compute relative motion from start to end
                T_s = np.matrix(seq[0]); T_e = np.matrix(seq[-1]);
                T = T_e * inv(T_s); R = T[:3,:3]; t = T[:3,3];
                # rotate forward unit direction by relative sequence transform
                d = R * v_forward
                # store motion data for plotting raw distribution
                p_s = T_s * np.matrix([[0.0],[0.0],[0.0],[1.0]], dtype=np.float)
                p_e = T_e * np.matrix([[0.0],[0.0],[0.0],[1.0]], dtype=np.float)
                length = dist(T_e[:3,3],T_s[:3,3])
                dist_idx = np.digitize(length,params.bins)-1
                dist_hist[dist_idx] += 1
                seq_directions.append(d);
                # map to ref direction and integrate histogram
                dists = np.array([ [(d.T*r)[0,0] for r in sub_dirs ] for sub_dirs in ref_dirs ])
                histogram[np.unravel_index(np.argmax(dists, axis=None), dists.shape)] += 1
                # accumulate euler angles
                accum_euler += np.absolute(mat_to_euler(R))
                # accumulate coordinates
                accum_coord += np.absolute([t[0,0],t[1,0],t[2,0]])

    # save params
    with open(f_out + '_params.txt', 'w') as f:
        f.write('Dataset: '+ds+'\n')
        f.write('Sequences: '+', '.join(sequences)+'\n')
        f.write('Sequence Length: '+str(params.seq_len)+'\n')
        f.write('Scanlines: '+str(params.scans)+'\n')
        f.write('Steps: '+str(params.steps)+'\n')
        f.write('Yaw: '+str(params.yaw)+'\n')
        f.write('Pitch: '+str(params.pitch)+'\n')

    return histogram, dist_hist, seq_directions, accum_coord, accum_euler

def compute_error_distributions(gt, estimates, sequences, params):
    # prepare dataset
    image_dir = os.path.join(gt, 'images')
    pose_dir = os.path.join(gt, 'poses_gt')
    n_workers = 1; overlap = params.seq_len - 1;
    print('seq_len = {},  overlap = {}'.format(params.seq_len, overlap))

    # prepare histograms
    v_forward = np.matrix([[0.0],[0.0],[1.0]], dtype=np.float)
    ref_eulers = [ [ (p,y) for y in np.linspace(params.yaw*params.steps, -params.yaw*params.steps, 2*params.steps+1)] for p in np.linspace(params.pitch*params.scans, -params.pitch*params.scans, 2*params.scans+1) ]
    ref_dirs = [ [ euler_to_mat(to_rad(x+(0,)))*v_forward for x in sub_eulers ] for sub_eulers in ref_eulers ]
    e_hist = np.zeros((2*params.scans+1, 2*params.steps+1), dtype=float); err_hist = np.zeros((2*params.scans+1, 2*params.steps+1), dtype=float);
    e_dist_hist = np.zeros(params.bins.shape, dtype=float); err_dist_hist = np.zeros(params.bins.shape, dtype=float);

    # loop over sequences
    for seq in sequences:
        n_poses = len(glob.glob(os.path.join(image_dir, seq, '*.png')))
        print('exp. #sub-sequences = {}, exp. #batches = {}'.format(n_poses-overlap, ceil((n_poses-overlap)/params.batch_size)))

        # create sub-sequenced eval dataloader
        df = get_data_info(image_dir, pose_dir, folder_list=[seq], seq_len_range=[params.seq_len, params.seq_len], overlap=overlap, sample_times=1, shuffle=False, sort=False)
        dataset = AbsolutePoseSequenceDataset(df)
        dataloader = DataLoader(dataset, batch_size=params.batch_size, drop_last=False, shuffle=False, num_workers=n_workers)

        # create sub-sequenced estimated dataloader
        create_pose_data([seq], estimates)
        est_df = get_data_info(image_dir, estimates, folder_list=[seq], seq_len_range=[params.seq_len, params.seq_len], overlap=overlap, sample_times=1, shuffle=False, sort=False)
        est_dataset = AbsolutePoseSequenceDataset(est_df)
        est_dataloader = DataLoader(est_dataset, batch_size=params.batch_size, drop_last=False, shuffle=False, num_workers=n_workers)

        # loop over sequence
        n_batch_eval = len(dataloader)
        n_batch_est = len(est_dataloader)

        for i, batch in enumerate(zip(dataloader, est_dataloader)):
            # NOTE batch: tensor of rank Bx(S-1)x6
            print('{} / {}'.format(i+1, n_batch_eval), end='\r', flush=True)
            # for all further predictions only integrate the last pose, since overlap=seq_len-1
            e_batch = batch[0].numpy(); est_batch = batch[1].numpy();
            for subseq in zip(e_batch, est_batch):
                # compute relative motion from start to end
                T_s = np.matrix(subseq[0][0]); T_e = np.matrix(subseq[0][-1]);
                T_s_est = np.matrix(subseq[1][0]); T_e_est = np.matrix(subseq[1][-1]);
                T = T_e * inv(T_s); R = T[:3,:3]; t = T[:3,3];
                T_est = T_e_est * inv(T_s_est); R_est = T_est[:3,:3]; t_est = T_est[:3,3];
                # rotate forward unit direction by relative sequence transform
                d = R * v_forward; d_est = R_est * v_forward;
                # store motion data for plotting raw distribution
                p_s = T_s * np.matrix([[0.0],[0.0],[0.0],[1.0]], dtype=np.float)
                p_e = T_e * np.matrix([[0.0],[0.0],[0.0],[1.0]], dtype=np.float)
                # map to ref direction and integrate eval histogram
                d_err = np.arccos(d.T * d_est); d_err = d_err if np.isfinite(d_err) else 0.0;
                dists = np.array([ [(d.T*r)[0,0] for r in sub_dirs ] for sub_dirs in ref_dirs ])
                idx = np.unravel_index(np.argmax(dists, axis=None), dists.shape)
                e_hist[idx] += 1
                err_hist[idx] += d_err
                # err_hist[idx] += 0 if d_err <= params.dir_th else 1
                # compute distance error and map to histogram bin
                length = dist(T_e[:3,3],T_s[:3,3]); est_length = dist(T_e_est[:3,3],T_s_est[:3,3]);
                error = np.abs(length-est_length)
                # accumulate GT and error distances
                dist_idx = np.digitize(length,params.bins)-1
                e_dist_hist[dist_idx] += 1
                err_dist_hist[dist_idx] += error
                # err_dist_hist[dist_idx] += 0 if error <= params.dist_th else 1

    # save params
    with open(f_out + '_params.txt', 'w') as f:
        f.write('Eval Dataset: '+gt+'\n')
        f.write('Estimates: '+estimates+'\n')
        f.write('Sequences: '+', '.join(sequences)+'\n')
        f.write('Sequence Length: '+str(params.seq_len)+'\n')
        f.write('Scanlines: '+str(params.scans)+'\n')
        f.write('Steps: '+str(params.steps)+'\n')
        f.write('Yaw: '+str(params.yaw)+'\n')
        f.write('Pitch: '+str(params.pitch)+'\n')

    return e_hist, e_dist_hist, err_hist, err_dist_hist

# parse arguments
argparser = argparse.ArgumentParser(description="computes a histogram over quantized motion directions retrieved from pose-sequences.")
argparser.add_argument('out', type=str, help="path where results will be saved")
argparser.add_argument('train_dataset', type=str, help="base directory of training dataset")
argparser.add_argument('eval_dataset', type=str, help="base directory of test dataset")
argparser.add_argument('estimates', type=str, help="base directory of model estimates (trajectories)")
argparser.add_argument('--train_sequences', '-t_seq', type=str, default=None, nargs='+', help="training video indices to transform")
argparser.add_argument('--eval_sequences', '-e_seq', type=str, default=None, nargs='+', help="test video indices to transform")
argparser.add_argument('--filename', '-fname', type=str, default='plot', help="filename of rendered plots")
argparser.add_argument('--batch_size', '-bs', type=int, default=8, help="batch size for transforming (default: 8)")
argparser.add_argument('--seq_len', '-sl', type=int, default=2, help="length of sub-sequences to sample from main sequence. The direction will be computed from first to last pose in the sub-sequence. (default: 2)")
argparser.add_argument('--yaw', '-y', type=float, default=5, help="reference yaw angle in deg (default: 20 deg)")
argparser.add_argument('--pitch', '-p', type=float, default=2, help="reference pitch angle in deg (default: 10 deg)")
argparser.add_argument('--scanlines', '-scans', type=int, default=3, help="number of steps along pitch (default: 3)")
argparser.add_argument('--steps', '-steps', type=int, default=3, help="number of steps along yaw (default: 3)")
argparser.add_argument('--dist_th', '-dth', type=float, default=1.0, help="distance threshold used for distance error distribution(default: 3)")
argparser.add_argument('--dir_th', '-dirth', type=float, default=1.0, help="direction threshold used for direction error distribution(default: 3)")
args = argparser.parse_args()

if __name__ == '__main__':
    if not (args.train_sequences and args.eval_sequences):
        print('you need to pass sequences for training, evaluation and estimates.')
        exit()

    # prepare dataset to test on
    eval_image_dir = os.path.join(args.eval_dataset, 'images')
    eval_pose_dir = os.path.join(args.eval_dataset, 'poses_gt')

    # prepare directory structure
    Path(args.out).mkdir(parents=True, exist_ok=True)
    f_out = os.path.join(args.out, args.filename)

    # make params
    params = Params(args.scanlines, args.steps, args.yaw, args.pitch, args.dist_th, args.dir_th, args.seq_len, args.batch_size)

    # compute distributions on training dataset
    t_hist, t_dist_hist, t_dirs, accum_coord, accum_euler = compute_dataset_distributions(args.train_dataset, args.train_sequences, params)

    # compute error distributions
    e_hist, e_dist_hist, err_hist, err_dist_hist = compute_error_distributions(args.eval_dataset, args.estimates, args.eval_sequences, params)

    # make plots
    plot_dist_error_distributions(f_out, params, e_dist_hist, err_dist_hist, t_dist_hist)
    plot_dir_error_distributions(f_out, params, e_hist, err_hist, t_hist)
    plt.show()
