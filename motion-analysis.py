#!/home/claudio/Apps/anaconda3/envs/PyTorch/bin/python
# bultins
import sys, os, glob, argparse
sys.path.append('..')
from math import ceil, pi
from pathlib import Path
# project dependencies
from thirdparty.data_helper import get_data_info
# external dependencies
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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

# parse arguments
argparser = argparse.ArgumentParser(description="computes a histogram over quantized motion directions retrieved from pose-sequences.")
argparser.add_argument('out', type=str, help="path where results will be saved")
argparser.add_argument('dataset', type=str, help="dataset base directory")
argparser.add_argument('sequences', type=str, nargs='+', help="video indices to transform")
argparser.add_argument('--filename', '-fname', type=str, default='plot', help="filename of rendered plots")
argparser.add_argument('--batch_size', '-bs', type=int, default=8, help="batch size for transforming (default: 8)")
argparser.add_argument('--seq_len', '-sl', type=int, default=2, help="length of sub-sequences to sample from main sequence. The direction will be computed from first to last pose in the sub-sequence. (default: 2)")
argparser.add_argument('--yaw', '-y', type=float, default=5, help="reference yaw angle in deg (default: 20 deg)")
argparser.add_argument('--pitch', '-p', type=float, default=2, help="reference pitch angle in deg (default: 10 deg)")
argparser.add_argument('--scanlines', '-scans', type=int, default=3, help="number of steps along pitch (default: 3)")
argparser.add_argument('--steps', '-steps', type=int, default=3, help="number of steps along yaw (default: 3)")
args = argparser.parse_args()

if __name__ == '__main__':
    # set dataset to test on
    image_dir = os.path.join(args.dataset, 'images')
    pose_dir = os.path.join(args.dataset, 'poses_gt')

    # prepare directory structure
    Path(args.out).mkdir(parents=True, exist_ok=True)
    f_out = os.path.join(args.out, args.filename)

    # prepare dataset
    n_workers = 1
    seq_len = args.seq_len
    overlap = seq_len - 1
    print('seq_len = {},  overlap = {}'.format(seq_len, overlap))

    # prepare histograms
    v_forward = np.matrix([[0.0],[0.0],[1.0]], dtype=np.float)
    y = args.yaw; p = args.pitch;
    ref_eulers = [ [ (p,y) for y in np.linspace(args.yaw*args.steps, -args.yaw*args.steps, 2*args.steps+1)] for p in np.linspace(args.pitch*args.scanlines, -args.pitch*args.scanlines, 2*args.scanlines+1) ]
    # ref_eulers = [[(p,y), (p,0), (p,-y)], [(0,y), (0,0), (0,-y)], [(-p,y), (-p,0), (-p,-y)]]
    ref_dirs = [ [ euler_to_mat(to_rad(x+(0,)))*v_forward for x in sub_eulers ] for sub_eulers in ref_eulers ]
    histogram = np.zeros((2*args.scanlines+1, 2*args.steps+1), dtype=float)
    accum_euler = np.array([0,0,0], dtype=np.float64)
    accum_coord = np.array([0,0,0], dtype=np.float64)

    # loop over sequences
    seq_motions, seq_directions, seq_translations, seq_distances = [], [], [], []
    for seq in args.sequences:
        n_poses = len(glob.glob(os.path.join(image_dir, seq, '*.png')))
        print('exp. #sub-sequences = {}, exp. #batches = {}'.format(n_poses-overlap, ceil((n_poses-overlap)/args.batch_size)))

        # create sub-sequenced dataset
        df = get_data_info(image_dir, pose_dir, folder_list=[seq], seq_len_range=[seq_len, seq_len], overlap=overlap, sample_times=1, shuffle=False, sort=False)
        dataset = AbsolutePoseSequenceDataset(df)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=n_workers)

        # loop over sequence
        n_batch = len(dataloader)

        for i, batch in enumerate(dataloader):
            # NOTE batch: tensor of rank Bx(S-1)x6
            print('{} / {}'.format(i+1, n_batch), end='\r', flush=True)
            # for all further predictions only integrate the last pose, since overlap=seq_len-1
            batch = batch.numpy()
            for j, seq in enumerate(batch):
                # compute relative motion from start to end
                T_s = np.matrix(seq[0])
                T_e = np.matrix(seq[-1])
                T = T_e * inv(T_s); R = T[:3,:3]; t = T[:3,3];
                # rotate forward unit direction by relative sequence transform
                d = R * v_forward
                # store motion data for plotting raw distribution
                p_s = T_s * np.matrix([[0.0],[0.0],[0.0],[1.0]], dtype=np.float)
                p_e = T_e * np.matrix([[0.0],[0.0],[0.0],[1.0]], dtype=np.float)
                length = dist(T_e[:3,3],T_s[:3,3])
                seq_distances.append(length); seq_directions.append(d); seq_motions.append(T); seq_translations.append(t);
                # map to ref direction and integrate histogram
                dists = np.array([ [(d.T*r)[0,0] for r in sub_dirs ] for sub_dirs in ref_dirs ])
                histogram[np.unravel_index(np.argmax(dists, axis=None), dists.shape)] += 1
                # accumulate euler angles
                accum_euler += np.absolute(mat_to_euler(R))
                # accumulate coordinates
                accum_coord += np.absolute([t[0,0],t[1,0],t[2,0]])

    # save params
    with open(f_out + '_params.txt', 'w') as f:
        f.write('Dataset: '+args.dataset+'\n')
        f.write('Sequences: '+', '.join(args.sequences)+'\n')
        f.write('Sequence Length: '+str(args.seq_len)+'\n')
        f.write('Scanlines: '+str(args.scanlines)+'\n')
        f.write('Steps: '+str(args.steps)+'\n')
        f.write('Yaw: '+str(args.yaw)+'\n')
        f.write('Pitch: '+str(args.pitch)+'\n')

    # 3D plot
    fig3d = plt.figure()
    ax = fig3d.add_subplot(1,1,1, projection='3d')
    dirs = np.asarray(seq_directions)
    # dirs[:,0,0] *= -1; dirs[:,1,0] *= -1;
    marker_color = '#FF9F1C'
    ax.scatter(-dirs[:,0,0], dirs[:,1,0], dirs[:,2,0], c=marker_color, marker='o', alpha=0.5)
    d = np.asarray(ref_dirs); z = np.asarray([0]*d.shape[0]*d.shape[1]);
    ax.quiver(z,z,z,d[:,:,0,0].flatten(),d[:,:,1,0].flatten(),d[:,:,2,0].flatten(), normalize=True, arrow_length_ratio=0.1, length=0.75, color=marker_color, alpha=0.5)
    ax.quiver(0,0,0,-1,0,0, normalize=True, arrow_length_ratio=0.1, length=0.25, color='r')
    ax.quiver(0,0,0,0,1,0, normalize=True, arrow_length_ratio=0.1, length=0.25, color='g')
    ax.quiver(0,0,0,0,0,1, normalize=True, arrow_length_ratio=0.1, length=0.25, color='b')
    ax.set_xlabel('X'); ax.set_xlim([-1.0,1.0]);
    ax.set_ylabel('Y'); ax.set_ylim([-1.0,1.0]);
    ax.set_zlabel('Z'); ax.set_zlim([ 0.0,1.0]);
    ax.view_init(elev=-45.0, azim=135.0)

    plt.savefig(f_out + '_3D_dist.png')

    print('gathered {} datapoints in total'.format(len(seq_motions)))
    print('render plots...')
    # 2D trajectory plots (projections)
    fig2d = plt.figure()
    # YX view
    ax = fig2d.add_subplot(1,3,1)
    ax.scatter(dirs[:,0,0], dirs[:,1,0], c=marker_color, alpha=0.5, marker='o')
    ax.quiver(0,0,1,0, color='r', scale=4.0)
    ax.quiver(0,0,0,1, color='g', scale=4.0)
    # ZX view
    ax = fig2d.add_subplot(1,3,2)
    ax.scatter(dirs[:,0,0], dirs[:,2,0], c=marker_color, alpha=0.5, marker='o')
    ax.quiver(0,0,1,0, color='r', scale=4.0)
    ax.quiver(0,0,0,1, color='b', scale=4.0)
    # YZ view
    ax = fig2d.add_subplot(1,3,3)
    ax.scatter(dirs[:,2,0], dirs[:,1,0], c=marker_color, alpha=0.5, marker='o')
    ax.quiver(0,0,1,0, color='b', scale=4.0)
    ax.quiver(0,0,0,1, color='g', scale=4.0)

    plt.savefig(f_out + '_2D_dist.png')

    # 2D angle plots
    figAngle = plt.figure()
    ax = figAngle.add_subplot(1,1,1)
    #  normalize histogram
    max_val = max(accum_euler); accum_euler = [ x/max_val for x in accum_euler ];
    ax.bar(['roll', 'pitch', 'yaw'], [accum_euler[2], accum_euler[0], accum_euler[1]], color=marker_color)

    plt.savefig(f_out + '_2D_angles.png')

    # 2D translation plots
    figTrans = plt.figure()
    ax = figTrans.add_subplot(1,1,1)
    #  normalize histogram
    max_val = max(accum_coord); accum_coord = [ x/max_val for x in accum_coord ];
    ax.bar(['x', 'y', 'z'], accum_coord, color=marker_color)

    plt.savefig(f_out + '_2D_trans.png')

    # histogram plot for directions
    figHist = plt.figure()
    ax = figHist.add_subplot(111, projection='3d')
    # normalize histogram
    norm_val = np.sum(histogram)
    histogram = np.array([ [ x/norm_val for x in row ] for row in histogram ])
    for i, p in enumerate(np.linspace(args.pitch*args.scanlines, -args.pitch*args.scanlines, 2*args.scanlines+1)):
        xs = np.linspace(args.yaw*args.steps, -args.yaw*args.steps, 2*args.steps+1)
        ys = histogram[i]
        ax.bar(xs, ys, zs=p, zdir='y', color=marker_color, alpha=0.8)
    ax.set_xlabel('Yaw'); ax.set_ylabel('Pitch'); ax.set_zlabel('Relative Frequency');

    plt.savefig(f_out + '_dir_hist.png')

    # histogram plot for distances
    figHistDist = plt.figure()
    ax = figHistDist.add_subplot()
    ax.hist(seq_distances, bins=10, color=marker_color, align='mid', rwidth=0.8, density=True)
    ax.set_xlabel('Meter\nKm/h'); ax.set_ylabel('Relative Frequency');
    labels = np.arange(0,3*(args.seq_len-1),1)
    plt.xticks(labels, [ str(l)+'\n'+str(int(l*3.6/(0.1*(seq_len-1)))) for l in labels ])

    plt.savefig(f_out + '_dist_hist.png')

    # # reference directions plot
    # figRef = plt.figure()
    # ax = figRef.add_subplot(1,1,1, projection='3d')
    # d = np.asarray(ref_dirs); z = np.asarray([0]*d.shape[0]);
    # ax.quiver(z,z,z,d[:,0,0],d[:,1,0],d[:,2,0], normalize=True, arrow_length_ratio=0.1, length=0.25, color=marker_color)
    # ax.set_xlabel('X'); ax.set_xlim([-0.5,0.5]);
    # ax.set_ylabel('Y'); ax.set_ylim([-0.5,0.5]);
    # ax.set_zlabel('Z'); ax.set_zlim([ 0.0,0.5]);
    # ax.view_init(elev=-45.0, azim=135.0)

    plt.show()
