# builtins
import os, glob, time
# external dependencies
import numpy as np
import pandas as pd
from thirdparty.utils import R_to_angle

def get_data_info(image_dir, pose_dir, folder_list, seq_len_range, overlap, sample_times=1, max_step=1, pad_y=False, shuffle=False, sort=True):
    # check inputs
    assert overlap < min(seq_len_range)
    assert max_step > 0
    # subsequene each sequence
    X_path, Y = [], []
    X_len = []
    for folder in folder_list:
        start_t = time.time()
        poses = np.load(os.path.join(pose_dir, '{}.npy'.format(folder))) # (n_images, 15)
        fpaths = glob.glob(os.path.join(image_dir, folder, '*.png'))
        fpaths.sort()
        # Random segment to sequences with diff lengths
        n_frames = len(fpaths)
        min_len, max_len = seq_len_range[0], seq_len_range[1]
        for i in range(sample_times):
            start = 0
            while True:
                n = np.random.random_integers(min_len, max_len)
                s = np.random.random_integers(1, max_step)
                if start + n*s <= n_frames:
                    x_seg = fpaths[start:start+n*s:s]
                    X_path.append(x_seg)
                    if not pad_y:
                        Y.append(poses[start:start+n*s:s])
                    else:
                        pad_zero = np.zeros((max_len-n, 15))
                        padded = np.concatenate((poses[start:start+n*s:s], pad_zero))
                        Y.append(padded.tolist())
                else:
                    print('Last %d frames is not used' %(start+(n*s)-n_frames))
                    break
                start += s * (n - overlap)
                X_len.append(len(x_seg))
        print('Folder {} finish in {} sec'.format(folder, time.time()-start_t))
    # Convert to pandas dataframes
    data = {'seq_len': X_len, 'image_path': X_path, 'pose': Y}
    df = pd.DataFrame(data, columns = ['seq_len', 'image_path', 'pose'])
    # Shuffle through all videos
    if shuffle:
        df = df.sample(frac=1)
    # Sort dataframe by seq_len
    if sort:
        df = df.sort_values(by=['seq_len'], ascending=False)
    return df

def create_pose_data(sequences, pose_dir):
    start_t = time.time()
    for seq in sequences:
        fn = '{}/out_{}.txt'.format(pose_dir, seq)
        if not os.path.exists(fn):
            continue
        print('Transforming {}...'.format(fn))
        with open(fn) as f:
            lines = [line.split('\n')[0] for line in f.readlines()]
            poses = [ R_to_angle([float(value) for value in l.split(' ')]) for l in lines] # list of pose (pose=list of 12 floats)
            poses = np.array(poses)
            np.save(pose_dir+'/{}.npy'.format(seq), poses)
            print('Sequence {}: shape={}'.format(seq, poses.shape))
    print('elapsed time = {}'.format(time.time()-start_t))
