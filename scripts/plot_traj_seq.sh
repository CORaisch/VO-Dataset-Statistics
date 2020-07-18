#!/bin/bash

evo_traj kitti -p --plot_mode xz --ref=/home/claudio/Cluster/Datasets/KITTI/poses_gt/${1}.txt /home/claudio/Cluster/DeepVO-pytorch/experiments/kitti_test2/test_on_KITTI/out_${1}.txt /home/claudio/Cluster/DeepVO-pytorch/experiments/kitti_test2_seqlen_10_15/test_on_KITTI/out_${1}.txt

echo "kitti_test2_seqlen_10_15:"
evo_rpe kitti /home/claudio/Cluster/Datasets/KITTI/poses_gt/${1}.txt /home/claudio/Cluster/DeepVO-pytorch/experiments/kitti_test2_seqlen_10_15/test_on_KITTI/out_${1}.txt

echo "kitti_test2:"
evo_rpe kitti /home/claudio/Cluster/Datasets/KITTI/poses_gt/${1}.txt /home/claudio/Cluster/DeepVO-pytorch/experiments/kitti_test2/test_on_KITTI/out_${1}.txt
