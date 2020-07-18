#!/bin/bash

# call this second from ThinkPad computer

TARGET=/home/claudio/Fujitsu/Projects/CARLA-Recording/.temp_qma
# call preprocess
/home/claudio/Projects/DeepVO-pytorch/preprocess_deepvo.py ${TARGET} 00 --no_image_processing

# call motion analysis
/home/claudio/Projects/Dataset-Statistics/motion-analysis.py ${TARGET} ${TARGET} 00 -sl 5
