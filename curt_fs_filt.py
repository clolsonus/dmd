#!/usr/bin/env python3

# fast/slow filter background tracker

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skvideo.io               # pip install scikit-video

parser = argparse.ArgumentParser(description='fs_filter')
parser.add_argument('video', help='video file')
parser.add_argument("--skip-frames", type=int, default=0)
args = parser.parse_args()

print("Opening ", args.video)
reader = skvideo.io.FFmpegReader(args.video, inputdict={}, outputdict={})

# process video at this scale factor
scale = 0.5

x = None
y = None
norm_x = None
norm_y = None
primed = False
counter = 0

fast_avg = None
slow_avg = None
diff_avg = None

fast_alpha = 0.4
slow_alpha = 0.4
diff_alpha = 0.001

last_gray = None

for frame in reader.nextFrame():
    counter += 1
    if counter <= args.skip_frames:
        continue
    
    frame = frame[:,:,::-1]     # convert from RGB to BGR (to make opencv happy)

    scaled = cv2.resize(frame, (0,0), fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    
    if fast_avg is None:
        fast_avg = gray.astype('float32')
    else:
        fast_avg = fast_avg*(1-fast_alpha) + gray.astype('float32')*fast_alpha
        
    if slow_avg is None:
        slow_avg = gray.astype('float32')
    else:
        slow_avg = slow_avg*(1-slow_alpha) + last_frame.astype('float32')*slow_alpha

    
    cv2.imshow("fast average", fast_avg.astype('uint8'))    
    cv2.imshow("slow average", slow_avg.astype('uint8'))
    
    diff = np.abs(slow_avg - fast_avg)
    if diff_avg is None:
        diff_avg = diff
    else:
        diff_avg = diff_avg*(1-diff_alpha) + diff*diff_alpha
    diff_max = np.max(diff_avg)
    cv2.imshow("diff average", (255*diff_avg/diff_max).astype('uint8'))

    last_frame = gray
    
    cv2.waitKey(1)
    

