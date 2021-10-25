#!/usr/bin/env python3

import argparse
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import skvideo.io               # pip install scikit-video

from pydmd import DMD

parser = argparse.ArgumentParser(description='virtual choir')
parser.add_argument('video', help='video file')
parser.add_argument('--scale', type=float, default=1.0, help='scale input')
parser.add_argument("--skip-frames", type=int, default=0)
parser.add_argument('--write', action='store_true', help='write out video')
args = parser.parse_args()

# pathname work
abspath = os.path.abspath(args.video)
filename, ext = os.path.splitext(abspath)
dirname = os.path.dirname(args.video)
mode_video = filename + "_modes.mp4"

metadata = skvideo.io.ffprobe(args.video)
#print(metadata.keys())
print(json.dumps(metadata["video"], indent=4))
fps_string = metadata['video']['@avg_frame_rate']
(num, den) = fps_string.split('/')
fps = float(num) / float(den)
codec = metadata['video']['@codec_long_name']
#w = int(round(int(metadata['video']['@width']) * scale))
#h = int(round(int(metadata['video']['@height']) * scale))
if "@duration" in metadata["video"]:
    total_frames = int(round(float(metadata['video']['@duration']) * fps))
else:
    total_frames = 1

print('fps:', fps)
print('codec:', codec)
#print('output size:', w, 'x', h)
print('total frames:', total_frames)

print("Opening ", args.video)
reader = skvideo.io.FFmpegReader(args.video, inputdict={}, outputdict={})

rows = 3
cols = 3
max_rank = ((rows*cols) * 2) - 2
print("max rank:", max_rank)

# process video at this scale factor
scale = args.scale
dmd_size = 200
window_size = 64

def draw_mode(label, mode, shape):
    real = np.abs(mode.real)
    equalized = 255 * (real / np.max(real))
    (h, w) = shape[:2]
    big = cv2.resize(np.flipud(equalized.reshape((dmd_size,dmd_size)).astype('uint8')), (w, h), interpolation=cv2.INTER_AREA)
    cv2.imshow(label, big)
    return big

x = None
y = None
norm_x = None
norm_y = None
primed = False
counter = 0

X = []
dmd = DMD(svd_rank=max_rank)

inputdict = {
    '-r': str(fps)
}
lossless = {
    # See all options: https://trac.ffmpeg.org/wiki/Encode/H.264
    '-vcodec': 'libx264',  # use the h.264 codec
    '-crf': '0',           # set the constant rate factor to 0, (lossless)
    '-preset': 'veryslow', # maximum compression
    '-r': str(fps)         # match input fps
}
sane = {
    # See all options: https://trac.ffmpeg.org/wiki/Encode/H.264
    '-vcodec': 'libx264',  # use the h.264 codec
    '-crf': '17',          # visually lossless (or nearly so)
    '-preset': 'medium',   # default compression
    '-r': str(fps)         # match input fps
}
if args.write:
    mode_writer = skvideo.io.FFmpegWriter(mode_video, inputdict=inputdict,
                                          outputdict=sane)

print("collecting video frames")
for frame in reader.nextFrame():
    counter += 1
    if counter <= args.skip_frames:
        continue
    
    frame = frame[:,:,::-1]     # convert from RGB to BGR (to make opencv happy)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow("input", scaled)

    small = cv2.resize(scaled, (dmd_size,dmd_size), interpolation=cv2.INTER_AREA)

    #X.append( gray.flatten() )
    X.append( np.flipud(small) )

    while len(X) > window_size:
        del X[0]

    dmd.fit(np.array(X))
    print(dmd.modes.shape)
    if len(dmd.eigs):
        #print(dmd.eigs)
        idx = np.argsort(np.abs(dmd.eigs-1))
        #idx = np.argsort(np.abs(dmd.eigs.imag))
        print(idx)
        print(dmd.eigs)
        print(dmd.eigs[idx[0]])
        print(dmd.reconstructed_data.shape)

        #for i in range(len(idx)):
        #    draw_mode("freq index: %d" % i, dmd.modes[:,idx[i]], scaled.shape)

        big = 255 * dmd.reconstructed_data[:,-1] / np.max(dmd.reconstructed_data[:,-1]) # avoid overflow
        big = cv2.resize(np.flipud(big.reshape((dmd_size,dmd_size)).astype('uint8')), (scaled.shape[1], scaled.shape[0]), interpolation=cv2.INTER_AREA)
        big = 255 * ( big / np.max(big) )
        cv2.imshow("reconstructed", big.astype('uint8'))
        
        def draw_text(img, label, x, y, subscale=1.0, just="center"):
            font_scale = subscale * h / 700
            size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                   font_scale, 1)
            if just == "center":
                locx = int(x - size[0][0]*0.5)
                locy = int(y + size[0][1]*1.5)
            elif just == "lower-right":
                locx = int(x - size[0][0])
                locy = int(y - size[0][1])

            cv2.putText(img, label, (locx, locy),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                        1, cv2.LINE_AA)

        (h, w) = scaled.shape[:2]
        grid = np.zeros( (h*rows, w*cols) ).astype('uint8')
        grid[0:h,0:w] = scaled
        draw_text(grid, "Original", w*0.5, 0)
        r = 0
        c = 1
        for i in range(0, max_rank, 2):
            if i >= len(idx):
                break
            #print(i)
            if c >= cols:
                r += 1
                c = 0
            #print("grid:", r, c, "i:", i)
            grid[r*h:(r+1)*h,c*w:(c+1)*w] = draw_mode("a", dmd.modes[:,idx[i]], scaled.shape)
            #grid[r*h:(r+1)*h,c*w:(c+1)*w] = scaled
            eig = dmd.eigs[idx[i]]
            label = "Mode: %d (%.4f + %.4fj)" % (i, eig.real, eig.imag)
            draw_text(grid, label, (c+0.5)*w, r*h)
            c += 1
        draw_text(grid, "www.uav.aem.umn.edu", w*(rows-0.03), h*(cols-0.03), just="lower-right")
        cv2.imshow("grid", grid)
        if args.write:
            mode_writer.writeFrame(grid)
        
        if 0xFF & cv2.waitKey(1) == 27:
            break

if False:
    (h, w) = X[0].shape
    print(w,h)

    print("running dmd")
    dmd = DMD(svd_rank=5)
    dmd.fit(np.array(X))

    dmd.plot_modes_2D(figsize=(12,5))

    print(X[0].shape[0])
    x1 = np.array(list(range(w)))
    x2 = np.array(list(range(h)))
    x1grid, x2grid = np.meshgrid(x1, x2)
    fig = plt.figure(figsize=(18,12))
    for id_subplot, snapshot in enumerate(dmd.reconstructed_data.T[:16], start=1):
        plt.subplot(4, 4, id_subplot)
        plt.pcolor(x1grid, x2grid, snapshot.reshape(x1grid.shape).real, vmin=-1, vmax=1)

    for eig in dmd.eigs:
        print('Eigenvalue {}: distance from unit circle {}'.format(eig, np.abs(eig.imag**2+eig.real**2 - 1)))

    dmd.plot_eigs(show_axes=True, show_unit_circle=True)

    for mode in dmd.modes.T:
        plt.plot(mode.real)
        plt.title('Modes')
    plt.show()

    for dynamic in dmd.dynamics:
        plt.plot(dynamic.real)
        plt.title('Dynamics')
    plt.show()

    fig = plt.figure(figsize=(17,6))

    for n, mode, dynamic in zip(range(131, 133), dmd.modes.T, dmd.dynamics):
        plt.subplot(n)
        plt.pcolor(x1grid, x2grid, (mode.reshape(-1, 1).dot(dynamic.reshape(1, -1))).real.T)

    plt.subplot(133)
    plt.pcolor(x1grid, x2grid, dmd.reconstructed_data.T.real)
    plt.colorbar()

    plt.show()

