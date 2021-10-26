#!/usr/bin/env python3

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skvideo.io               # pip install scikit-video

from pydmd import DMD

parser = argparse.ArgumentParser(description='virtual choir')
parser.add_argument('video', help='video file')
parser.add_argument("--skip-frames", type=int, default=0)
args = parser.parse_args()

print("Opening ", args.video)
reader = skvideo.io.FFmpegReader(args.video, inputdict={}, outputdict={})

# process video at this scale factor
scale = 0.5
dmd_size = 200
window_size = 32
max_rank = 7

def draw_mode(label, mode, shape):
    real = np.abs(mode.real)
    equalized = 255 * (real / np.max(real))
    (h, w) = shape[:2]
    big = cv2.resize(np.flipud(equalized.reshape((dmd_size,dmd_size)).astype('uint8')), (w, h), interpolation=cv2.INTER_AREA)
    cv2.imshow(label, big)
    
x = None
y = None
norm_x = None
norm_y = None
primed = False
counter = 0

X = []
dmd = DMD(svd_rank=max_rank)

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

        for i in range(len(idx)):
            draw_mode("freq index: %d" % i, dmd.modes[:,idx[i]], scaled.shape)

        big = 255 * dmd.reconstructed_data[:,-1] / np.max(dmd.reconstructed_data[:,-1]) # avoid overflow
        big = cv2.resize(np.flipud(big.reshape((dmd_size,dmd_size)).astype('uint8')), (scaled.shape[1], scaled.shape[0]), interpolation=cv2.INTER_AREA)
        big = 255 * ( big / np.max(big) )
        cv2.imshow("reconstructed", big.astype('uint8'))
        
        cv2.waitKey(1)
    
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

