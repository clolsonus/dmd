#!/usr/bin/env python3

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skvideo.io               # pip install scikit-video

from pydmd import DMD

parser = argparse.ArgumentParser(description='virtual choir')
parser.add_argument('video', help='video file')
parser.add_argument("--skip-frames", type=int)
args = parser.parse_args()

print("Opening ", args.video)
reader = skvideo.io.FFmpegReader(args.video, inputdict={}, outputdict={})

# process video at this scale factor
scale = 0.1

x = None
y = None
norm_x = None
norm_y = None
primed = False
counter = 0

X = []

print("collecting video frames")
for frame in reader.nextFrame():
    counter += 1
    if counter <= args.skip_frames:
        continue
    if counter % 2 != 0:
        continue
    
    frame = frame[:,:,::-1]     # convert from RGB to BGR (to make opencv happy)
    small = cv2.resize(frame, (200,200), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    #X.append( gray.flatten() )
    X.append( np.flipud(gray) )

    if len(X) > 200:
        break

(h, w) = X[0].shape
print(w,h)

print("running dmd")
dmd = DMD(svd_rank=3)
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

