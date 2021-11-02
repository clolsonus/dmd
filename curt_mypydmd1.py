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
parser.add_argument('--scale', type=float, default=1.0, help='scale input')
args = parser.parse_args()

print("Opening ", args.video)
reader = skvideo.io.FFmpegReader(args.video, inputdict={}, outputdict={})

counter = 0
scale = args.scale
dmd_size = 200
max_rank = 9
X = []

# non-changing pixel (bike-vid)
#hp = 0.9
#wp = 0.85

# changing pixel (bike-vid)
#hp = 0.6
#wp = 0.56

# for moving video
hp = 0.25
wp = 0.8

px_t = []
px_val = []
print("collecting video frames")
for frame in reader.nextFrame():
    px_t.append(counter * 0.1)
    counter += 1
    if counter <= args.skip_frames:
        continue
    
    frame = frame[:,:,::-1]     # convert from RGB to BGR (to make opencv happy)
    scaled = cv2.resize(frame, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    (h, w) = gray.shape
    u = int(h*hp)
    v = int(w*wp)
    px_val.append( gray[ u, v ] )
    show = cv2.merge( (gray, gray, gray) )
    cv2.circle(show, (v, u), 5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("selected pixel", show)
    cv2.waitKey(1)
    # X.append( np.flipud(gray) )
    small = cv2.resize(gray, (dmd_size,dmd_size), interpolation=cv2.INTER_AREA)
    X.append( np.flipud(small) )

    if len(X) > 200:
        break

#(h, w) = X[0].shape
#print(w,h)

print("running dmd")
dmd = DMD(svd_rank=max_rank)
dmd.fit(np.array(X))

print("dmd frequency:", dmd.frequency)

recon_t = []
recon_val = []

for i in range(dmd.reconstructed_data.shape[1]):
    big = dmd.reconstructed_data[:,i]
    #big = 255 * dmd.reconstructed_data[:,i] / np.max(dmd.reconstructed_data[:,i]) # avoid overflow
    big = cv2.resize(np.flipud(big.reshape((dmd_size,dmd_size)).astype('uint8')), (w, h), interpolation=cv2.INTER_AREA)
    #big = cv2.resize(np.flipud(big.reshape((h,w)).astype('uint8')), (w, h), interpolation=cv2.INTER_AREA)
    big = 255 * ( big / np.max(big) )
    u = int(h*hp)
    v = int(w*wp)
    recon_t.append(i*0.1)
    recon_val.append( big[ u, v ] )
    cv2.imshow("reconstructed", big.astype('uint8'))
    cv2.waitKey(1)
    
# plot pixel value
plt.figure()
plt.plot(px_t, px_val, label="Original")
plt.plot(recon_t, recon_val, label="DMD Approximation")
plt.title("Single Pixel Perspective")
plt.xlabel("Time (sec)")
plt.ylabel("Pixel value (0-255)")
plt.ylim(0,255)
plt.legend()
plt.show()

dmd.plot_modes_2D(figsize=(12,5))

# print(X[0].shape[0])
# x1 = np.array(list(range(w)))
# x2 = np.array(list(range(h)))
# x1grid, x2grid = np.meshgrid(x1, x2)
# fig = plt.figure(figsize=(18,12))
# for id_subplot, snapshot in enumerate(dmd.reconstructed_data.T[:16], start=1):
#     plt.subplot(4, 4, id_subplot)
#     plt.pcolor(x1grid, x2grid, snapshot.reshape(x1grid.shape).real, vmin=-1, vmax=1)

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

# fig = plt.figure(figsize=(17,6))

# for n, mode, dynamic in zip(range(131, 133), dmd.modes.T, dmd.dynamics):
#     plt.subplot(n)
#     plt.pcolor(x1grid, x2grid, (mode.reshape(-1, 1).dot(dynamic.reshape(1, -1))).real.T)
    
# plt.subplot(133)
# plt.pcolor(x1grid, x2grid, dmd.reconstructed_data.T.real)
# plt.colorbar()

plt.show()

