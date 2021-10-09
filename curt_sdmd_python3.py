#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
import skvideo.io               # pip install scikit-video

parser = argparse.ArgumentParser(description='virtual choir')
parser.add_argument('video', help='video file')
parser.add_argument("--skip-frames", type=int)
args = parser.parse_args()

print("Opening ", args.video)
reader = skvideo.io.FFmpegReader(args.video, inputdict={}, outputdict={})

# number of Gram-Schmidt iterations
nGram = 5

# max rank
r0 = 49

# "forgetting" factor
alpha = 0.1

# convenience
eps = np.finfo(float).eps

# process video at this scale factor
scale = 0.25

def SDMD_ComputeModes(K_tilde, Qx):
    # Compute eigenvalues and eigenvectors of K_tilde
    (evalsK, evecsK) = np.linalg.eig(K_tilde);
    print("eigen values:", evalsK)

    print("evalsK:", evalsK)
    print("evalsK-1:", evalsK-1)
    # sort 1+0i to front
    idx = np.argsort(np.abs(evalsK-1))
    print(idx)
    evalsK = evalsK[idx]
    evecsK = evecsK[:,idx]

    # Vectorize eigenvalues
    evalsK = np.diag(evalsK)
    
    # Compute modes
    modesK = Qx @ evecsK

    return modesK, evalsK, evecsK


x = None
y = None
norm_x = None
norm_y = None
primed = False
counter = 0

for frame in reader.nextFrame():
    counter += 1
    if counter <= args.skip_frames:
        continue
    if counter % 4 != 0:
        continue
    
    frame = frame[:,:,::-1]     # convert from RGB to BGR (to make opencv happy)
    small = cv2.resize(frame, (0,0), fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    cv2.imshow("orig", small)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    x = y
    norm_x = norm_y
    y = gray.flatten()
    y = y.reshape((y.shape[0], 1))
    norm_y = np.linalg.norm(y)
    # if y is not None:
    #     cv2.imshow("y", y.reshape(gray.shape))
    # if x is not None:
    #     cv2.imshow("x", x.reshape(gray.shape))
    
    # algorithm begins on the 2nd frame
    if x is None:
        continue

    print(x.shape, y.shape, norm_x, norm_y)
    
    # prime the pump
    if not primed:
        Qx = x / norm_x
        Qy = y / norm_y
        Gx = np.matrix( [[ norm_x**2 ]] )
        Gy = np.matrix( [[ norm_y**2 ]] )
        A_matrix = np.matrix( [[ norm_x*norm_y ]] )
        primed = True
        continue

    # STEP 1: Gram-Schmidt orthogonalization
    x_tilde = np.zeros( (Qx.shape[1], 1) )
    #print("x_tilde:", x_tilde)
    y_tilde = np.zeros( (Qy.shape[1], 1) )
    ex = np.copy(x.astype('float'))
    cv2.imshow("ex-start", ex.reshape(gray.shape[:2]).astype('uint8'))
    ey = np.copy(y.astype('float'))
    
    # Start iterative Gram-Schmidt
    for iGram in range(nGram):
        dx = Qx.T @ ex       # Qx'*ex (verify)
        #print("dx:", dx)
        dy = Qy.T @ ey       # qy'*ey
        x_tilde += dx
        y_tilde += dy
        ex -= Qx @ dx
        #print("shape ex:", ex.shape)
        ey -= Qy @ dy
        
    cv2.imshow("ey", ey.reshape(gray.shape[:2]).astype('uint8'))

    # STEP 2: check if expansion of Basis is necessary
    # Check for x 
    # CONFUSION HERE ( / norm_x)
    norm_ex = np.linalg.norm(ex)
    print("norm_ex:", norm_ex, "norm_x:", norm_x)
    if norm_ex / norm_x > eps:
        # Update basis for ex
        # Qx = [Qx ex/norm(ex)];
        Qx = np.hstack( [Qx, ex/norm_ex] )
        
        # Do zero padding for Gx and A_matrix to increase size
        # Gx = [Gx zeros(size(Gx,1),1); zeros(1,size(Gx,2)+1)];
        Gx = np.hstack( [Gx, np.zeros((Gx.shape[0],1))] )
        Gx = np.vstack( [Gx, np.zeros(Gx.shape[1])] )
        # A_matrix = [A_matrix zeros(size(A_matrix,1),1)];
        A_matrix = np.hstack( [A_matrix, np.zeros([A_matrix.shape[0], 1])] )
    
    # Check for y
    norm_ey = np.linalg.norm(ey)
    if norm_ey / norm_y > eps:
        # Update basis for y
        # Qy = [Qy ey/norm(ey)];
        Qy = np.hstack( [Qy, ey/norm_ey] )
        
        # Do zero padding for Gy and A_matrix
        # Gy = [Gy zeros(size(Gy,1),1); zeros(1,size(Gy,2)+1)];
        Gy = np.hstack( [Gy, np.zeros((Gy.shape[0],1))] )
        Gy = np.vstack( [Gy, np.zeros(Gy.shape[1])] )
        # A_matrix = [A_matrix; zeros(1,size(A_matrix,2))];
        A_matrix = np.vstack( [A_matrix, np.zeros([1, A_matrix.shape[1]])] )
    
    # STEP 3: Check if POD compression is needed
    if r0 > 0:
        # Check for x
        print("Qx shape:", Qx.shape)
        if Qx.shape[1] > r0:
            (eval, evec) = np.linalg.eig(Gx)
            print("Gx: eval", eval)
            #print("eig:", eval.shape, evec.shape)
            idx = eval.argsort()[::-1]   
            eval = eval[idx]
            evec = evec[:,idx]
            qx = evec[:,:r0]
            #print("Qx, qx shapes:", Qx.shape, qx.shape)
            Qx = Qx @ qx
            #Qx = Qx[:,:r0]*(1-alpha) + Qx @ qx * alpha
            A_matrix = A_matrix @ qx
            Gx = np.diag(eval[:r0])
            print("Gx:", Gx)
        
        # Check for y
        if Qy.shape[1] > r0:
            (eval, evec) = np.linalg.eig(Gy)
            idx = eval.argsort()[::-1]   
            eval = eval[idx]
            evec = evec[:,idx]
            qy = evec[:,:r0]
            Qy = Qy @ qy
            # Qy = Qy[:,:r0]*(1-alpha) + Qy @ qy * alpha
            A_matrix = qy.T @ A_matrix
            Gy = np.diag(eval[:r0])
    
    # STEP 4: Update step
    print("Qx shape:", Qx.shape, "x shape:", x.shape)
    x_tilde = Qx.T @ x
    print("x_tilde shape:", x_tilde.shape)
    y_tilde = Qy.T @ y

    if True:
        # Forgetting factor
        print(y_tilde.shape, x_tilde.shape)
        #A_matrix = A_matrix*(1-alpha) + y_tilde.T @ x_tilde * alpha
        A_matrix = A_matrix*(1-alpha) + y_tilde @ x_tilde.T * alpha
        Gx = Gx*(1-alpha) + x_tilde @ x_tilde.T * alpha
        Gy = Gy*(1-alpha) + y_tilde @ y_tilde.T * alpha
    else:
        # not forgetting
        A_matrix += y_tilde @ x_tilde.T
        Gx += x_tilde @ x_tilde.T
        Gy += y_tilde @ y_tilde.T
    

    #print("A_matrix:", A_matrix.shape, "\n", A_matrix)
    # K_tilde = Qx'*Qy*A_matrix*pinv(Gx);
    K_tilde = Qx.T @ Qy @ A_matrix @ np.linalg.pinv(Gx)
    modesK, evalsK, W = SDMD_ComputeModes(K_tilde, Qx)
    #print("modesK:", modesK.shape, evalsK.shape, W.shape)
    #print("sliceK:", modesK[:,0].shape)
    mode0 = 2*modesK[:,0].real
    #print(np.min(mode0), np.max(mode0))
    mode0 = 255 * mode0 / np.max(mode0)
    mode0 = mode0.reshape(gray.shape[:2])
    print(np.min(mode0.astype('uint8')), np.max(mode0.astype('uint8')))
    cv2.imshow("mode0", mode0.astype('uint8'))
    cv2.waitKey(1)
