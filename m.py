from glob import glob
import numpy as np
import os
import cv2

imgs = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in glob(os.sep.join(['./imgs/', '*.jpg']))]

cImgSize = imgs[0].shape[::-1] # (w, h)
cBoardSize = (9, 6)

cLen = len(imgs)
assert 0 == cLen%2
cLen = cLen//2

def detectCorners(img):
    ok, corners = cv2.findChessboardCorners(img, cBoardSize)
    assert ok
    cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    return corners

cImgPoints = [detectCorners(img).reshape(-1, 2) for img in imgs]

cObjPoints = np.zeros((np.prod(cBoardSize), 3), np.float32)
cObjPoints[:, :2] = np.indices(cBoardSize).T.reshape(-1, 2)
cObjPoints = cLen*[cObjPoints]

K0 = cv2.initCameraMatrix2D(cObjPoints, cImgPoints[:cLen], cImgSize, 0)
K1 = cv2.initCameraMatrix2D(cObjPoints, cImgPoints[cLen:], cImgSize, 0)

rms, K0, D0, K1, D1, R, T, E, F = cv2.stereoCalibrate(
    cObjPoints, cImgPoints[:cLen], cImgPoints[cLen:],
    K0, None, K1, None,
    cImgSize,
    flags = cv2.CALIB_FIX_ASPECT_RATIO
            | cv2.CALIB_ZERO_TANGENT_DIST
            | cv2.CALIB_USE_INTRINSIC_GUESS
            | cv2.CALIB_SAME_FOCAL_LENGTH
            | cv2.CALIB_RATIONAL_MODEL
            | cv2.CALIB_FIX_K3
            | cv2.CALIB_FIX_K4
            | cv2.CALIB_FIX_K5,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
)

R0, R1, P0, P1, Q, roi0, roi1 = cv2.stereoRectify(K0, D0, K1, D1, cImgSize, R, T, flags = cv2.CALIB_ZERO_DISPARITY, alpha = 0)

def rectify(img, K, D, R, P):
    m0, m1 = cv2.initUndistortRectifyMap(K, D, R, P, cImgSize, cv2.CV_16SC2)
    return cv2.remap(img, m0, m1, cv2.INTER_LINEAR)

melonL = rectify(cv2.imread('melonL.jpg'), K0, D0, R0, P0)
melonR = rectify(cv2.imread('melonR.jpg'), K1, D1, R1, P1)

melon = np.concatenate((melonL, melonR), axis = 1)
melon[::40, :] = (0, 255, 0)

def cb(e, x, y, f, p):
    global start
    global end
    if x < cv2.getTrackbarPos('disparities', 'depth')*16 or x > cImgSize[0]:
        return
    if e == cv2.EVENT_LBUTTONDOWN:
        start = points3d[y][x]
        print('起始:', (round(start[0],2),round(start[1],2), round(start[2],2)))
    if e == cv2.EVENT_LBUTTONUP:
        end = points3d[y][x]
        print('终止:', (round(end[0],2),round(end[1],2),round(end[2],2)))
        distance = np.sqrt(np.sum((start - end)**2))
        print('距离:', distance, 'cm')

cv2.namedWindow('depth')
cv2.namedWindow('rectify')
cv2.createTrackbar('disparities', 'depth', 5, 60, lambda x: None)
cv2.createTrackbar('block', 'depth', 3, 32, lambda x: None)
cv2.setMouseCallback('rectify', cb, None)

while True:
    d = cv2.getTrackbarPos('disparities', 'depth')*16
    b = cv2.getTrackbarPos('block', 'depth')

    matcher0 = cv2.StereoSGBM_create(0, d, b, 24*b, 96*b, 12, 10, 50, 32, 63, cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    matcher1 = cv2.ximgproc.createRightMatcher(matcher0)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher0)
    wls_filter.setLambda(80000)
    wls_filter.setSigmaColor(1.3)
    disp0 = np.int16(matcher0.compute(melonL, melonR))
    disp1 = np.int16(matcher1.compute(melonR, melonL))

    depth = wls_filter.filter(disp0, melonL, None, disp1).astype(np.float32)/16.
    points3d = cv2.reprojectImageTo3D(depth, Q)

    frame = melon.copy()
    cv2.line(frame, (d, 0), (d, cImgSize[1]), (0, 255, 0), 1)
    cv2.imshow('rectify', frame)

    depthImg = depth.copy()
    depthImg = np.uint8(cv2.normalize(depthImg, depthImg, 255, 0, cv2.NORM_MINMAX))
    cv2.imshow('depth', depthImg)

    cv2.waitKey(100)