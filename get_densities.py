import cv2
from math import sqrt
from skimage.feature import blob_log
from skimage.morphology import skeletonize
from skimage.draw import disk
import numpy as np

def rem_opticdisk(image):
    nimage = image.copy()
    nimage = cv2.bitwise_not(nimage)
    blobs_log = blob_log(nimage, max_sigma=30, num_sigma=10, threshold=.1)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    #OD cirlce
    t = 1
    idx = np.argsort(blobs_log.T[2])[-t] #largest radius
    y,x,r = blobs_log[idx]
    rr, cc = disk((y, x), r)
    while max(rr) > 255 or max(cc) > 255 or min(rr) < 0 or min(cc) < 0 :
        t+=1
        idx = np.argsort(blobs_log.T[2])[-t] #largest radius
        y,x,r = blobs_log[idx]
        rr, cc = disk((y, x), r)
    image[rr,cc] = 0
    return image


def threshold(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(image)
    #im_res = cv2.adaptiveThreshold(equalized,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,0)
    th, _ = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    desired_th = th*0.9
    _, th_img = cv2.threshold(equalized, desired_th, 255, cv2.THRESH_BINARY)
    return th_img

def profusion_density(image):
    bin_image = threshold(image)
    bin_image[bin_image==255] = 1
    return sum(sum(bin_image))/(256*256), bin_image

def vessel_density(bin_image):
    skeleton = skeletonize(bin_image)
    return sum(sum(skeleton))/(256*256), skeleton