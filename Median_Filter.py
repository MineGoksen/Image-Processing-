import cv2 as cv
import numpy as np


def cv_median_filter(img):
    pad_img = np.pad(img, (2, 2), 'reflect')
    median = cv.medianBlur(pad_img, 5)
    return median


def own_median_filter(img):
    pad_img = np.pad(img, (2, 2), 'reflect')
    m, n = pad_img.shape
    new_img = np.zeros([m, n])

    for i in range(2, m - 2):
        for j in range(2, n - 2):
            new_img[i, j] = np.median((pad_img[i - 2:i + 3, j - 2:j + 3]).flatten())

    new_img = new_img.astype(np.uint8)
    return new_img


def psnr(img,lena):
    box = cv.blur(img, (5, 5))
    gaussian = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)
    median = cv.medianBlur(img, 5)
    psnr_box = cv.PSNR(box, lena)
    psnr_gaus = cv.PSNR(gaussian, lena)
    psnr_median = cv.PSNR(median, lena)
    return psnr_median, psnr_box, psnr_gaus


def weighted_median_filter(img):
    pad_img = np.pad(img, (2, 2), 'reflect')
    m, n = pad_img.shape
    new_img = np.zeros([m, n])

    for i in range(2, m - 2):
        for j in range(2, n - 2):
            flatten_arr = (pad_img[i - 2:i + 3, j - 2:j + 3]).flatten()
            flatten_arr_1 = np.append(flatten_arr, pad_img[i,j])
            flatten_arr_2 = np.append(flatten_arr_1, pad_img[i, j])
            new_img[i,j] = np.median(flatten_arr_2)

    new_img = new_img.astype(np.uint8)
    return new_img

def opencv(img):
    #pad_img = np.pad(img, (2, 2), 'reflect')
    box = cv.blur(img, (5, 5), borderType=cv.BORDER_REPLICATE)
    gaussian = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)
    return box,gaussian

if __name__ == '__main__':
    img = cv.imread('noisyImage.jpg', 0)
    lena = cv.imread('lena_grayscale_hq.jpg', 0)

    output2 = cv_median_filter(img)
    output1 = own_median_filter(img)
    weighted = weighted_median_filter(img)
    box_filter, gaus_filter = opencv(img)

    output1_1 = output1[2:output1.shape[0] - 2, 2:output1.shape[1] - 2]
    output2_2 = output2[2:output2.shape[0] - 2, 2:output2.shape[1] - 2]
    weighted1 = weighted[2:weighted.shape[0] - 2, 2:weighted.shape[1] - 2]

    diff = (np.abs(output1_1 - output2_2))
    print("Difference between opencv median filter and own median filter %d" % np.sum(np.sum(diff)))
    cv.imshow('Diff', diff)

    psnr_median, psnr_box, psnr_gaus = psnr(img,lena)
    print("SORU 2")
    print("PSNR value for opencv median filter % f" % psnr_median)
    print("PSNR value for opencv box filter % f" % psnr_box)
    print("PSNR value for opencv gaussian filter 7x7 % f" % psnr_gaus)
    ##########################################################################################################

    print("SORU3")
    print("PSNR value for own median filter % f" % (cv.PSNR(output1_1, lena)))
    print("PSNR value for opencv box filter % f" % (cv.PSNR(box_filter, lena)))
    print("PSNR value for opencv gaussian filter % f" % (cv.PSNR(gaus_filter, lena)))
    print("PSNR value for opencv median filter % f" % (cv.PSNR(output2_2, lena)))
    print("PSNR value for own weighted box filter % f" % (cv.PSNR(weighted1, lena)))

    ############  SORU4  ###################################################################################
    num_rows, num_cols = weighted1.shape[:2]
    temp = np.float32([[1, 0, 7], [0, 1, 5]])
    incr_psnr = cv.warpAffine(weighted1, temp, (num_cols, num_rows))
    ############  SORU4  ###################################################################################

    cv.imshow(f'Shifted Image for Question 4 Compare with weighted median image PSNR: {cv.PSNR(lena,incr_psnr)}', incr_psnr)
    cv.imshow(f'OpenCV Median Filter PSNR: {(cv.PSNR(output2_2, lena))}', output2_2)
    cv.imshow(f'Own Median Filter PSNR: {(cv.PSNR(output1_1, lena))}', output1_1)
    cv.imshow(f'Center weighted median filter PSNR: {(cv.PSNR(weighted1, lena))}', weighted)
    cv.imshow(f'OpenCV Box Filter PSNR: {(cv.PSNR(box_filter, lena))}', box_filter)
    cv.imshow(f'OpenCV Gaussian Filter PSNR: {(cv.PSNR(gaus_filter, lena))}', gaus_filter)
    cv.waitKey(0)

