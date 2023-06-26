import numpy as np
import cv2


def cv_box_filter(img, filter_size):
    padding_size = (filter_size - 1) / 2
    pad_img = np.pad(img, pad_width=int(padding_size))
    blur = cv2.blur(pad_img, (filter_size, filter_size), 0)
    return blur

def own_box_filter(img, filter_size):
    padding_size = (filter_size - 1) / 2
    pad_img = np.pad(img, pad_width=int(padding_size))
    m, n = pad_img.shape
    new_img = np.zeros([m, n])

    for i in range(int(padding_size), int(m - padding_size)):
        for j in range(int(padding_size), int(n - padding_size)):
            new_img[i, j] = np.sum(pad_img[int(i-padding_size):int(i+1+padding_size),
                                   int(j-padding_size):int(j+1+padding_size)]) / (filter_size * filter_size)

    new_img = new_img.astype(np.uint8)
    return new_img


def seperate_filter(img, filter_size):
    padding_size = (filter_size - 1) / 2
    pad_img = np.pad(img, pad_width=int(padding_size))
    m, n = pad_img.shape
    new_img = np.zeros([m, n])

    for i in range(int(padding_size), int(m - padding_size)):
        for j in range(int(padding_size), int(n - padding_size)):
            new_img[i, j] = np.sum(pad_img[int(i-padding_size):int(i+padding_size+1), j]) / filter_size
            new_img[i, j] = np.sum(pad_img[i, int(j-padding_size):int(j+padding_size+1)]) / filter_size

    new_img = np.round_(new_img).astype(np.uint8)
    return new_img


if __name__ == '__main__':
    img = cv2.imread('lena_grayscale_hq.jpg', 0)
    output_1_1 = own_box_filter(img, 3)
    output_1_2 = own_box_filter(img, 11)
    output_1_3 = own_box_filter(img, 21)

    output_2_1 = cv_box_filter(img, 3)
    output_2_2 = cv_box_filter(img, 11)
    output_2_3 = cv_box_filter(img, 21)

    output_3_1 = seperate_filter(img, 3)
    output_3_2 = seperate_filter(img, 11)
    output_3_3 = seperate_filter(img, 21)

    output_1_1 = output_1_1[1:output_1_1.shape[0] - 1, 1:output_1_1.shape[1] - 1]
    output_1_2 = output_1_2[5:output_1_2.shape[0] - 5, 5:output_1_2.shape[1] - 5]
    output_1_3 = output_1_3[10:output_1_3.shape[0] - 10, 10:output_1_3.shape[1] - 10]

    output_2_1 = output_2_1[1:output_2_1.shape[0] - 1, 1:output_2_1.shape[1] - 1]
    output_2_2 = output_2_2[5:output_2_2.shape[0] - 5, 5:output_2_2.shape[1] - 5]
    output_2_3 = output_2_3[10:output_2_3.shape[0] - 10, 10:output_2_3.shape[1] - 10]

    output_3_1 = output_3_1[1:output_3_1.shape[0] - 1, 1:output_3_1.shape[1] - 1]
    output_3_2 = output_3_2[5:output_3_2.shape[0] - 5, 5:output_3_2.shape[1] - 5]
    output_3_3 = output_3_3[10:output_3_3.shape[0] - 10, 10:output_3_3.shape[1] - 10]

    filter_3 = np.concatenate((output_1_1, output_2_1,output_3_1), axis=1)
    filter_11 = np.concatenate((output_1_2, output_2_2,output_3_2), axis=1)
    filter_21 = np.concatenate((output_1_3, output_2_3, output_3_3), axis=1)

    abs1_2_1 = cv2.absdiff(output_1_1,output_2_1)
    abs1_2_2 = cv2.absdiff(output_1_2, output_2_2)
    abs1_2_3 = cv2.absdiff(output_1_3, output_2_3)

    abs3_2_1 = cv2.absdiff(output_3_1, output_2_1)
    abs3_2_2 = cv2.absdiff(output_3_2, output_2_2)
    abs3_2_3 = cv2.absdiff(output_3_3, output_2_3)

    tmp1 = np.concatenate((abs1_2_1, abs1_2_2, abs1_2_3), axis=1)
    tmp2 = np.concatenate((abs3_2_1, abs3_2_2, abs3_2_3), axis=1)

    cv2.imshow('Output_1_1, Output_2_1, Output_3_1 (3X3)', filter_3)
    cv2.imshow('Output_1_2, Output_2_2, Output_3_2 (11X11)', filter_11)
    cv2.imshow('Output_1_3, Output_2_3, Output_3_3 (21X21)', filter_21)
    cv2.imshow('Absolute Difference between Output 1 and Output 2',  tmp1)
    cv2.imshow('Absolute Difference between Output 3 and Output 2',  tmp2)
    """
    cv2.imshow('Output_1_1', output_1_1)
    cv2.imshow('Output_1_2', output_1_2)
    cv2.imshow('Output_1_3', output_1_3)
    cv2.imshow('Output_2_1', output_2_1)
    cv2.imshow('Output_2_2', output_2_2)
    cv2.imshow('Output_2_3', output_2_3)
    cv2.imshow('Output_3_1', output_3_1)
    cv2.imshow('Output_3_2', output_3_2)
    cv2.imshow('Output_3_3', output_3_3)
    """
    cv2.waitKey(0)

