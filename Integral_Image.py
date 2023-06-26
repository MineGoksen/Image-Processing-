import numpy as np
import cv2 as cv

def cv_integral_image(img,n,m):
    imgIntegral = cv.integral(img)
    image = imgIntegral
    return image, imgIntegral/imgIntegral.max()

def own_integral_image(img):
    w = 3
    x = (w - 1) / 2
    pad_img = np.pad(img, (1, 1))
    m, n = pad_img.shape
    new_img = np.zeros([m, n])

    for i in range(int(x), int(m - x)):
        for j in range(int(x), int(n - x)):
            new_img[i, j] = np.sum(pad_img[0:int(i + 1), 0:int(j + 1)])

    new_img = new_img[:-1, :-1]
    new_img = new_img / new_img.max()
    return new_img

def soru2(img):
    img = cv.copyMakeBorder(img, 1, 1, 1, 1, cv.BORDER_REPLICATE)
    m,n = img.shape
    new_img = np.zeros((m, n))

    for i in range(m-1, 2,-1):
        for j in range(n-1, 2,-1):
            new_img[i][j] += img[i][j]
            new_img[i][j] += img[i-3][j-3]
            new_img[i][j] -= img[i][j-3]
            new_img[i][j] -= img[i-3][j]
            new_img[i][j] = np.round(new_img[i][j]/9)

    new_img = new_img.astype(np.uint8)
    return new_img


if __name__ == '__main__':
    img = cv.imread("lena_grayscale_hq.jpg",0)
    n,m = img.shape
    cv_img_integral, cv_img_integral2 = cv_integral_image(img,n,m)
    own_img_integral = own_integral_image(img)

    integral_box_filter = soru2(cv_img_integral)[3:, 3:]
    opencv_box = cv.blur(img, (3, 3),borderType= 0)

    diff = np.sum(cv.absdiff(cv_img_integral2, own_img_integral))
    diff2 = np.sum(cv.absdiff(integral_box_filter,opencv_box))

    print(f"Comparision for Q1: {diff}")
    print(f"Comparision for Q2: {diff2}")


    cv.imshow("Opencv Box Filter", opencv_box)
    cv.imshow("OpenCv Integral Image", cv_img_integral2)
    cv.imshow(f"Own Integral Image (Q1) AbsDiff: {diff}", own_img_integral)
    cv.imshow("Integral Box Filter (Q2)", integral_box_filter)
    cv.imshow(f"Difference of opencv Box filter and integral box filter AbsDiff: {diff2}", cv.absdiff(integral_box_filter, opencv_box) * 100)
    cv.waitKey()



