import random
import numpy as np
import cv2 as cv

completed = 0;
def hist_eq(img, threshold):
    #HW1 Histogram equalization
    m,n = img.shape
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    t = 0
    for i in range(0,len(hist)):
        if hist[i] > threshold:
            t += (hist[i] - threshold)
            hist[i] = threshold

    count = 0
    while(count<=t):
        rand = random.randint(0,255)
        if(hist[rand]+1 <= threshold):
            hist[rand] += 1
            count += 1

    cdf = np.cumsum(hist)
    gmin = np.where(np.array(cdf) == min(p for p in cdf if p > 0))
    index = gmin[0][0]
    hmin = cdf[index]

    T = ((cdf - hmin) * 255 / (n*m - hmin))
    rounded = np.round_(T).astype(np.uint8)

    img = rounded[img]

    return img

def clahe(img,tile,thold):
    pad = int((tile-1)/2)
    pad_img = np.pad(img, int(pad), mode='symmetric')
    m,n = pad_img.shape
    new_img = np.zeros([m, n])
    counter = 0
    for i in range(int(pad), int(m - pad)):
        for j in range(int(pad), int(n - pad)):
            if(counter % 20000 == 0):
                print("...")
            temp = hist_eq(pad_img[int(i-pad):int(i+pad),int(j-pad):int(j+pad)], thold)
            new_img[i,j] = temp[int(pad),int(pad)]
            counter+=1

    new_img = new_img.astype('uint8')
    return new_img[pad:-pad, pad:-pad]

if __name__ == '__main__':
    print("Kodun calisma suresi yaklasik 5.30dk'dir.")
    img = cv.imread("test7.jpg",0)
    x = clahe(img,32,8)
    cv.imshow("My CLAHE",x)
    cv.waitKey(0)


