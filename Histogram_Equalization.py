import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

#PERVİN MİNE GÖKŞEN
#own_histogram_eq fonksiyonunda numpy kütüphaneleri kullanılmadığı için yavaş çalışmaktadır.

def own_histogram_eq(name,m,n):
    img = cv2.imread(name, 0)
    count = []
    mn = float(m) * float(n)

    #hist
    for k in range(0, 256):
        tmp = 0
        for i in range(m):
            for j in range(n):
                if img[i, j] == k:
                    tmp += 1.0 / mn
        count.append(tmp)
    temp = 0

    #cdf calculate
    for i in range (0,256):
        temp += count[i]
        count[i] = temp

    for i in range(0,256):
        count[i] = (count[i]) * 255
        count[i] = np.round_(count[i]).astype(np.uint8)

    #equalization
    for i in range(m):
        for j in range(n):
            for k in range(0, 256):
                if img[i,j] == k :
                    img[i,j] = count[k]

    cv2.imwrite('output1.jpg',img)

def opencvHist(name):
    src = cv2.imread(name, 0)
    dst = cv2.equalizeHist(src)
    cv2.imwrite('output2.jpg', dst)

def alternative(name,m,n):
    nm = float(n)*float(m)
    img = cv2.imread(name, 0)

    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = np.cumsum(hist)

    #min değerinin index numarallarını dönüyor.
    gmin = np.where(np.array(cdf) == min(p for p in cdf if p > 0))
    #min değerin ilk görüldüğü indexi aldım.
    index = gmin[0][0]
    hmin = cdf[index]

    T = ((cdf - hmin)*255 / (nm - hmin))
    #rounded = [round(x) for x in T]
    rounded = np.round_(T).astype(np.uint8)

    for i in range(0,m):
        for j in range(0,n):
            img[i,j] = rounded[img[i,j]]

    cv2.imwrite('output3.jpg',img)

if __name__ == '__main__':
    img = cv2.imread('test2.jpg', 0)
    m,n = (img.shape)

    alternative('test2.jpg',m,n)
    own_histogram_eq('test2.jpg', m, n)
    opencvHist('test2.jpg')

    o1 = cv2.imread('output1.jpg', 0)
    o2 = cv2.imread('output2.jpg', 0)
    o3 = cv2.imread('output3.jpg', 0)
    abs1_2 = cv2.imread('abs1_2.jpg', 0)
    abs2_3 = cv2.imread('abs2_3.jpg', 0)

    o11, bins_1 = np.histogram(o1.ravel(), 256, [0, 256])
    o12, bins_2 = np.histogram(o2.ravel(), 256, [0, 256])
    o13, bins_3 = np.histogram(o3.ravel(), 256, [0, 256])

    fig, axs = plt.subplots(3, sharex=True, sharey=True)

    axs[0].set_title('Pencere kapatıldıktan sonra resimler gelmektedir.\n\nOwn Histogram Equalization')
    axs[0].plot(o11)
    axs[1].set_title('OpenCv Histogram Equalization')
    axs[1].plot(o12)
    axs[2].set_title('Alternative Histogram Equalization')
    axs[2].plot(o13)
    plt.show()

    abs1_2 = 100*np.abs(o1-o2)
    cv2.imwrite("abs1_2.jpg", abs1_2)

    abs2_3 = 100 * np.abs(o2 - o3)
    cv2.imwrite("abs2_3.jpg", abs2_3)

    cv2.imshow('OUTPUT 1 -Own Histogram Image- ', o1)
    cv2.imshow('OUTPUT 2 -OpenCv Image- ', o2)
    cv2.imshow('OUTPUT 3 -Alternative Image-', o3)
    cv2.imshow('Own Histogram Vs OpenCv Image', abs1_2)
    cv2.imshow('Alternative Vs OpenCv Image', abs2_3)
    cv2.waitKey(0)






