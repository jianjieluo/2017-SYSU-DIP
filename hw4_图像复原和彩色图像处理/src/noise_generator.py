from PIL import Image
import numpy as np
import random


def Gaussian_noise(input_img, mu, sigma):
    newIm = Image.new('L', input_img.size)

    img = np.array(input_img, dtype='f')
    height, width = img.shape

    noise = np.random.normal(mu, sigma, img.shape)
    img += noise

    for x in range(height):
        for y in range(width):
            newIm.putpixel((y, x), int(img[x][y]))

    return newIm


def Salt_and_pepper_noise(input_img, ps, pp):
    newIm = Image.new('L', input_img.size)

    img = np.array(input_img, dtype='f')
    height, width = img.shape

    # flag 标志椒盐分布情况
    # 0 表示正常
    # 1 表示有盐噪声
    # 2 表示有椒噪声
    flag = np.zeros(img.shape)

    # 注意作用域
    def get_random_raw_point():
        while True:
            x = random.randint(0, height - 1)
            y = random.randint(0, width - 1)
            if flag[x, y] == 0:
                return (x, y)

    # 根据ps来得到盐噪声的点的数目
    salt_num = round(img.size * ps)
    while (salt_num > 0):
        coor = get_random_raw_point()
        img[coor] = 255
        flag[coor] = 1
        salt_num -= 1

    # 根据pp来得到椒噪声的点的数目
    pep_num = round(img.size * pp)
    while (pep_num > 0):
        coor = get_random_raw_point()
        img[coor] = 0
        flag[coor] = 2
        pep_num -= 1

    for x in range(height):
        for y in range(width):
            newIm.putpixel((y, x), int(img[x][y]))

    return newIm
