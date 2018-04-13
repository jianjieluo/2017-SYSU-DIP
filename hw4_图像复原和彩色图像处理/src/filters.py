from PIL import Image
from statistics import median
import numpy as np

from hw2func import *


def arithmetic_mean_filter(input_img, scale):
    m, n = scale
    filter_ = np.ones(scale) / (m * n)
    return filter2d(input_img, filter_)


def geometric_mean_filter(input_img, scale):
    a = int((scale[0] - 1) / 2)
    b = int((scale[1] - 1) / 2)

    img = np.array(input_img, dtype='f')
    height, width = img.shape

    newIm = Image.new('L', input_img.size)
    for x in range(height):
        for y in range(width):
            rst = x - a
            red = x + a + 1
            cst = y - b
            ced = y + b + 1

            if rst < 0:
                rst = 0
            if red > height:
                red = height
            if cst < 0:
                cst = 0
            if ced > width:
                ced = width

            submatrix = img[rst:red, cst:ced]
            temp = 1
            m = red - rst
            n = ced - cst
            for i in range(m):
                for j in range(n):
                    if temp == 0:
                        break
                    temp *= submatrix[i][j]
                if temp == 0:
                    break

            newIm.putpixel((y, x), int(
                pow(temp, 1 / (m * n))))
    return newIm


def median_filter(input_img, scale):
    a = int((scale[0] - 1) / 2)
    b = int((scale[1] - 1) / 2)

    img = np.array(input_img, dtype='f')
    height, width = img.shape

    newIm = Image.new('L', input_img.size)
    for x in range(height):
        for y in range(width):
            rst = x - a
            red = x + a + 1
            cst = y - b
            ced = y + b + 1

            if rst < 0:
                rst = 0
            if red > height:
                red = height
            if cst < 0:
                cst = 0
            if ced > width:
                ced = width

            submatrix = img[rst:red, cst:ced].reshape(
                (1, (red - rst) * (ced - cst)))

            newIm.putpixel((y, x), int(median(submatrix[0])))
    return newIm


def max_filter(input_img, scale):
    a = int((scale[0] - 1) / 2)
    b = int((scale[1] - 1) / 2)

    img = np.array(input_img, dtype='f')
    height, width = img.shape

    newIm = Image.new('L', input_img.size)
    for x in range(height):
        for y in range(width):
            rst = x - a
            red = x + a + 1
            cst = y - b
            ced = y + b + 1

            if rst < 0:
                rst = 0
            if red > height:
                red = height
            if cst < 0:
                cst = 0
            if ced > width:
                ced = width

            submatrix = img[rst:red, cst:ced].reshape(
                (1, (red - rst) * (ced - cst)))

            newIm.putpixel((y, x), int(max(submatrix[0])))
    return newIm


def min_filter(input_img, scale):
    a = int((scale[0] - 1) / 2)
    b = int((scale[1] - 1) / 2)

    img = np.array(input_img, dtype='f')
    height, width = img.shape

    newIm = Image.new('L', input_img.size)
    for x in range(height):
        for y in range(width):
            rst = x - a
            red = x + a + 1
            cst = y - b
            ced = y + b + 1

            if rst < 0:
                rst = 0
            if red > height:
                red = height
            if cst < 0:
                cst = 0
            if ced > width:
                ced = width

            submatrix = img[rst:red, cst:ced].reshape(
                (1, (red - rst) * (ced - cst)))

            newIm.putpixel((y, x), int(min(submatrix[0])))
    return newIm


def harmonic_mean_filter(input_img, scale):
    return contra_harmonic_mean_filter(input_img, scale, Q=-1)


# def harmonic_mean_filter_brute_test(input_img, scale):
#     a = int((scale[0] - 1) / 2)
#     b = int((scale[1] - 1) / 2)

#     img = np.array(input_img, dtype='f')
#     height, width = img.shape

#     newIm = Image.new('L', input_img.size)
#     for x in range(height):
#         for y in range(width):
#             rst = x - a
#             red = x + a + 1
#             cst = y - b
#             ced = y + b + 1

#             if rst < 0:
#                 rst = 0
#             if red > height:
#                 red = height
#             if cst < 0:
#                 cst = 0
#             if ced > width:
#                 ced = width

#             submatrix = img[rst:red, cst:ced]
#             newvalue = submatrix.size / sum(sum(1 / submatrix))
#             newIm.putpixel((y, x), int(newvalue))
#     return newIm


def contra_harmonic_mean_filter(input_img, scale, Q):
    a = int((scale[0] - 1) / 2)
    b = int((scale[1] - 1) / 2)

    img = np.array(input_img, dtype='f')
    height, width = img.shape

    newIm = Image.new('L', input_img.size)
    for x in range(height):
        for y in range(width):
            rst = x - a
            red = x + a + 1
            cst = y - b
            ced = y + b + 1

            if rst < 0:
                rst = 0
            if red > height:
                red = height
            if cst < 0:
                cst = 0
            if ced > width:
                ced = width

            submatrix = img[rst:red, cst:ced]

            up = sum(sum(pow(submatrix, Q + 1)))
            down = sum(sum(pow(submatrix, Q)))

            newvalue = up / down
            try:
                newIm.putpixel((y, x), int(newvalue))
            except ValueError:
                newIm.putpixel((y, x), 0)
    return newIm
