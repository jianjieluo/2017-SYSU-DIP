from PIL import Image
import numpy as np
from statistics import mean

import filters
import noise_generator
from hw2func import *
from rgb2hsi import *


def task_1():
    img = Image.open('./task_1.png')

    # 算术均值
    filters.arithmetic_mean_filter(img, (3, 3)).save(
        './output/task_1/arithmetic_mean_filter3x3.png')
    filters.arithmetic_mean_filter(img, (9, 9)).save(
        './output/task_1/arithmetic_mean_filter9x9.png')

    # 谐波均值
    filters.harmonic_mean_filter(img, (3, 3)).save(
        './output/task_1/harmonic_mean_filter3x3.png')
    filters.harmonic_mean_filter(img, (9, 9)).save(
        './output/task_1/harmonic_mean_filter9x9.png')

    # 逆谐波均值
    filters.contra_harmonic_mean_filter(img, (3, 3), Q=-1.5).save(
        './output/task_1/contra_harmonic_mean_filter3x3.png')
    filters.contra_harmonic_mean_filter(img, (9, 9), Q=-1.5).save(
        './output/task_1/contra_harmonic_mean_filter9x9.png')


def task_2():
    img = Image.open('./task_2.png').convert('L')

    # 高斯噪声实验
    noise_generator.Gaussian_noise(img, 0, 40).save(
        './output/task_2/p1/add_Gaussian_noise.png')
    img1 = Image.open('./output/task_2/p1/add_Gaussian_noise.png').convert('L')
    # 算术均值
    filters.arithmetic_mean_filter(img1, (3, 3)).save(
        './output/task_2/p1/算术平均3x3.png')
    filters.arithmetic_mean_filter(img1, (5, 5)).save(
        './output/task_2/p1/算术平均5x5.png')
    filters.arithmetic_mean_filter(img1, (7, 7)).save(
        './output/task_2/p1/算术平均7x7.png')
    filters.arithmetic_mean_filter(img1, (9, 9)).save(
        './output/task_2/p1/算术平均9x9.png')
    # 几何均值
    filters.geometric_mean_filter(img1, (3, 3)).save(
        './output/task_2/p1/几何平均3x3.png')
    filters.geometric_mean_filter(img1, (5, 5)).save(
        './output/task_2/p1/几何平均5x5.png')
    # 中值滤波
    filters.median_filter(img1, (3, 3)).save(
        './output/task_2/p1/中值滤波3x3.png')
    filters.median_filter(img1, (5, 5)).save(
        './output/task_2/p1/中值滤波5x5.png')

    # 盐噪声实验
    noise_generator.Salt_and_pepper_noise(img, ps=0.2, pp=0).save(
        './output/task_2/p2/add_Salt_noise.png')
    img2 = Image.open('./output/task_2/p2/add_Salt_noise.png')
    # 调和均值处理
    filters.harmonic_mean_filter(img2, (3, 3)).save(
        './output/task_2/p2/调和均值3x3.png')
    filters.harmonic_mean_filter(img2, (5, 5)).save(
        './output/task_2/p2/调和均值5x5.png')

    # 逆谐波均值处理 Q > 0
    filters.contra_harmonic_mean_filter(img2, (3, 3), Q=1.5).save(
        './output/task_2/p2/逆谐波均值3x3q1.5.png')

    # 逆谐波均值处理 Q < 0
    filters.contra_harmonic_mean_filter(img2, (3, 3), Q=-1.5).save(
        './output/task_2/p2/逆谐波均值3x3q-1.5.png')
    filters.contra_harmonic_mean_filter(img2, (3, 3), Q=-2).save(
        './output/task_2/p2/逆谐波均值3x3q-2.0.png')
    filters.contra_harmonic_mean_filter(img2, (3, 3), Q=-2.5).save(
        './output/task_2/p2/逆谐波均值3x3q-2.5.png')

    # 椒盐噪声实验
    noise_generator.Salt_and_pepper_noise(img, ps=0.2, pp=0.2).save(
        './output/task_2/p3/add_Salt_and_pepper_noise.png')
    img3 = Image.open('./output/task_2/p3/add_Salt_and_pepper_noise.png')
    # 算术均值
    filters.arithmetic_mean_filter(img3, (3, 3)).save(
        './output/task_2/p3/算术平均3x3.png')
    filters.arithmetic_mean_filter(img3, (5, 5)).save(
        './output/task_2/p3/算术平均5x5.png')
    # 几何均值
    filters.geometric_mean_filter(img3, (3, 3)).save(
        './output/task_2/p3/几何平均3x3.png')
    filters.geometric_mean_filter(img3, (5, 5)).save(
        './output/task_2/p3/几何平均5x5.png')
    # 最大值滤波
    filters.max_filter(img3, (3, 3)).save(
        './output/task_2/p3/最大值滤波3x3.png')
    filters.max_filter(img3, (5, 5)).save(
        './output/task_2/p3/最大值滤波5x5.png')

    # 最小值滤波
    filters.min_filter(img3, (3, 3)).save(
        './output/task_2/p3/最小值滤波3x3.png')
    filters.min_filter(img3, (5, 5)).save(
        './output/task_2/p3/最小值滤波5x5.png')

    # 中值滤波
    filters.median_filter(img3, (3, 3)).save(
        './output/task_2/p3/中值滤波3x3.png')
    filters.median_filter(img3, (5, 5)).save(
        './output/task_2/p3/中值滤波5x5.png')


def task_3():
    img = Image.open('./29.png')
    Rimg, Gimg, Bimg = img.split()

    def prob_1():
        newRimg = equalize_hist(Rimg)
        newGimg = equalize_hist(Gimg)
        newBimg = equalize_hist(Bimg)

        newIm1 = Image.new('RGB', img.size)
        width, height = img.size

        for x in range(height):
            for y in range(width):
                newIm1.putpixel((y, x), (newRimg.getpixel((y, x)),
                                         newGimg.getpixel((y, x)), newBimg.getpixel((y, x))))

        newIm1.save('./output/task_3/RGB分别直方图均衡化再合并.png')

    def prob_2():
        pr_rk_R = get_gray_statistics(Rimg)
        pr_rk_G = get_gray_statistics(Gimg)
        pr_rk_B = get_gray_statistics(Rimg)

        mean_pr_rk = [mean([pr_rk_R[x], pr_rk_G[x], pr_rk_B[x]])
                      for x in range(len(pr_rk_R))]
        # 构建变换T
        T = [0] * 256
        for i in range(256):
            T[i] = round(255 * sum(mean_pr_rk[0:i + 1]))

        newIm2 = Image.new('RGB', img.size)
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                r = Rimg.getpixel((i, j))
                g = Gimg.getpixel((i, j))
                b = Bimg.getpixel((i, j))
                newIm2.putpixel((i, j), (int(T[r]), int(T[g]), int(T[b])))

        newIm2.save('./output/task_3/RGB分别直方图取平均直方图再映射.png')

    def prob_3():
        rgb = np.array(img, dtype=float)
        rgb /= 255
        width, height = img.size

        hsi = np.zeros((height, width, 3), dtype=float)
        for x in range(height):
            for y in range(width):
                hsi[x][y] = rgb2hsi(rgb[x, y])

        I_matrix = hsi[:, :, 2] * 255
        tempIm = Image.new('L', img.size)
        for x in range(height):
            for y in range(width):
                tempIm.putpixel((y, x), int(I_matrix[x][y]))
        eq_I_matrix = np.array(equalize_hist(tempIm), dtype=float) / 255
        hsi[:, :, 2] = eq_I_matrix

        newIm3 = Image.new('RGB', img.size)

        for x in range(height):
            for y in range(width):
                newIm3.putpixel((y, x), hsi2rgb(hsi[x][y]))
        newIm3.save('./output/task_3/RGB转HSI处理后再转回RGB.png')

    prob_1()
    prob_2()
    prob_3()


if __name__ == '__main__':
    # task_1()
    task_2()
    # task_3()
