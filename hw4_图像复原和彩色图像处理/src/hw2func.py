from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def get_gray_statistics(input_img):
    """
    Generate the pr_rk of the input_img

    Args:
        input_img (PIL.Image): The input image

    Return:
        pr_rk (list): 列表的下标是灰度值，列表里的元素是对应下标灰度值在input_img中的频率
    """
    res = [0] * 256

    for i in range(input_img.size[0]):
        for j in range(input_img.size[1]):
            res[input_img.getpixel((i, j))] += 1

    totalpixels = input_img.size[0] * input_img.size[1]
    pr_rk = [x / totalpixels for x in res]
    return pr_rk


def save_hist(input_img, title):
    """
    Save the histogram of the input_img with a title in output dir.

    Args:
        intput_img (PIL.Image): The input Image
        title (str): A string used as the title of the histogram

    Return:
        None
    """

    pr_rk = get_gray_statistics(input_img)

    plt.clf()

    plt.bar(range(256), pr_rk, edgecolor="gray")
    plt.xlim((0, 256))
    # 根据实际图片来设置y轴的范围使得图像更加直观
    plt.ylim((0, 0.02))
    plt.xlabel('gray level')  # 给 x 轴添加标签
    plt.ylabel('pr(rk)')  # 给 y 轴添加标签
    plt.title(title)  # 添加图形标题

    plt.savefig('./output/' + title + '.png')


def diff(img1, img2):
    """
    辅助函数，来把两张图片的直方图显示在同一张画布上，与作业无直接关系。
    """
    pr_rk1 = get_gray_statistics(img1)
    pr_rk2 = get_gray_statistics(img2)
    plt.clf()
    plt.plot(pr_rk1, 'bo')
    plt.plot(pr_rk2, 'ro')
    plt.xlim((0, 256))
    plt.ylim((0, 0.02))
    plt.xlabel('gray level')  # 给 x 轴添加标签
    plt.ylabel('pr(rk)')  # 给 y 轴添加标签
    plt.show()


def equalize_hist(input_img):
    """
    Processing the input_img for a equalized histogram.

    Args:
        input_img (PIL.image): The input image

    Return:
        newIm (PIL.image): A new Image with a more equalized histogram
    """
    pr_rk = get_gray_statistics(input_img)
    # 构建变换T
    T = [0] * 256
    for i in range(256):
        T[i] = round(255 * sum(pr_rk[0:i + 1]))

    newIm = Image.new('L', input_img.size)
    for i in range(input_img.size[0]):
        for j in range(input_img.size[1]):
            b = input_img.getpixel((i, j))
            newIm.putpixel((i, j), int(T[b]))

    return newIm


def filter2d(input_img, filter_):
    """
    Use filter_ to process the input_img

    Args:
        input_img (PIL.Image): The input image.
        filter_ (2-dimention np.array): The filter

    Return:
        newIm (PIL.Image): The new image after processing
    """
    m, n = filter_.shape
    a = int((m - 1) / 2)
    b = int((n - 1) / 2)

    img = np.array(input_img)
    height, width = img.shape

    # convert the original matrix
    # matrix = np.lib.pad(img, ((m-1,m-1), (n-1,n-1)), 'constant', constant_values=0)
    matrix = np.zeros((height + 2 * m - 2, width + 2 * n - 2))
    for x in range(height):
        for y in range(width):
            matrix[x + m - 1, y + n - 1] = img[x][y]

    newIm = Image.new('L', input_img.size)

    for x in range(height):
        for y in range(width):
            submatrix = matrix[x + m - 1 - a:x + m - 1 +
                               a + 1, y + n - 1 - b:y + n - 1 + b + 1]
            newvalue = round((submatrix * filter_).sum())
            newIm.putpixel((y, x), int(newvalue))

    return newIm


def high_boost(input_img, k):
    """
    Process the image with high_boost function

    Args:
        input_img (PIL.Image): the input image
        k (int): mask 前面的权值

    Return：
        newIm (PIL.Image): The new image after processing

    """
    # 请注意这里需要修改np.array的数据类型，从uint->int
    blur_img = np.array(filter2d(input_img, np.ones((3, 3)) / 9)).astype(int)
    old_img = np.array(input_img, dtype='int')

    newIm = Image.new('L', input_img.size)

    for x in range(old_img.shape[0]):
        for y in range(old_img.shape[1]):
            # mask 默认的化这个是要非负的，这个就比较坑了，所以要在前面转掉
            mask = old_img[x][y] - blur_img[x][y]
            newIm.putpixel((y, x), int(old_img[x][y] + k * mask))

    return newIm
