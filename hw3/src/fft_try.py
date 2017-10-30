from PIL import Image
import matplotlib as plot
import numpy as np
import math

def show_save_img(input_img, name):
    """
    根据input_img的这个矩阵购进一个新的图像并且保存为name.png
    """
    m, n = input_img.shape

    new_img = Image.new('L', (n, m))
    for x in range(m):
        for y in range(n):
            new_img.putpixel((y,x), int(input_img[x, y]))
    
    new_img.show()
    new_img.save('./output/' + name + '.png')


def dft1d(f):
    """
    使用矩阵运算的方法来计算一维的傅里叶变换
    """
    M = f.shape[0]
    W = np.array([[np.exp(-1j*2*np.pi*u*x/M) for x in range(M)] for u in range(M)])
    return W.dot(f)

def idft1d(F):
    """
    使用矩阵运算的方法来计算一维的傅里叶逆变换
    """
    M = F.shape[0]
    W = np.array([[np.exp(1j*2*np.pi*u*x/M) for x in range(M)] for u in range(M)])
    return W.dot(F) / M


def center(input_img):
    """
    通过每个元素乘-1的x+y次方来进行中心化
    """
    m, n = input_img.shape
    for x in range(m):
        for y in range(n):
            input_img[x, y] *= (-1) ** (x+y)
    return input_img

def demarcate(input_img):
    """
    将img矩阵的灰度值标定到[0,255]
    """
    dx = (input_img.max() - input_img.min()) / 256
    m, n = input_img.shape
    for x in range(m):
        for y in range(n):
            input_img[x,y] = int(input_img[x,y] / dx)
    return input_img


def FFT(f):
    f = f.astype(float)
    N = f.shape[0]
    if N % 2 > 0:
        raise ValueError("size of f must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return dft1d(f)
    else:
        X_even = FFT(f[::2])
        X_odd = FFT(f[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N / 2] * X_odd,
                               X_even + factor[N / 2:] * X_odd])

def FFT2d(input_img):
    m,n = input_img.shape
    Fuv = np.zeros((m, n), dtype="complex128")
    for x in range(m):
        print ("dft: x: ", x)
        Fuv[x,:] = FFT(input_img[x])

    for y in range(n):
        print ("dft: y: ", y)
        Fuv[:,y] = FFT(Fuv[:, y])
    return Fuv

def IFFT2d(input_img):
    m,n = input_img.shape
    Fuv = np.zeros((m, n), dtype="complex128")
    input_img = input_img.conjugate()
    for x in range(m):
        print ("dft: x: ", x)
        Fuv[x,:] = FFT(input_img[x])

    for y in range(n):
        print ("dft: y: ", y)
        Fuv[:,y] = FFT(Fuv[:, y])
    return Fuv.conjugate().real / (m * n)

def Fourier_Spectrum(input_img):
    """
    输入： img一张图片的矩阵表示
    返回： img这张图片的傅里叶频谱图
    """
    m, n = input_img.shape

    # Step 1: 中心化
    input_img = center(input_img)

    # Step 2: 进行傅里叶变换
    Fuv = FFT2d(input_img)

    # Step 3: 对Fuv的每一项取模，算出对应的谱， 取log降低过高的值，其中取log的公式使用书本的例4.13
    vfunc = np.vectorize(lambda val : 1 + math.log(np.abs(val)))
    spectrum_img = np.zeros((m,n), dtype="float")
    for x in range(m):
        spectrum_img[x,:] = vfunc(Fuv[x,:])

    # Step 4: 标定
    spectrum_img = demarcate(spectrum_img)

    return spectrum_img

if __name__ == '__main__':
    # 计算img的傅里叶频谱图
    img = np.array(Image.open('./29.png'), dtype=int)
    show_save_img(Fourier_Spectrum(img), name="fft_Fourier_Spectrum_img")

    # 使用fft对img进行傅里叶变换，再用ifft进行反变换，输出反变换之后的图片
    img = np.array(Image.open('./29.png'), dtype=int)
    Fuv = FFT2d(img)
    fxy = IFFT2d(Fuv)
    show_save_img(fxy, "fft_and_ifft_result")
