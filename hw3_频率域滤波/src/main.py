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

def dft2d (input_img, flags):
    """
    通过flags来指定二维的傅里叶正变换或者逆变换
    通过离散二维DFT的可分性用三层循环来实现二维DFT

    if flags == 'dft': 返回一个 m X n 大小的complex的矩阵
    if flags == 'idft': 返回一个 m X n 大小的int的灰度值矩阵
    """
    m, n = input_img.shape
    
    if flags == 'dft':
        Fuv = np.zeros((m, n), dtype="complex128")
        for x in range(m):
            print ("dft: x: ", x)
            Fuv[x,:] = dft1d(input_img[x])

        for y in range(n):
            print ("dft: y: ", y)
            Fuv[:,y] = dft1d(Fuv[:, y])
        return Fuv

    elif flags == 'idft':
        fxy = np.zeros((m, n), dtype="complex128")
        for x in range(m):
            print ("idft: x: ", x)
            fxy[x,:] = idft1d(input_img[x])

        for y in range(n):
            print ("idft: y: ", y)
            fxy[:,y] = idft1d(fxy[:, y])
        return fxy.real.astype(int)


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


def Fourier_Spectrum(input_img):
    """
    输入： img一张图片的矩阵表示
    返回： img这张图片的傅里叶频谱图
    """
    m, n = input_img.shape

    # Step 1: 中心化
    input_img = center(input_img)

    # Step 2: 进行傅里叶变换
    Fuv = dft2d(input_img, flags='dft')

    # Step 3: 对Fuv的每一项取模，算出对应的谱， 取log降低过高的值，其中取log的公式使用书本的例4.13
    vfunc = np.vectorize(lambda val : 1 + math.log(np.abs(val)))
    spectrum_img = np.zeros((m,n), dtype="float")
    for x in range(m):
        spectrum_img[x,:] = vfunc(Fuv[x,:])

    # Step 4: 标定
    spectrum_img = demarcate(spectrum_img)

    return spectrum_img


def filter2d_freq(input_img, filter_):
    """
    使用课本的过程来进行频率域滤波
    """

    def padding(input_img, P, Q):
        m, n = input_img.shape
        pad_img = np.zeros((P, Q), dtype=float)
        for x in range(m):
            pad_img[x,0:n] = input_img[x,0:n]
        return pad_img

    m, n = input_img.shape
    c, d = filter_.shape
    pad_img = padding(input_img, m+c-1, n+c-1)
    pad_img = center(pad_img)

    Fuv = dft2d(pad_img, "dft")
    # Fuv = np.fft.fft2(pad_img)

    # generate H(u,v)
    pad_mask = padding(filter_, m+c-1, n+d-1)

    # 为了消去黑边，将滤波器中心移到左上角
    shift_c = int((c-1) / 2)
    shift_d = int((d-1) / 2)
    pad_mask = np.append(pad_mask, pad_mask[0:shift_c], 0)
    pad_mask = np.delete(pad_mask, [x for x in range(shift_c)], 0)
    pad_mask = np.append(pad_mask, pad_mask[:, 0:shift_d], 1)
    pad_mask = np.delete(pad_mask, [x for x in range(shift_d)], 1)

    pad_mask = center(pad_mask)
    
    Huv = dft2d(pad_mask, "dft")
    # Huv = np.fft.fft2(pad_mask)

    Guv = Huv * Fuv
    gxy = center(dft2d(Guv, "idft"))
    # gxy = center(np.fft.ifft2(Guv).real)
    output_img = gxy[0:m, 0:n]

    output_img = demarcate(output_img)

    return output_img

if __name__ == '__main__':
    # 计算img的傅里叶频谱图
    img = np.array(Image.open('./29.png'), dtype=int)
    show_save_img(Fourier_Spectrum(img), name="Fourier_Spectrum_img")

    # 使用dft对img进行傅里叶变换，再用idft进行反变换，输出反变换之后的图片
    img = np.array(Image.open('./29.png'), dtype=int)
    Fuv = dft2d(img, "dft")
    fxy = dft2d(Fuv, "idft")
    show_save_img(fxy, "dft_and_idft_result")

    # 频率域滤波实验
    img = np.array(Image.open('./29.png'), dtype=int)
    show_save_img(filter2d_freq(img, np.ones((3,3)) / 9), "meanfilter3x3")
    img = np.array(Image.open('./29.png'), dtype=int)
    show_save_img(filter2d_freq(img, np.ones((7,7)) / 49), "meanfilter7x7")
    img = np.array(Image.open('./29.png'), dtype=int)
    show_save_img(filter2d_freq(img, np.ones((11,11)) / 121), "meanfilter11x11")
    img = np.array(Image.open('./29.png'), dtype=int)
    show_save_img(filter2d_freq(img, np.array([[1,1,1], [1,-8,1], [1,1,1]])), "Lap2")




    