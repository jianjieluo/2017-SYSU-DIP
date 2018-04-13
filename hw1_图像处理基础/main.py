# Python3 version 3.5.2

from PIL import Image
from math import floor
import os

def save_and_show(img, name):
    """
    Show the img and save it to ./output dir
    """
    if os.path.exists('./output') is not True:
        os.makedirs('./output')
    img.save(os.path.join('./output', name))
    img.show()


def get_src_point_info(despos, th, tw):
    """
    Generate the int i,j of the lefttop point in the src image,
    and calculate the decimal part u, v of height and width respectively.
    """
    # t = srcsize / dessize
    # 先竖向再横向
    srch = (despos[0] + 1) * th - 1
    srcw = (despos[1] + 1) * tw - 1
    return floor(srch), srch-floor(srch), floor(srcw), srcw-floor(srcw)


def scale(input_img, size):
    """
    Use bi-linear for interpolation to scale the input_img.

    Args:
        input_img (two-dimensional matrices): storing input image.
        size (tuple): (width, height) defining the spatial resolution of output.

    Return:
        output_img (PIL.Image.Image object): output image.
    """
    twidth = len(input_img[0]) / size[0]
    theight = len(input_img) / size[1]

    newIm = Image.new("L", (size[0], size[1]))

    for w in range(size[0]):
        for h in range(size[1]):
            i, u, j, v = get_src_point_info((h, w), th = theight, tw = twidth)
            
            if u == 0 and v == 0:
                newIm.putpixel((w,h), input_img[i][j])
            elif u == 0 and v != 0:
                pixel = (1-v) * input_img[i][j] + v * input_img[i][j+1]
                newIm.putpixel((w,h), int(pixel))
            elif u != 0 and v == 0:
                pixel = (1-u) * input_img[i][j] + u * input_img[i+1][j]
                newIm.putpixel((w,h), int(pixel))
            else:
                pixel = (1-u)*(1-v)*input_img[i][j] + (1-u)*v*input_img[i][j+1] \
                    + u*(1-v)*input_img[i+1][j] + u*v*input_img[i+1][j+1]        
                newIm.putpixel((w,h), int(pixel))
    return newIm


def quantize(input_img, level):
    """
    Function used to quantize an image.

    Args:
        input_img (two-dimensional matrices): storing input image.
        level (int): in [1, 256] defining the number of gray levels of output.

    Return:
        output_img (PIL.Image.Image object): output image.
    """
    size_per_level = int(256 / level)

    levelgray = {
        0: 0,
        level-1: 255
    }
    for i in range(1, level-1):
        levelgray[i] = int(sum(range(i * size_per_level, (i + 1) * size_per_level)) / size_per_level )
    
    # check the dict and get the corresponding gray level content.
    width = len(input_img[0])
    height = len(input_img)
    newIm = Image.new("L", (width, height))
    for h in range(height):
        for w in range(width):
            newIm.putpixel((w, h), levelgray[int(input_img[h][w] / size_per_level)])
    return newIm

    
if __name__ == '__main__':
    img = Image.open('./29.png')
    
    # Generate 2-dimensional matrix of the input image.
    input_img = []
    for i in range(img.size[1]):
        row = [ img.getpixel((j, i)) for j in range(img.size[0]) ]
        input_img.append(row)

    # Prob 2.2.1
    save_and_show(scale(input_img, (192, 128)), "scale_192x128.png")
    save_and_show(scale(input_img, (96, 64)), "scale_96x64.png")
    save_and_show(scale(input_img, (48, 32)), "scale_48x32.png")
    save_and_show(scale(input_img, (24, 16)), "scale_24x16.png")
    save_and_show(scale(input_img, (12, 8)), "scale_12x8.png")

    # Prob 2.2.2 - 2.2.4
    save_and_show(scale(input_img, (300, 200)), "scale_300x200.png")
    save_and_show(scale(input_img, (450, 300)), "scale_450x300.png")
    save_and_show(scale(input_img, (500, 200)), "scale_500x200.png")

    # Prob 2.3
    save_and_show(quantize(input_img, 128), "graylevel_128.png")
    save_and_show(quantize(input_img, 32), "graylevel_32.png")
    save_and_show(quantize(input_img, 8), "graylevel_8.png")
    save_and_show(quantize(input_img, 4), "graylevel_4.png")
    save_and_show(quantize(input_img, 2), "graylevel_2.png")
