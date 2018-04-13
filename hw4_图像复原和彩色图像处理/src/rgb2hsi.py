import numpy as np
from statistics import mean
from math import sqrt


def rgb2hsi(point):
    def getH():
        r, g, b = point
        up = 0.5 * (r - g + r - b)
        # 剧坑！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        down = sqrt((r - g) * (r - g) + (r - b) * (g - b)) + 0.000000001
        xita = np.arccos(up / down)
        return xita / (2 * np.pi) if b <= g else (2 * np.pi - xita) / (2 * np.pi)

    def getS():
        # 剧坑！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        if sum(point) == 0:
            return 1
        return 1 - 3 * min(point) / sum(point)

    def getI():
        return mean(point)

    return (getH(), getS(), getI())


def hsi2rgb(point):
    h, s, i = point
    h *= (2 * np.pi)

    if 0 <= h < 2 * np.pi / 3:
        b = i * (1 - s)
        r = i * (1 + (s * np.cos(h) / (np.cos(np.pi / 3 - h) + 0.000001)))
        g = 3 * i - (r + b)
        try:
            return (int(r * 255), int(g * 255), int(b * 255))
        except ValueError:
            print(point)
    elif 2 * np.pi / 3 <= h < 4 * np.pi / 3:
        h -= 2 * np.pi / 3
        r = i * (1 - s)
        g = i * (1 + (s * np.cos(h) / (np.cos(np.pi / 3 - h) + 0.000001)))
        b = 3 * i - (r + g)
        try:
            return (int(r * 255), int(g * 255), int(b * 255))
        except ValueError:
            print(point)
    elif 4 * np.pi / 3 <= h < 2 * np.pi:
        h -= 4 * np.pi / 3
        g = i * (1 - s)
        b = i * (1 + (s * np.cos(h) / (np.cos(np.pi / 3 - h) + 0.000001)))
        r = 3 * i - (g + b)
        try:
            return (int(r * 255), int(g * 255), int(b * 255))
        except ValueError:
            print(point)
