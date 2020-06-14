# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 01:12:54 2020

@author: karina
"""

from skimage import draw, transform, io, color, exposure
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import math

BRIGHT_RECTANGLE = 255
DARK_RECTANGLE = 50

def draw_haar_feature(w1, w2, w3, h, angle):
    img = np.zeros((h, w1 + w2 + w3), dtype=np.uint8)
    rr, cc = draw.rectangle((0, 0), extent=(h, w1), shape=img.shape)
    img[rr, cc] = BRIGHT_RECTANGLE
    rr, cc = draw.rectangle((0, w1), extent=(h, w2), shape=img.shape)
    img[rr, cc] = DARK_RECTANGLE
    if w3:
        rr, cc = draw.rectangle((0, w1 + w2), extent=(h, w3), shape=img.shape)
        img[rr, cc] = BRIGHT_RECTANGLE

    img = transform.rotate(img, 360 - angle, resize=True, preserve_range=False)

    return img


def create_dataset():
    dataset = []
    os.chdir("Car")
    img_list = os.listdir()
    for img in img_list:
        if os.path.isdir(img):
            continue
        image = io.imread(img)
        img_gray = color.rgb2gray(image)
        img_contrast = exposure.adjust_sigmoid(img_gray, cutoff=0.5, gain=60, inv=False)

        dataset.append((image, img_contrast))
    os.chdir(os.getcwd() + "/../")
    return dataset


def feature_detection(img, haar_feature_img):
    coord = (-1, -1)
    max_value = -1
    size = haar_feature_img.shape
    for x in range(img.shape[0] - size[0]):
        for y in range(img.shape[1] - size[1]):
            cur_value = detect(img[x:x + size[0], y:y + size[1]], haar_feature_img)
            if cur_value > max_value:
                max_value = cur_value
                coord = x, y
    return coord, max_value


def detect(img, haar_feature_img):
    if img.shape != haar_feature_img.shape:
        print("ERROR: SIZES NOT EQUAL")
        raise IndexError

    # этот параметр подбирается. есть граница, выше которой ничего определятся не будет
    # если он слишком мал, то будет находится ненужное (по идее)
    threshold = 35 * haar_feature_img.shape[0] * haar_feature_img.shape[1] / 255  # ~17.5
    bright, dark = 0, 0
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if haar_feature_img[x][y] == BRIGHT_RECTANGLE / 255:
                bright += img[x][y]
            elif haar_feature_img[x][y] == DARK_RECTANGLE / 255:
                dark += img[x][y]
    if bright - dark > threshold:
        return bright - dark
    return -1


def draw_angled_rec(x0, y0, width, height, angle, img, color):
    _angle = angle * math.pi / 180.0
    b = math.cos(_angle) * 0.5
    a = math.sin(_angle) * 0.5
    pt0 = (int(x0 - a * height - b * width),
           int(y0 + b * height - a * width))
    pt1 = (int(x0 + a * height - b * width),
           int(y0 - b * height - a * width))
    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

    cv2.line(img, pt0, pt1, color, 1)
    cv2.line(img, pt1, pt2, color, 1)
    cv2.line(img, pt2, pt3, color, 1)
    cv2.line(img, pt3, pt0, color, 1)


def highlight_feature(img, c_img, coord, size, angle, haar_feature, n):
    c_img = color.gray2rgb(c_img)
    if coord != (-1, -1):
        if size[0] > size[1]:
            x0 = coord[1] + size[1] / 2
            y0 = coord[0] + size[0] / 2
            draw_angled_rec(x0, y0, size[0], size[1], angle, img, (255, 0, 0))
            draw_angled_rec(x0, y0, size[0], size[1], angle, c_img, (1, 0, 0))
        else:
            x0 = coord[1] + size[0] / 2
            y0 = coord[0] + size[1] / 2
            draw_angled_rec(x0, y0, size[1], size[0], angle, img, (255, 0, 0))
            draw_angled_rec(x0, y0, size[1], size[0], angle, c_img, (1, 0, 0))

    label = "Car detected" if coord[0] != -1 else "No car detected"
    fig, ax = plt.subplots(1, 3, squeeze=False)
    ax[0][0].set(title='Original', xlabel='Found in total')
    ax[0][0].imshow(img)
    ax[0][1].set(title='High contrast', xlabel=label)
    ax[0][1].imshow(c_img)
    ax[0][2].set(title='Haar-like feature', xlabel="Angle: {}".format(angle))
    ax[0][2].imshow(haar_feature)
    io.show()

    os.chdir("Car")
    os.chdir("Detected")
    img = img[:, :, ::-1]

    cv2.imwrite(str(n) + " " + str(angle) + ".jpg", img)
    os.chdir(os.getcwd() + "/../../")


if __name__ == '__main__':

    car_dataset = create_dataset()

    # диапазоны углов для каждой фотки, чтобы не проверять все углы (очень долго)
    for n, image in enumerate(car_dataset):

        if n == 0:
            start, end = -90, -5
        elif n == 1:
            start, end = -120, -80
        elif n == 2:
            start, end = -40, -20
        else:
            start, end = 0, 30

        for i in range(start, end, 3):
            haar_feature = draw_haar_feature(w1=5, w2=5, w3=5, h=7, angle=i)
            coordinates, value = feature_detection(image[1], haar_feature)
            print('Image', n, ', Angle: ', i)
            print(coordinates, value)
            highlight_feature(image[0], image[1], coordinates, haar_feature.shape, i, haar_feature, n)

