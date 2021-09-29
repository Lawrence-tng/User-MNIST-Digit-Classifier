import matplotlib
import numpy
import numpy as np
from numpy import asarray
import matplotlib as plt  # plt.show
import matplotlib.pyplot as plp
from matplotlib.pyplot import subplots
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.datasets import mnist
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenuBar, QMenu, QAction, QFileDialog
from PyQt5.QtGui import QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint
import sys
from PIL import Image, ImageFilter
import math
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])
print(x_train[0].shape)
plp.imshow(x_train[0], cmap='gray')
plp.show()

def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva

def shape():
    x = imageprepare('RE.png')
    x = asarray(x)
    x = x * 255
    x.reshape
    print(x)
    print(x.shape)
    y = np.reshape(x, (28,28))
    print(y.shape)
    print(y)


shape()
