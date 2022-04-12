import time

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
from keras import layers, optimizers

#first
np.random.seed(0)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# f, ax = plp.subplots(1, 10, figsize=(20, 20))

# for i in range(10):
#     sample = x_train[y_train == i][0]
#     ax[i].imshow(sample, cmap="gray")
#     ax[i].set_title("Lable: {}".format(i), fontsize=16)

# plp.show()

y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# for i in range(10):
#     print(y_train[i])


# reshape data
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
# reshapes vector to 2d

# Create model 
model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
# compile model
opt = optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
batch_size = 32
epochs = 10
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)
model.save('final_model.h5')

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Loss: {}, Test Acc: {}".format(test_loss, test_acc))

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# single example
# random_idx = np.random.choice(len(x_test))
# x_sample = x_test[random_idx]
# y_true = np.argmax(y_test, axis=1)
# y_sample_true = y_true[random_idx]
# y_sample_pred_class = y_pred_classes[random_idx]
#
# plp.title("predicted: {}, True: {}".format(y_sample_pred_class, y_sample_true), fontsize=16)
# plp.imshow(x_sample.reshape(28, 28), cmap='gray')
# plp.show()


# GUI



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

def load_image(filename):
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0
    return img

#classify
def classifyNumber():
    x = [imageprepare('RE.png')]
    x_drawn = asarray(x)
    x_drawn = x_drawn * 255
    x_drawn = x_drawn.reshape((x_drawn.shape[0], -1))
    #model = load_model('final_model.h5')
    prediction = model.predict_classes(x_drawn)
    # pred_classes = np.argmax(prediction, axis=1)
    plp.title("predicted: {}".format(prediction), fontsize=16)
    plp.imshow(x_drawn.reshape(28, 28), cmap='gray')
    print(x_drawn[0])
    plp.show()



class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        top = 400
        left = 400
        width = 200
        height = 200

        self.setGeometry(top, left, width, height)
        self.setWindowTitle("MNIST DIGIT CLASSIFIER")

        self.image = QImage(self.size(), QImage.Format_Grayscale8)
        self.image.fill(Qt.white)

        self.drawing = False
        self.brushSize = 20
        self.brushColor = Qt.black

        self.lastPoint = QPoint()

        mainMenu = self.menuBar()
        FileMenu = mainMenu.addMenu("File")

        saveAction = QAction("Save", self)
        saveAction.triggered.connect(self.save)
        FileMenu.addAction(saveAction)

        clearAction = QAction("clear", self)
        clearAction.setShortcut("c")
        clearAction.triggered.connect(self.clear)
        mainMenu.addAction(clearAction)

    def mousePressEvent(self, event):

        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

    def save(self):
        #filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG(*.png);;JPEG(*.jpg *.jpeg);; ALL Files(*.*)")
        filePath = "RE.png"
        if filePath == "":
            return
        self.image.save(filePath)
        time.sleep(2)
        classifyNumber()

    def clear(self):
        self.image.fill(Qt.white)
        window.update()



if __name__ == "__main__":
        app = QApplication(sys.argv)
        window = Window()
        window.show()
        app.exec()

