from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenuBar, QMenu, QAction, QFileDialog
from PyQt5.QtGui import QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint
import sys


RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        top = 400
        left = 400
        width = 800
        height = 600

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



if __name__ == "__main__":
        app = QApplication(sys.argv)
        window = Window()
        window.show()
        app.exec()