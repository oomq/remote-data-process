import sys
from PyQt5.QtWidgets import QGraphicsScene, QApplication, QGraphicsView, QMainWindow, QAction, QFileDialog, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt, QRectF
import cv2
from mainwindows import Ui_MainWindow
from mmengine.fileio import dump
from rich import print_json
from mmcls.apis import ImageClassificationInferencer
import numpy as np
from osgeo import gdal
# try:###
#     from osgeo import gdal
# except ImportError:
#     gdal = None

swim_configs = r"E:\omq\omqproject\mmpretrain-1.x\swim/swinv2-large-w16_in21k-pre_16xb64_in1k-256px.py"
swim_weights = r"E:\omq\omqproject\mmpretrain-1.x\swim/best_accuracy_top1_epoch_150.pth"

class ImageDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # self.ui.pic = QGraphicsView(self)
        # self.setCentralWidget(self.ui.pic)
        self.ui.pic.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.ui.pic.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.scene = QGraphicsScene(self)
        self.ui.pic.setScene(self.scene)
        self.ui.pic.setDragMode(QGraphicsView.ScrollHandDrag)
        self.ui.pic.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.ui.pic.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        self.ui.pushButton.clicked.connect(self.openImage)
        self.ui.pushButton_2.clicked.connect(self.detectImage)
        self.ui.pic.wheelEvent = self.wheelEvent


        # self.inferencer = ImageClassificationInferencer(
        #     model=swim_configs,
        #     weights=swim_weights,
        # )



    def wheelEvent(self, event):
        zoom_factor = 1.2

        if event.angleDelta().y() > 0:
            self.ui.pic.scale(zoom_factor, zoom_factor)
        else:
            self.ui.pic.scale(1 / zoom_factor, 1 / zoom_factor)

    def openImage(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "",
        "Image Files (*.jpg *.png *.jpeg *.bmp *.tif *.tiff);;All Files (*)", options=options)

        if file_name:
            # self.current_image = cv2.imread(file_name)

            ds = gdal.Open(file_name,gdal.GA_ReadOnly)
            if ds is None:
                raise Exception(f"Unable to open file: {file_name}")
            self.current_image = np.einsum("ijk->jki", ds.ReadAsArray())


            pixmap = QPixmap(file_name)
            self.scene.clear()
            self.scene.addPixmap(pixmap)
            self.ui.pic.setSceneRect(QRectF(pixmap.rect()))

    def detectImage(self):
        if self.current_image is not None:
            detected_image = self.current_image.copy()
            results = self.inferencer(detected_image, show=False)
            cv2.rectangle(detected_image, (50, 50), (200, 200), (0, 255, 0), 2)
            self.ui.txt.setText("Detected Categories: {}".format(results[0]["pred_class"]))

            self.displayImage(detected_image)

    def displayImage(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        # self.ui.pic.setPixmap(pixmap)
        self.ui.pic.setSceneRect(QRectF(pixmap.rect()))

def main():
    app = QApplication(sys.argv)
    window = ImageDetectionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
