import sys
import cv2
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QFileDialog, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt



from GUI.styles import (
    GENERAL_STYLE, GROUP_BOX_STYLE, LABEL_STYLE,
    COMBO_BOX_STYLE, SLIDER_STYLE, SPIN_BOX_STYLE
)


class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()

        self.initializeUI()
        self.styleUI()
        self.setupLayout()
        self.setReadOnly(False)

    def initializeUI(self):
        self.image = None
        self.isReadOnly = True 
        self.image_label = QLabel("Double-click to upload an image", self)
        self.image_label.setAlignment(Qt.AlignCenter)

    def styleUI(self):
        self.setWindowTitle("Image Viewer")
        self.resize(600, 400)

        self.setStyleSheet(GENERAL_STYLE)
        self.image_label.setStyleSheet(LABEL_STYLE)
        self.image_label.setStyleSheet("border: 2px dashed gray; padding: 20px;")

    def setupLayout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and  not(self.isReadOnly):
            self.openImage()

    def openImage(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp)")
        if file_path:
            self.displayImage(file_path)

    def displayImage(self, file_path):
        self.image = cv2.imread(file_path)
        if self.image is not None:
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # Convert OpenCV BGR to RGB
            height, width, channels = image.shape
            bytes_per_line = channels * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            self.image_label.setPixmap(
                pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

    def setReadOnly(self, enabled: bool):
        self.isReadOnly = enabled
        if enabled:
            self.image_label.setText("Image viewing mode only")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()


    viewer.show()
    sys.exit(app.exec_())
