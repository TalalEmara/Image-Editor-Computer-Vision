import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QFileDialog, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal

from GUI.styles import GENERAL_STYLE, LABEL_STYLE


class ImageViewer(QWidget):
    imageChanged = pyqtSignal(object)

    def __init__(self):
        super().__init__()

        self.initializeUI()
        self.styleUI()
        self.setupLayout()
        self.setReadOnly(False)  # Default: Uploading enabled

    def initializeUI(self):
        self.image = None
        self.isReadOnly = True  # Controls whether image uploading is allowed
        self.image_label = QLabel("Double-click to upload an image", self)
        self.image_label.setAlignment(Qt.AlignCenter)

    def styleUI(self):
        self.setStyleSheet(GENERAL_STYLE)
        self.image_label.setStyleSheet(LABEL_STYLE)
        self.image_label.setStyleSheet("border: 2px dashed gray; padding: 20px;")

    def setupLayout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and not self.isReadOnly:
            self.openImage()

    def openImage(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp)")
        if file_path:
            self.displayImage(cv2.imread(file_path))

    def displayImage(self, image):
        """Displays an image in the QLabel."""
        if image is not None:
            self.image = image
            self.imageChanged.emit(self.image)  # Emit signal when image changes

            # Convert OpenCV BGR to RGB for display
            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            height, width, channels = image_rgb.shape
            bytes_per_line = channels * width
            q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            self.image_label.setPixmap(
                pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
            )

    def setImage(self, image: np.ndarray):
        self.displayImage(image)

    def setReadOnly(self, enabled: bool):
        """Enables or disables image uploading."""
        self.isReadOnly = enabled
        if enabled:
            self.image_label.setText("Image viewing mode only")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()


    viewer.show()
    sys.exit(app.exec_())
