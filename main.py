
from PyQt5.QtWidgets import (
    QApplication, QMainWindow
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow
)
from PyQt5.QtCore import Qt

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing")
        self.initializeParameters()
        print("Parameters initialized")

        self.initializeUI()
        print("UI initialized")

        self.connectUI()
        print("UI connected")

    def initializeParameters(self):
        
        print("Params initialized")

    def initializeUI(self):
       
        print("UI components initialized")

    def connectUI(self):

        print("Ui connected")

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())
