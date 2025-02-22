
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout
from GUI.modes import ModeSelector
from GUI.parameters_Panel import ParametersPanel
from GUI.ImageViewer import ImageViewer
from Core.NoiseAdder import add_uniform_noise,add_gaussian_noise,add_salt_pepper_noise

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing")
        self.initializeParameters()
        

        self.initializeUI()
      
        self.connectUI()
        self.setStyleSheet("""QMainWindow {background-color: #F5F5F0;}""")

    def initializeParameters(self):
        
        print("Params initialized")

    def initializeUI(self):
        self.modes_panel = ModeSelector()
        self.parameters_panel = ParametersPanel()
        self.mainInputViewer = ImageViewer()

        self.outputViewer = ImageViewer()
        self.outputViewer.setReadOnly(True)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        main_layout.addLayout(self.modes_panel.createmodePanel())
        main_layout.addWidget(self.mainInputViewer)
        main_layout.addWidget(self.outputViewer)
        main_layout.addWidget(self.parameters_panel)

        print("UI components initialized")

    def connectUI(self):
        self.modes_panel.mode_selected.connect(self.parameters_panel.updateGroupBox)
        self.mainInputViewer.imageChanged.connect(self.processImage)

        print("Ui connected")
    def processImage(self,image):
        image = add_gaussian_noise(image, 2,100)
        self.outputViewer.setImage(image)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())
