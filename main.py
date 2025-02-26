from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout
from GUI.modes import ModeSelector
from GUI.parameters_Panel import ParametersPanel
from GUI.ImageViewer import ImageViewer
from Core.NoiseAdder import add_uniform_noise, add_gaussian_noise, add_salt_pepper_noise
from Core.frequencyFilter import add_HighPass_filter, add_LowPass_filter
from Core.equalize import equalization, show_equalized_histograms
from Core.histogram import show_histograms
from Core.gray import rgb_to_grayscale
import cv2
import numpy as np

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing")
        self.current_mode = None
        self.current_parameters = {}
        self.original_image = None
        
        self.initializeUI()
        self.connectUI()
        self.setStyleSheet("""QMainWindow {background-color: #F5F5F0;}""")

    def initializeUI(self):
        self.modes_panel = ModeSelector()
        self.parameters_panel = ParametersPanel()
        self.mainInputViewer = ImageViewer()

        self.outputViewer = ImageViewer()
        self.outputViewer.setReadOnly(True)
        self.mainInputViewer.setFixedSize(300, 400)
        self.outputViewer.setFixedSize(300, 400)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        main_layout.addLayout(self.modes_panel.createmodePanel())
        main_layout.addWidget(self.mainInputViewer)
        main_layout.addWidget(self.outputViewer)
        main_layout.addWidget(self.parameters_panel)

        print("UI components initialized")

    def connectUI(self):
        self.modes_panel.mode_selected.connect(self.onModeChanged)
        self.parameters_panel.parameter_changed.connect(self.onParameterChanged)
        self.mainInputViewer.imageChanged.connect(self.onImageChanged)
        # self.mainInputViewer.imageChanged.connect(self.processImage)
        
        print("UI connected")
    
    def onModeChanged(self, mode):
        self.current_mode = mode
        self.parameters_panel.updateGroupBox(mode)
        self.current_parameters = self.parameters_panel.parameters.copy()
        if self.input_image is not None:
            QApplication.processEvents()  
            self.processImage()

    
    def onParameterChanged(self, parameters):
        self.current_parameters = parameters
        if self.input_image is not None:
            self.processImage()
    
    def onImageChanged(self, image):
        self.input_image = image.copy()
        self.processImage()

    def processImage(self):
        output_image = self.input_image.copy()
    
        if self.current_mode == "Frequency Domain Filter":
            if "Frequency Domain Filter" in self.current_parameters:
                filter_type = self.current_parameters.get("Frequency Domain Filter")
                cut_off = self.current_parameters.get("CutOff Freq:", 50)
                
                if filter_type == "Low Pass Filter":
                    output_image = add_LowPass_filter(output_image, cut_off)
                elif filter_type == "High Pass Filter":
                    output_image = add_HighPass_filter(output_image, cut_off)
        
        elif self.current_mode == "Equalization":
            output_image = equalization(self.input_image)
            show_equalized_histograms(output_image)

        elif self.current_mode=="Histogram":
            output_image=show_histograms(self.input_image)

        elif self.current_mode=="Gray/Color":
            output_image=rgb_to_grayscale(self.input_image)


        self.outputViewer.setImage(output_image)


    # def processImage(self,image):
    #     # test noise
    #     image = add_gaussian_noise(image, 2,100)
    #     self.outputViewer.setImage(image)

    #     #test frequency filters
    #     # image = add_LowPass_filter(image, 99)
    #     # image = add_HighPass_filter(image, 1)
    #     # self.outputViewer.setImage(image)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())
