from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout
import sys
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    
from GUI.modes import ModeSelector
from GUI.parameters_Panel import ParametersPanel
from GUI.ImageViewer import ImageViewer
from Core.NoiseAdder import add_uniform_noise, add_gaussian_noise, add_salt_pepper_noise
from Core.frequencyFilter import add_HighPass_filter, add_LowPass_filter
from Core.equalize import equalization, show_equalized_histograms
from Core.histogram import show_histograms
from Core.filters import average_filter, gaussian_filter, median_filter
from Core.gray import rgb_to_grayscale
import cv2
import numpy as np

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing")
        self.resize(int(1920*4/5), int(1080*4/5))
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

        self.mainInputViewer.setFixedWidth(int(self.width()*2/5))
        self.outputViewer.setFixedWidth(int(self.width()*2/5))

        main_widget = QWidget()


        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        modesLayout = QVBoxLayout()
        modesLayout.addLayout(self.modes_panel.createmodePanel())
        modesLayout.addWidget(self.parameters_panel)
        modesLayout.addStretch()

        imagesLayout = QHBoxLayout()
        inputLayout = QVBoxLayout()
        outputLayout = QVBoxLayout()
        inputLayout.addWidget(self.mainInputViewer,50)
        outputLayout.addWidget(self.outputViewer,50)
        imagesLayout.addLayout(inputLayout)
        imagesLayout.addLayout(outputLayout)


        main_layout.addLayout(modesLayout,20)
        main_layout.addLayout(imagesLayout,80)
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
            output_image=self.input_image
            show_histograms(self.input_image)

        elif self.current_mode=="Gray":
            output_image=rgb_to_grayscale(self.input_image)
        
        elif self.current_mode == "Noise & Filter":
            if "Noise" in self.current_parameters:
                noise_type = self.current_parameters.get("Noise")

                if noise_type == "Uniform":
                    min_val = self.current_parameters.get("Min:", -50)
                    max_val = self.current_parameters.get("Max:", 50)
                    output_image = add_uniform_noise(output_image, (min_val, max_val))

                elif noise_type == "Gaussian":
                    mean = self.current_parameters.get("Mean:", 0)
                    std_dev = self.current_parameters.get("Std Dev:", 10)
                    output_image = add_gaussian_noise(output_image, mean, std_dev)

                elif noise_type == "Salt & Pepper":
                    prob = self.current_parameters.get("prob:", 0.01)
                    salt_ratio = self.current_parameters.get("salt ratio:", 0.5)
                    output_image = add_salt_pepper_noise(output_image, prob, salt_ratio)

            if "Noise Filter" in self.current_parameters:
                filter_type = self.current_parameters.get("Noise Filter")
                kernel_size = self.current_parameters.get("Kernel Size:", 5)

                if filter_type == "Average":
                    output_image = average_filter(output_image, kernel_size)
                elif filter_type == "Gaussian":
                    sigma = self.current_parameters.get("Sigma:", 1.5)
                    output_image = gaussian_filter(output_image, kernel_size, sigma)
                elif filter_type == "Median":
                    output_image = median_filter(output_image, kernel_size)
        
        self.outputViewer.setImage(output_image)


    # def processImage(self,image):
    #     # test noise
    #     image = add_gaussian_noise(image, 2,100)
    #     self.outputViewer.setImage(image)

    #     #test frequency filters
    #     # image = add_LowPass_filter(image, 99)
    #     # image = add_HighPass_filter(image, 1)
    #     # self.outputViewer.setImage(image)


window = ImageProcessingApp()
window.show()
sys.exit(app.exec_())
