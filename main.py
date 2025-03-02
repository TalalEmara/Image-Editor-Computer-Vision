from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QSlider, QLabel
import sys
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    
from GUI.modes import ModeSelector
from GUI.parameters_Panel import ParametersPanel
from GUI.ImageViewer import ImageViewer
from Core.NoiseAdder import add_uniform_noise, add_gaussian_noise, add_salt_pepper_noise
from Core.frequencyFilter import add_HighPass_filter, add_LowPass_filter, generate_hybrid_image, generate_hybrid_imageK
from Core.equalize import equalization, show_equalized_histograms
from Core.histogram import show_histograms, get_histogram_widget
from Core.filters import average_filter, gaussian_filter, median_filter
from Core.gray import rgb_to_grayscale
from Core.thresholding import globalThreshold,sauvolaThresholding
from Core.hybrid import hybrid_image
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

        self.input_image = None
        #could be array for better handling
        self.secondaryInput = None
        
        self.initializeUI()
        self.connectUI()
        self.setStyleSheet("""QMainWindow {background-color: #F5F5F0;}""")

    def initializeUI(self):
        self.modes_panel = ModeSelector()

        self.parameters_panel = ParametersPanel()
        self.mainInputViewer = ImageViewer()
        self.secondaryInputViewer = ImageViewer()

        self.outputViewer = ImageViewer()
        self.outputViewer.setReadOnly(True)

        self.mainInputViewer.setFixedWidth(int(self.width()*2/5))
        self.outputViewer.setFixedWidth(int(self.width()*2/5))

        main_widget = QWidget()


        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        self.modesLayout = QVBoxLayout()
        self.modesLayout.addLayout(self.modes_panel.createmodePanel())
        self.modesLayout.addWidget(self.parameters_panel)
        self.modesLayout.addStretch()

        imagesLayout = QHBoxLayout()
        self.inputLayout = QVBoxLayout()
        outputLayout = QVBoxLayout()
        self.inputLayout.addWidget(self.mainInputViewer,50)
        outputLayout.addWidget(self.outputViewer,50)
        imagesLayout.addLayout(self.inputLayout)
        imagesLayout.addLayout(outputLayout)


        main_layout.addLayout(self.modesLayout, 20)
        main_layout.addLayout(imagesLayout,80)
        print("UI components initialized")

    def connectUI(self):
        self.modes_panel.mode_selected.connect(self.onModeChanged)
        self.parameters_panel.parameter_changed.connect(self.onParameterChanged)
        self.mainInputViewer.imageChanged.connect(self.onImageChanged)
        self.secondaryInputViewer.imageChanged.connect(self.onSecondaryHybridChanged)
        # self.mainInputViewer.imageChanged.connect(self.processImage)
        
        print("UI connected")

    from PyQt5.QtWidgets import QSlider, QLabel

    def onModeChanged(self, mode): #for debug onlly
        self.secondaryInputViewer.hide()  # Hide secondary image viewer initially
        self.current_mode = mode
        self.parameters_panel.updateGroupBox(mode)
        self.current_parameters = self.parameters_panel.parameters.copy()
        if self.input_image is not None:
            QApplication.processEvents()
            self.processImage()

        if self.current_mode == "Hybrid Images":
            self.mixButton = QPushButton("Mix")
            self.inputLayout.addWidget(self.mixButton)
            self.inputLayout.addWidget(self.secondaryInputViewer, 50)
            self.secondaryInputViewer.show()

            # Create sliders
            self.weightSlider = QSlider()
            self.lowPassSlider = QSlider()
            self.highPassSlider = QSlider()

            # Set slider ranges
            self.weightSlider.setRange(0, 100)  # Weight: 0.0 - 1.0 (scaled by 100)
            self.lowPassSlider.setRange(1, 100)  # Cutoff freq for low-pass filter
            self.highPassSlider.setRange(1, 100)  # Cutoff freq for high-pass filter

            # Set default values
            self.weightSlider.setValue(50)  # Default weight = 0.5
            self.lowPassSlider.setValue(3)  # Default low-pass cutoff
            self.highPassSlider.setValue(70)  # Default high-pass cutoff

            # Labels for sliders
            self.weightLabel = QLabel("Weight: 0.5")
            self.lowPassLabel = QLabel("Low Pass Cutoff: 3")
            self.highPassLabel = QLabel("High Pass Cutoff: 70")

            # Add sliders and labels to layout
            self.inputLayout.addWidget(self.weightLabel)
            self.inputLayout.addWidget(self.weightSlider)
            self.inputLayout.addWidget(self.lowPassLabel)
            self.inputLayout.addWidget(self.lowPassSlider)
            self.inputLayout.addWidget(self.highPassLabel)
            self.inputLayout.addWidget(self.highPassSlider)

            # Connect sliders to update labels
            self.weightSlider.valueChanged.connect(self.updateSliderLabels)
            self.lowPassSlider.valueChanged.connect(self.updateSliderLabels)
            self.highPassSlider.valueChanged.connect(self.updateSliderLabels)

            # Connect mix button
            self.mixButton.clicked.connect(self.mix)

    def updateSliderLabels(self):
        """ Update the labels with current slider values """
        weight = self.weightSlider.value() / 100  # Scale back to 0.0 - 1.0
        low_cutoff = self.lowPassSlider.value()
        high_cutoff = self.highPassSlider.value()

        self.weightLabel.setText(f"Weight: {weight:.2f}")
        self.lowPassLabel.setText(f"Low Pass Cutoff: {low_cutoff}")
        self.highPassLabel.setText(f"High Pass Cutoff: {high_cutoff}")

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
    def onSecondaryHybridChanged(self, image):
        self.secondaryInput = image.copy()

    def mix(self):
        if self.input_image is None or self.secondaryInput is None:
            print("Error: One or both images are None!")
            return  # Avoid crashing

        weight = self.weightSlider.value() / 100  # Convert back to 0.0 - 1.0
        low_cutoff = self.lowPassSlider.value()
        high_cutoff = self.highPassSlider.value()

        self.outputViewer.setImage(hybrid_image(rgb_to_grayscale(self.input_image), rgb_to_grayscale(self.secondaryInput)))
            # generate_hybrid_image(self.input_image, self.secondaryInput, weight, low_cutoff, high_cutoff))
            # generate_hybrid_imageK(self.input_image, self.secondaryInput))


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
            self.parameters_panel.updateEqualizedHistogram(output_image)

        elif self.current_mode=="Histogram":
            self.parameters_panel.updateHistogram(self.input_image)

        elif self.current_mode=="Gray":
            output_image=rgb_to_grayscale(self.input_image)
 
        elif self.current_mode == "Threshold":
           
            threshold_type = self.current_parameters.get("Threshold")
            
            if threshold_type == "Global":
                threshold_value = int(self.current_parameters.get("Threshold:", 127))
                output_image = globalThreshold(rgb_to_grayscale(self.input_image), threshold_value)
            
            elif threshold_type == "Local":
                window_size = int(self.current_parameters.get("window size", 101))
                if window_size % 2 == 0:  
                    window_size += 1
                output_image = sauvolaThresholding(rgb_to_grayscale(self.input_image), window_size)
                    

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
