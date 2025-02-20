from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QComboBox, QSlider, QSpinBox, QApplication
)
class ParametersPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.current_group_box = None
        
        self.parameter_panel = QVBoxLayout(self)

    
    def updateGroupBox(self, selected_mode):
        if self.current_group_box:
            self.parameter_panel.removeWidget(self.current_group_box)
            self.current_group_box.setParent(None)
            self.current_group_box.deleteLater()
            self.current_group_box = None
        
        if selected_mode == "Noise":
            noise_controls = [
                ("Label", "Noise Type:"),
                ("ComboBox", ["Uniform", "Gaussian", "Salt & Pepper"]),
                ("Label", "Noise:"),
                ("Slider", (0, 100))
            ]
            self.current_group_box = self.createGroupBox("Noise Parameters", noise_controls)

        elif selected_mode == "Filter Noise":
            filter_controls = [
                ("Label", "Filter Type:"),
                ("ComboBox", ["Average", "Gaussian", "Median"]),
                ("Label", "Kernel Size:"),
                ("SpinBox", (1, 15))
            ]
            self.current_group_box = self.createGroupBox("Filter Parameters", filter_controls)

        elif selected_mode == "Edge Detection":
            edge_controls = [
                ("Label", "Edge Detector:"),
                ("ComboBox", ["Sobel", "Roberts", "Prewitt", "Canny"]),
                ("Label", "Threshold:"),
                ("Slider", (0, 255))
            ]
            self.current_group_box = self.createGroupBox("Edge Detection Parameters", edge_controls)

        elif selected_mode == "Threshold":
            thresholding_controls = [
                ("Label", "Method:"),
                ("ComboBox", ["Local", "Global"]),
                ("Label", "Threshold Value:"),
                ("Slider", (0, 255))
            ]
            self.current_group_box = self.createGroupBox("Thresholding Parameters", thresholding_controls)

        if self.current_group_box:
            self.parameter_panel.addWidget(self.current_group_box)
            self.parameter_panel.update()  
            self.update()  

        
    def createGroupBox(self, title, controls):
        group = QGroupBox(title)
        layout = QVBoxLayout()

        for control_type, control_data in controls:
            if control_type == "Label":
                label = QLabel(control_data)
                layout.addWidget(label)
            elif control_type == "ComboBox":
                combo = QComboBox()
                combo.addItems(control_data)
                combo.currentIndexChanged.connect(lambda _, cb=combo: self.updateParams(cb))
                layout.addWidget(combo)
            elif control_type == "Slider":
                slider = QSlider(Qt.Horizontal)
                slider.setRange(*control_data)
                layout.addWidget(slider)
            elif control_type == "SpinBox":
                spin = QSpinBox()
                spin.setRange(*control_data)
                layout.addWidget(spin)

        group.setLayout(layout)
        return group

    def updateParams(self, sender):
        selected_item = sender.currentText()
        
        if selected_item == "Uniform":
            self.updateGroupBox("Noise")
            self.current_group_box.layout().addWidget(QLabel("Min Value:"))
            min_slider = QSlider(Qt.Horizontal)
            min_slider.setRange(0, 100)
            self.current_group_box.layout().addWidget(min_slider)

            self.current_group_box.layout().addWidget(QLabel("Max Value:"))
            max_slider = QSlider(Qt.Horizontal)
            max_slider.setRange(0, 100)
            self.current_group_box.layout().addWidget(max_slider)

        elif selected_item == "Gaussian":
            self.updateGroupBox("Noise")
            self.current_group_box.layout().addWidget(QLabel("Mean:"))
            mean_slider = QSlider(Qt.Horizontal)
            mean_slider.setRange(0, 100)
            self.current_group_box.layout().addWidget(mean_slider)

            self.current_group_box.layout().addWidget(QLabel("Standard Deviation:"))
            std_slider = QSlider(Qt.Horizontal)
            std_slider.setRange(0, 100)
            self.current_group_box.layout().addWidget(std_slider)

        elif selected_item == "Salt & Pepper":
            self.updateGroupBox("Noise")
            self.current_group_box.layout().addWidget(QLabel("Salt:"))
            salt_slider = QSlider(Qt.Horizontal)
            salt_slider.setRange(0, 100)
            self.current_group_box.layout().addWidget(salt_slider)

            self.current_group_box.layout().addWidget(QLabel("Pepper:"))
            pepper_slider = QSlider(Qt.Horizontal)
            pepper_slider.setRange(0, 100)
            self.current_group_box.layout().addWidget(pepper_slider)

