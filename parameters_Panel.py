from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QComboBox, QSlider, 
    QSpinBox, QHBoxLayout, QSizePolicy
)


from styles import (
    GENERAL_STYLE, GROUP_BOX_STYLE, LABEL_STYLE, 
    COMBO_BOX_STYLE, SLIDER_STYLE, SPIN_BOX_STYLE
)


class ParametersPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parameter_panel = QVBoxLayout(self)
        self.parameter_panel.setContentsMargins(0, 0, 0, 0)
        self.current_group_boxes = []
        self.setupUI()
        
    def setupUI(self):
        self.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Minimum
        )
        self.stylingUi(self)

    def stylingUi(self, widget):
       
        self.setStyleSheet(GENERAL_STYLE)

        for group_box in self.findChildren(QGroupBox):
            group_box.setStyleSheet(GROUP_BOX_STYLE)

        for label in self.findChildren(QLabel):
            label.setStyleSheet(LABEL_STYLE)
        
        for combo in self.findChildren(QComboBox):
            combo.setStyleSheet(COMBO_BOX_STYLE)

        for slider in self.findChildren(QSlider):
            slider.setStyleSheet(SLIDER_STYLE)

        for spinbox in self.findChildren(QSpinBox):
            spinbox.setStyleSheet(SPIN_BOX_STYLE)
       
    
    def createSliderWithSpinBox(self, label_text, min_val, max_val):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        label = QLabel(label_text)
        label.setMinimumWidth(100)
     
        
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setFixedWidth(50)
        slider.setStyleSheet(SLIDER_STYLE)
        
        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setFixedWidth(70)
        spinbox.setStyleSheet(SPIN_BOX_STYLE)
        
        slider.valueChanged.connect(spinbox.setValue)
        spinbox.valueChanged.connect(slider.setValue)
        
        layout.addWidget(label)
        layout.addWidget(slider)
        layout.addWidget(spinbox)
        
        return container

    def createGroupBox(self, config):
        group = QGroupBox(config['title'])
        group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        group.setStyleSheet(GROUP_BOX_STYLE)
        
        layout = QVBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(10, 10, 10, 10)

        if 'type_selector' in config:
            selector_config = config['type_selector']
            
            type_container = QWidget()
            type_layout = QHBoxLayout(type_container)
            type_layout.setContentsMargins(0, 0, 0, 0)
            
            type_label = QLabel(selector_config['label'])
            type_label.setStyleSheet(LABEL_STYLE)
            
            type_combo = QComboBox()
            type_combo.addItems(selector_config['options'])
            type_combo.setStyleSheet(COMBO_BOX_STYLE)
            
            type_layout.addWidget(type_label)
            type_layout.addWidget(type_combo)
            layout.addWidget(type_container)

            controls_widget = QWidget()
            controls_layout = QVBoxLayout(controls_widget)
            controls_layout.setContentsMargins(0, 5, 0, 0)
            controls_layout.setSpacing(5)
            layout.addWidget(controls_widget)

            def updateControls(selected_type):
                for i in reversed(range(controls_layout.count())):
                    controls_layout.itemAt(i).widget().setParent(None)

                if selected_type in selector_config['controls']:
                    for control in selector_config['controls'][selected_type]:
                        if control['type'] == 'slider':
                            control_widget = self.createSliderWithSpinBox(
                                control['label'],
                                *control['range']
                            )
                            controls_layout.addWidget(control_widget)
                
                controls_widget.adjustSize()
                group.adjustSize()
                self.adjustSize()

            type_combo.currentTextChanged.connect(updateControls)
            updateControls(selector_config['options'][0])

        group.setLayout(layout)
        return group

    def updateGroupBox(self, selected_mode):
        # Clear existing group boxes
        for group_box in self.current_group_boxes:
            self.parameter_panel.removeWidget(group_box)
            group_box.setParent(None)
            group_box.deleteLater()
        self.current_group_boxes.clear()

        if selected_mode == "Noise & Filter":
            noise_config = {
                'title': 'Noise',
                'type_selector': {
                    'label': 'Noise Type:',
                    'options': ['Uniform', 'Gaussian', 'Salt & Pepper'],
                    'controls': {
                        'Uniform': [
                            {'label': 'Min Value:', 'type': 'slider', 'range': (0, 255)},
                            {'label': 'Max Value:', 'type': 'slider', 'range': (0, 255)}
                        ],
                        'Gaussian': [
                            {'label': 'Mean:', 'type': 'slider', 'range': (0, 255)},
                            {'label': 'Std Dev:', 'type': 'slider', 'range': (0, 50)}
                        ],
                        'Salt & Pepper': [
                            {'label': 'Salt:', 'type': 'slider', 'range': (0, 100)},
                            {'label': 'Pepper:', 'type': 'slider', 'range': (0, 100)}
                        ]
                    }
                }
            }
            
            filter_config = {
                'title': 'Noise Filter',
                'type_selector': {
                    'label': 'Filter Type:',
                    'options': ['None', 'Average', 'Gaussian', 'Median'],
                    'controls': {
                        'Average': [
                            {'label': 'Kernel Size:', 'type': 'slider', 'range': (3, 15)}
                        ],
                        'Gaussian': [
                            {'label': 'Kernel Size:', 'type': 'slider', 'range': (3, 15)},
                            {'label': 'Sigma:', 'type': 'slider', 'range': (1, 10)}
                        ],
                        'Median': [
                            {'label': 'Kernel Size:', 'type': 'slider', 'range': (3, 15)}
                        ]
                    }
                }
            }

            self.current_group_boxes.extend([
                self.createGroupBox(noise_config),
                self.createGroupBox(filter_config)
            ])

        elif selected_mode == "Edge Detection":
            edge_config = {
                'title': 'Edge Detection Parameters',
                'type_selector': {
                    'label': 'Edge Detector:',
                    'options': ['Sobel', 'Roberts', 'Prewitt', 'Canny'],
                    'controls': {
                        'Sobel': [
                            {'label': 'Threshold:', 'type': 'slider', 'range': (0, 255)}
                        ],
                        'Roberts': [
                            {'label': 'Threshold:', 'type': 'slider', 'range': (0, 255)}
                        ],
                        'Prewitt': [
                            {'label': 'Threshold:', 'type': 'slider', 'range': (0, 255)}
                        ],
                        'Canny': [
                            {'label': 'Low Threshold:', 'type': 'slider', 'range': (0, 255)},
                            {'label': 'High Threshold:', 'type': 'slider', 'range': (0, 255)}
                        ]
                    }
                }
            }
            
            self.current_group_boxes.append(self.createGroupBox(edge_config))

        for group_box in self.current_group_boxes:
            self.parameter_panel.addWidget(group_box)

        self.parameter_panel.addStretch()
        self.adjustSize()

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication, QMainWindow
    
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Parameters Panel Test")
            self.parameters_panel = ParametersPanel()
            self.setCentralWidget(self.parameters_panel)
            self.parameters_panel.updateGroupBox("Noise & Filter") 
            # self.parameters_panel.updateGroupBox("Edge Detection") 
            self.setStyleSheet("""
                
                QMainWindow {
                    background-color: #1E1E2E;
                }
            """)
    
    app = QApplication([])
    mainWin = MainWindow()
    mainWin.show()
    app.exec_()


    ##if i want to access each one alone I think this is better but comment it inistially 

# from PyQt5.QtCore import Qt, pyqtSignal
# from PyQt5.QtWidgets import (
#     QWidget, QVBoxLayout, QGroupBox, QLabel, QComboBox, QSlider, 
#     QSpinBox, QHBoxLayout, QSizePolicy
# )

# from styles import (
#     GENERAL_STYLE, GROUP_BOX_STYLE, LABEL_STYLE, 
#     COMBO_BOX_STYLE, SLIDER_STYLE, SPIN_BOX_STYLE
# )


# class ParametersPanel(QWidget):
    
    
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.parameter_panel = QVBoxLayout(self)
#         self.parameter_panel.setContentsMargins(0, 0, 0, 0)
#         self.current_mode = None
        
#         self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
    
#         self.createComponents()
#         self.setupConnections()
#         self.stylingUi()
      
#         self.parameter_panel.addStretch()

#     def createSliderWithSpinBox(self, label_text, min_val, max_val, default_value=None):
#         container = QWidget()
#         layout = QHBoxLayout(container)
#         layout.setContentsMargins(0, 0, 0, 0)
#         layout.setSpacing(5)
        
#         label = QLabel(label_text)
#         label.setMinimumWidth(100)
        
#         slider = QSlider(Qt.Horizontal)
#         slider.setRange(min_val, max_val)
#         slider.setMinimumWidth(110)
        
#         spinbox = QSpinBox()
#         spinbox.setRange(min_val, max_val)
#         spinbox.setFixedWidth(70)
      
#         if default_value is not None:
#             slider.setValue(default_value)
#             spinbox.setValue(default_value)
     
#         slider.valueChanged.connect(spinbox.setValue)
#         spinbox.valueChanged.connect(slider.setValue)
        
#         layout.addWidget(label)
#         layout.addWidget(slider)
#         layout.addWidget(spinbox)
        
#         return container, slider, spinbox

#     def createComponents(self):
#        #Noise group
#         self.noise_group = QGroupBox("Noise")
#         self.noise_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
#         noise_layout = QVBoxLayout(self.noise_group)
#         noise_layout.setSpacing(5)
#         noise_layout.setContentsMargins(10, 10, 10, 10)
      
#         noise_type_container = QWidget()
#         noise_type_layout = QHBoxLayout(noise_type_container)
#         noise_type_layout.setContentsMargins(0, 0, 0, 0)
        
#         type_label = QLabel("Noise Type:")
#         self.noise_type_combo = QComboBox()
#         self.noise_type_combo.addItems(['Uniform', 'Gaussian', 'Salt & Pepper'])
        
#         noise_type_layout.addWidget(type_label)
#         noise_type_layout.addWidget(self.noise_type_combo)
#         noise_layout.addWidget(noise_type_container)
     
#         self.uniform_container = QWidget()
#         uniform_layout = QVBoxLayout(self.uniform_container)
#         uniform_layout.setContentsMargins(0, 0, 0, 0)
        
#         self.gaussian_container = QWidget()
#         gaussian_layout = QVBoxLayout(self.gaussian_container)
#         gaussian_layout.setContentsMargins(0, 0, 0, 0)
        
#         self.salt_pepper_container = QWidget()
#         salt_pepper_layout = QVBoxLayout(self.salt_pepper_container)
#         salt_pepper_layout.setContentsMargins(0, 0, 0, 0)
     
#         uniform_min_container, self.uniform_min_slider, self.uniform_min_spinbox = self.createSliderWithSpinBox(
#             "Min Value:", 0, 255, 0
#         )
#         uniform_max_container, self.uniform_max_slider, self.uniform_max_spinbox = self.createSliderWithSpinBox(
#             "Max Value:", 0, 255, 100
#         )
#         uniform_layout.addWidget(uniform_min_container)
#         uniform_layout.addWidget(uniform_max_container)
        
#         gaussian_mean_container, self.gaussian_mean_slider, self.gaussian_mean_spinbox = self.createSliderWithSpinBox(
#             "Mean:", 0, 255, 128
#         )
#         gaussian_stddev_container, self.gaussian_stddev_slider, self.gaussian_stddev_spinbox = self.createSliderWithSpinBox(
#             "Std Dev:", 0, 50, 10
#         )
#         gaussian_layout.addWidget(gaussian_mean_container)
#         gaussian_layout.addWidget(gaussian_stddev_container)
        
#         salt_container, self.salt_pepper_salt_slider, self.salt_pepper_salt_spinbox = self.createSliderWithSpinBox(
#             "Salt:", 0, 100, 5
#         )
#         pepper_container, self.salt_pepper_pepper_slider, self.salt_pepper_pepper_spinbox = self.createSliderWithSpinBox(
#             "Pepper:", 0, 100, 5
#         )
#         salt_pepper_layout.addWidget(salt_container)
#         salt_pepper_layout.addWidget(pepper_container)
        
#         noise_layout.addWidget(self.uniform_container)
#         noise_layout.addWidget(self.gaussian_container)
#         noise_layout.addWidget(self.salt_pepper_container)
      
#         self.gaussian_container.hide()
#         self.salt_pepper_container.hide()
        
#         self.parameter_panel.addWidget(self.noise_group)
#         self.noise_group.hide()  # Hidden by default
        
#         # Filter group
#         self.filter_group = QGroupBox("Noise Filter")
#         self.filter_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
#         filter_layout = QVBoxLayout(self.filter_group)
#         filter_layout.setSpacing(5)
#         filter_layout.setContentsMargins(10, 10, 10, 10)

#         filter_type_container = QWidget()
#         filter_type_layout = QHBoxLayout(filter_type_container)
#         filter_type_layout.setContentsMargins(0, 0, 0, 0)
        
#         filter_type_label = QLabel("Filter Type:")
#         self.filter_type_combo = QComboBox()
#         self.filter_type_combo.addItems(['None', 'Average', 'Gaussian', 'Median'])
        
#         filter_type_layout.addWidget(filter_type_label)
#         filter_type_layout.addWidget(self.filter_type_combo)
#         filter_layout.addWidget(filter_type_container)
        
#         self.average_filter_container = QWidget()
#         average_layout = QVBoxLayout(self.average_filter_container)
#         average_layout.setContentsMargins(0, 0, 0, 0)
        
#         self.gaussian_filter_container = QWidget()
#         gaussian_filter_layout = QVBoxLayout(self.gaussian_filter_container)
#         gaussian_filter_layout.setContentsMargins(0, 0, 0, 0)
        
#         self.median_filter_container = QWidget()
#         median_layout = QVBoxLayout(self.median_filter_container)
#         median_layout.setContentsMargins(0, 0, 0, 0)
        
#         avg_kernel_container, self.average_kernel_slider, self.average_kernel_spinbox = self.createSliderWithSpinBox(
#             "Kernel Size:", 3, 15, 3
#         )
#         average_layout.addWidget(avg_kernel_container)
        
#         gauss_kernel_container, self.gaussian_filter_kernel_slider, self.gaussian_filter_kernel_spinbox = self.createSliderWithSpinBox(
#             "Kernel Size:", 3, 15, 3
#         )
#         gauss_sigma_container, self.gaussian_filter_sigma_slider, self.gaussian_filter_sigma_spinbox = self.createSliderWithSpinBox(
#             "Sigma:", 1, 10, 1
#         )
#         gaussian_filter_layout.addWidget(gauss_kernel_container)
#         gaussian_filter_layout.addWidget(gauss_sigma_container)
        
#         median_kernel_container, self.median_kernel_slider, self.median_kernel_spinbox = self.createSliderWithSpinBox(
#             "Kernel Size:", 3, 15, 3
#         )
#         median_layout.addWidget(median_kernel_container)

#         filter_layout.addWidget(self.average_filter_container)
#         filter_layout.addWidget(self.gaussian_filter_container)
#         filter_layout.addWidget(self.median_filter_container)
        
#         self.average_filter_container.hide()
#         self.gaussian_filter_container.hide()
#         self.median_filter_container.hide()
        
#         self.parameter_panel.addWidget(self.filter_group)
#         self.filter_group.hide()  # Hidden by default
        
#         # EDGE DETECTOR
#         self.edge_group = QGroupBox("Edge Detection Parameters")
#         self.edge_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
#         edge_layout = QVBoxLayout(self.edge_group)
#         edge_layout.setSpacing(5)
#         edge_layout.setContentsMargins(10, 10, 10, 10)
        
#         edge_type_container = QWidget()
#         edge_type_layout = QHBoxLayout(edge_type_container)
#         edge_type_layout.setContentsMargins(0, 0, 0, 0)
        
#         edge_type_label = QLabel("Edge Detector:")
#         self.edge_type_combo = QComboBox()
#         self.edge_type_combo.addItems(['Sobel', 'Roberts', 'Prewitt', 'Canny'])
        
#         edge_type_layout.addWidget(edge_type_label)
#         edge_type_layout.addWidget(self.edge_type_combo)
#         edge_layout.addWidget(edge_type_container)
      
#         self.sobel_container = QWidget()
#         sobel_layout = QVBoxLayout(self.sobel_container)
#         sobel_layout.setContentsMargins(0, 0, 0, 0)
        
#         self.roberts_container = QWidget()
#         roberts_layout = QVBoxLayout(self.roberts_container)
#         roberts_layout.setContentsMargins(0, 0, 0, 0)
        
#         self.prewitt_container = QWidget()
#         prewitt_layout = QVBoxLayout(self.prewitt_container)
#         prewitt_layout.setContentsMargins(0, 0, 0, 0)
        
#         self.canny_container = QWidget()
#         canny_layout = QVBoxLayout(self.canny_container)
#         canny_layout.setContentsMargins(0, 0, 0, 0)
   
#         sobel_thresh_container, self.sobel_threshold_slider, self.sobel_threshold_spinbox = self.createSliderWithSpinBox(
#             "Threshold:", 0, 255, 128
#         )
#         sobel_layout.addWidget(sobel_thresh_container)
        
#         roberts_thresh_container, self.roberts_threshold_slider, self.roberts_threshold_spinbox = self.createSliderWithSpinBox(
#             "Threshold:", 0, 255, 128
#         )
#         roberts_layout.addWidget(roberts_thresh_container)
        
#         prewitt_thresh_container, self.prewitt_threshold_slider, self.prewitt_threshold_spinbox = self.createSliderWithSpinBox(
#             "Threshold:", 0, 255, 128
#         )
#         prewitt_layout.addWidget(prewitt_thresh_container)
        
#         canny_low_container, self.canny_low_threshold_slider, self.canny_low_threshold_spinbox = self.createSliderWithSpinBox(
#             "Low Threshold:", 0, 255, 50
#         )
#         canny_high_container, self.canny_high_threshold_slider, self.canny_high_threshold_spinbox = self.createSliderWithSpinBox(
#             "High Threshold:", 0, 255, 150
#         )
#         canny_layout.addWidget(canny_low_container)
#         canny_layout.addWidget(canny_high_container)
     
#         edge_layout.addWidget(self.sobel_container)
#         edge_layout.addWidget(self.roberts_container)
#         edge_layout.addWidget(self.prewitt_container)
#         edge_layout.addWidget(self.canny_container)
        
#         self.roberts_container.hide()
#         self.prewitt_container.hide()
#         self.canny_container.hide()
        
#         self.parameter_panel.addWidget(self.edge_group)
#         self.edge_group.hide() 

#     def setupConnections(self):
#         self.noise_type_combo.currentTextChanged.connect(self.onNoiseTypeChanged)
#         self.filter_type_combo.currentTextChanged.connect(self.onFilterTypeChanged)
#         self.edge_type_combo.currentTextChanged.connect(self.onEdgeTypeChanged)




#     def onNoiseTypeChanged(self, noise_type):
#         self.uniform_container.hide()
#         self.gaussian_container.hide()
#         self.salt_pepper_container.hide()
#         if noise_type == "Uniform":
#             self.uniform_container.show()
#         elif noise_type == "Gaussian":
#             self.gaussian_container.show()
#         elif noise_type == "Salt & Pepper":
#             self.salt_pepper_container.show()


#     def onFilterTypeChanged(self, filter_type):
#         self.average_filter_container.hide()
#         self.gaussian_filter_container.hide()
#         self.median_filter_container.hide()
#         if filter_type == "Average":
#             self.average_filter_container.show()
#         elif filter_type == "Gaussian":
#             self.gaussian_filter_container.show()
#         elif filter_type == "Median":
#             self.median_filter_container.show()


#     def onEdgeTypeChanged(self, edge_type):
#         self.sobel_container.hide()
#         self.roberts_container.hide()
#         self.prewitt_container.hide()
#         self.canny_container.hide()
#         if edge_type == "Sobel":
#             self.sobel_container.show()
#         elif edge_type == "Roberts":
#             self.roberts_container.show()
#         elif edge_type == "Prewitt":
#             self.prewitt_container.show()
#         elif edge_type == "Canny":
#             self.canny_container.show()

#     def stylingUi(self):
#         self.setStyleSheet(GENERAL_STYLE)

#         for group_box in self.findChildren(QGroupBox):
#             group_box.setStyleSheet(GROUP_BOX_STYLE)

#         for label in self.findChildren(QLabel):
#             label.setStyleSheet(LABEL_STYLE)
        
#         for combo in self.findChildren(QComboBox):
#             combo.setStyleSheet(COMBO_BOX_STYLE)

#         for slider in self.findChildren(QSlider):
#             slider.setStyleSheet(SLIDER_STYLE)

#         for spinbox in self.findChildren(QSpinBox):
#             spinbox.setStyleSheet(SPIN_BOX_STYLE)

#     def updateGroupBox(self, selected_mode):
#         self.current_mode = selected_mode
   
#         self.noise_group.hide()
#         self.filter_group.hide()
#         self.edge_group.hide()
       
#         if selected_mode == "Noise & Filter":
#             self.noise_group.show()
#             self.filter_group.show()
         
            
#         elif selected_mode == "Edge Detection":
#             self.edge_group.show()
            

# if __name__ == "__main__":
#     from PyQt5.QtWidgets import QApplication, QMainWindow
    
#     class MainWindow(QMainWindow):
#         def __init__(self):
#             super().__init__()
#             self.setWindowTitle("Parameters Panel Test")
#             self.parameters_panel = ParametersPanel()
#             self.setCentralWidget(self.parameters_panel)
            
#             self.parameters_panel.updateGroupBox("Noise & Filter")
            
#             self.setStyleSheet("""
#                 QMainWindow {
#                     background-color: #1E1E2E;
#                 }
#             """)
    
#     app = QApplication([])
#     mainWin = MainWindow()
#     mainWin.show()
#     app.exec_()