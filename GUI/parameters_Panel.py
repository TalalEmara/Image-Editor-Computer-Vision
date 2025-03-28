from PyQt5.QtCore import Qt,QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QComboBox, QSlider, QPushButton,
    QSpinBox, QHBoxLayout, QSizePolicy,QDoubleSpinBox
)
from Core.histogram import show_histograms
from Core.equalize  import  show_equalized_histograms
from PyQt5.QtCore import pyqtSignal

from GUI.styles import (
    GENERAL_STYLE, GROUP_BOX_STYLE, LABEL_STYLE,
    COMBO_BOX_STYLE, SLIDER_STYLE, SPIN_BOX_STYLE,BUTTON_STYLE
)

class ParametersPanel(QWidget):
    parameter_changed = pyqtSignal(dict)


    def __init__(self, parent=None):
        super().__init__(parent)
        self.parameter_panel = QVBoxLayout(self)
        self.parameter_panel.setContentsMargins(0, 0, 0, 0)
        self.current_group_boxes = []
        self.parameters = {}

        self.setupUI()

    def update_parameter(self, key, value):
        self.parameters[key] = value
        self.parameter_changed.emit(self.parameters)


    def setupUI(self):
        # self.setSizePolicy(
        #     QSizePolicy.Preferred,
        #     QSizePolicy.Minimum
        # )
        self.stylingUi(self)

    def stylingUi(self, widget):
        self.setStyleSheet(GENERAL_STYLE)
        # debug function
        # self.setAttribute(Qt.WA_StyledBackground, True)
        # self.setStyleSheet("background-color:#2D2D2D;")

        # self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.parameter_panel.setAlignment(Qt.AlignTop)

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


    def createSliderWithSpinBox(self, label_text, min_val, max_val, step=1, default=None):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        label = QLabel(label_text)
        label.setMinimumWidth(100)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setSingleStep(int(step))  # Ensure step is an integer
        slider.setMinimumWidth(100)
        slider.setStyleSheet(SLIDER_STYLE)

        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setSingleStep(int(step))  # Ensure step is an integer
        spinbox.setFixedWidth(70)
        spinbox.setStyleSheet(SPIN_BOX_STYLE)

        # slider.valueChanged.connect(spinbox.setValue)
        default_value = default if default is not None else min_val
        slider.setValue(default_value)
        spinbox.setValue(default_value)

        spinbox.valueChanged.connect(slider.setValue)
        slider.sliderReleased.connect(lambda: spinbox.setValue(slider.value()))

        layout.addWidget(label)
        layout.addWidget(slider)
        layout.addWidget(spinbox)

        return container


    def createDoubleSpinBox(self, label_text, min_val, max_val, step=0.01):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        label = QLabel(label_text)
        label.setMinimumWidth(100)

        spinbox = QDoubleSpinBox()
        spinbox.setStyleSheet(SPIN_BOX_STYLE)
        spinbox.setDecimals(2)
        spinbox.setSingleStep(step)
        spinbox.setRange(min_val, max_val)
        spinbox.setFixedWidth(70)

        def emit_parameter_change():
            self.update_parameter(label_text, spinbox.value())

        spinbox.valueChanged.connect(emit_parameter_change)

        layout.addWidget(label)
        layout.addWidget(spinbox)

        return container


    def createControlWidget(self, control, controls_layout):
        if control['type'] == 'slider':
            control_widget = self.createSliderWithSpinBox(
                control['label'], control['range'][0], control['range'][1], 
                step=control.get('step', 1),  
                default=control.get('default', control['range'][0])  
            )

            controls_layout.addWidget(control_widget)
            slider = control_widget.findChildren(QSlider)[0]
            spinbox = control_widget.findChildren(QSpinBox)[0]

            def emit_parameter_change():
                self.update_parameter(control['label'], spinbox.value())

            slider.sliderReleased.connect(emit_parameter_change)
            spinbox.valueChanged.connect(emit_parameter_change)
            slider.sliderReleased.connect(lambda: spinbox.setValue(slider.value()))
            self.update_parameter(control['label'], spinbox.value())

        elif control['type'] == 'doubleSpin':
            control_widget = self.createDoubleSpinBox(control['label'], control['range'][0], control['range'][1], control.get('step', 0.01))
            controls_layout.addWidget(control_widget)
            spinbox = control_widget.findChildren(QDoubleSpinBox)[0]

            def emit_parameter_change():
                self.update_parameter(control['label'], spinbox.value())

            spinbox.valueChanged.connect(emit_parameter_change)
            self.update_parameter(control['label'], spinbox.value())

        elif control['type'] == 'button':
            control_widget = QPushButton(control['label'])
            control_widget.setStyleSheet(BUTTON_STYLE)
            control_widget.setFixedWidth(120)
            button_layout = QHBoxLayout()
            button_layout.addWidget(control_widget)
            button_layout.setAlignment(Qt.AlignCenter)
            controls_layout.addLayout(button_layout)

    def createGroupBox(self, config):
        group = QGroupBox(config['title'])
        group.setStyleSheet(GROUP_BOX_STYLE)

        layout = QVBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(10, 10, 10, 10)

        if 'type_selector' in config and config['type_selector'] is not None:
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
                        self.createControlWidget(control, controls_layout)

                

            type_combo.currentTextChanged.connect(lambda text: self.update_parameter(config['title'], text))
            type_combo.currentTextChanged.connect(updateControls)
            

            self.update_parameter(config['title'], selector_config['options'][0])
            updateControls(selector_config['options'][0])

        else:
            controls_widget = QWidget()
            controls_layout = QVBoxLayout(controls_widget)
            controls_layout.setContentsMargins(0, 5, 0, 0)
            controls_layout.setSpacing(5)
            layout.addWidget(controls_widget)

            for control in config.get('controls', []):
                self.createControlWidget(control, controls_layout)

        group.setLayout(layout)
        return group


    def updateGroupBox(self, selected_mode):
        # Clear existing group boxes
        for group_box in self.current_group_boxes:
            self.parameter_panel.removeWidget(group_box)
            group_box.setParent(None)
            group_box.deleteLater()
        self.current_group_boxes.clear()
        self.parameter_panel.takeAt(0)
        self.parameters.clear()

        if selected_mode == "Histogram":
            self.histogram_group = QWidget()
            self.histogram_layout = QVBoxLayout(self.histogram_group)
            self.parameter_panel.addWidget(self.histogram_group)
            self.current_group_boxes.append(self.histogram_group)

        elif selected_mode == "Equalization":
            self.Equalized_histogram_group = QWidget()
            self.Equalized_histogram_layout = QVBoxLayout(self.Equalized_histogram_group)
            self.parameter_panel.addWidget(self.Equalized_histogram_group)
            self.current_group_boxes.append(self.Equalized_histogram_group)


        elif selected_mode == "Noise & Filter":
            noise_config = {
                'title': 'Noise',
                'type_selector': {
                    'label': 'Noise Type:',
                    'options': ['Uniform', 'Gaussian', 'Salt & Pepper'],
                    'controls': {
                        'Uniform': [{'label': 'Min:', 'type': 'slider', 'range': (-255, 255)},
                            {'label': 'Max:', 'type': 'slider', 'range': (-255, 255)}],
                        'Gaussian': [
                            {'label': 'Mean:', 'type': 'slider', 'range': (-50, 50)},
                            {'label': 'Std Dev:', 'type': 'slider', 'range': (0, 100)}
                        ],
                        'Salt & Pepper': [
                             {'label': 'prob:', 'type': 'doubleSpin', 'range': (0.01, 1.0), 'step': 0.01},
                    {'label': 'salt ratio:', 'type': 'doubleSpin', 'range': (0.01, 1.0), 'step': 0.01}
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

            self.current_group_boxes.append(self.createGroupBox(noise_config))
            self.current_group_boxes.append(self.createGroupBox(filter_config))
        
        elif selected_mode == "Edge Detection":
            edge_config = {
                'title': 'Edge Detection',
                'type_selector': {
                    'label': 'Edge Detector:',
                    'options': ['Sobel', 'Roberts', 'Prewitt', 'Canny'],
                    'controls': {
                        'Sobel': [
                            {'label': 'Kernal size:', 'type': 'slider', 'range': (3, 9), 'step': 2, 'default': 3}
                        ],
                        'Prewitt': [
                            {'label': 'Kernal size:', 'type': 'slider', 'range': (3, 9), 'step': 2, 'default': 3}
                        ],
                        'Canny': [
                            {'label': 'Low Threshold:', 'type': 'slider', 'range': (1, 100), 'step': 1, 'default': 50},
                            {'label': 'High Threshold:', 'type': 'slider', 'range': (100, 200), 'step': 1, 'default': 150}
                        ]
                    }
                }
            }

            self.current_group_boxes.append(self.createGroupBox(edge_config))

        elif selected_mode == "Frequency Domain Filter":
            Frequency_config = {
                'title': 'Frequency Domain Filter',
                'type_selector': {
                    'label': 'Frequency Filter:',
                    'options': ['High Pass Filter','Low Pass Filter'],
                    'controls': {
                        'High Pass Filter': [
                            {'label': 'CutOff Freq:', 'type': 'slider', 'range': (1, 100)}
                        ],
                        'Low Pass Filter': [
                            {'label': 'CutOff Freq:', 'type': 'slider', 'range': (1, 100),'default': 70}
                        ]
                    }
                }
            }
            self.current_group_boxes.append(self.createGroupBox(Frequency_config))

        
        elif selected_mode == "Hybrid Images":
            hyprid_config = {
                'title': 'Hybrid Images',
                'type_selector': None,  
                'controls': [
                    {'label': 'sigma', 'type': 'slider', 'range': (1, 200), 'default': 8}
                ]
            }

            hybrid_group = self.createGroupBox(hyprid_config)

            self.mix_button = QPushButton("Mix")
            self.mix_button.setStyleSheet(BUTTON_STYLE)
            self.mix_button.setFixedWidth(120)

            # Add the button to the group box layout
            mix_button_layout = QHBoxLayout()
            mix_button_layout.addWidget(self.mix_button)
            mix_button_layout.setAlignment(Qt.AlignCenter)
            hybrid_group.layout().addLayout(mix_button_layout)

            # Store the group box
            self.current_group_boxes.append(hybrid_group)


        elif selected_mode == "Threshold":
            threshold_config = {
                'title': 'Threshold',
                'type_selector': {
                    'label': 'Threshold Type:',
                    'options': ['Global', 'Local'],
                    'controls': {
                        'Global': [{'label': 'Threshold:', 'type': 'slider', 'range': (0, 255)}],
                        'Local': [{'label': 'window size', 'type': 'slider', 'range': (3, 255),'step': 2}]
                    }
                }
            }
            self.current_group_boxes.append(self.createGroupBox(threshold_config))

        elif selected_mode == "Snake":
            snake_config = {
                'title': 'Snake Parameters',
                'type_selector': None,
                'controls': [
                    {'label': 'alpha', 'type': 'doubleSpin', 'range': (0, 1), 'step': 0.01, 'default': 0.1},
                    {'label': 'beta', 'type': 'doubleSpin', 'range': (0, 1), 'step': 0.01, 'default': 0.1},
                    {'label': 'gamma', 'type': 'doubleSpin', 'range': (0, 1), 'step': 0.01, 'default': 1.0}
                ]
            }
            self.current_group_boxes.append(self.createGroupBox(snake_config))


        for group_box in self.current_group_boxes:
            self.parameter_panel.addWidget(group_box)

        self.parameter_panel.addStretch()
        # self.adjustSize()

        if self.parameters:
            self.parameter_changed.emit(self.parameters)

    def updateHistogram(self, image):

        if hasattr(self, 'histogram_group'):
            layout = self.histogram_group.layout()
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.setParent(None)

        histogram_widget = show_histograms(image)
        layout.addWidget(histogram_widget)

    def updateEqualizedHistogram(self, image):
        if hasattr(self, 'Equalized_histogram_group'):
            layout = self.Equalized_histogram_group.layout()

            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.setParent(None)

            histogram_widget = show_equalized_histograms(image)
            layout.addWidget(histogram_widget)


    def createRangeSlider(self, label_text, min_val=-255, max_val=255, default_min=-50, default_max=50):


        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        label = QLabel(label_text)
        label.setMinimumWidth(120)

        range_slider = QRangeSlider(Qt.Horizontal)
        range_slider.setRange(min_val, max_val)

        range_slider.setValue([default_min, default_max])

        def update_range():
            min_value, max_value = range_slider.value()
            if min_value >= max_value:
                min_value = max_value - 1
                range_slider.setValue([min_value, max_value])
            self.update_parameter(f"{label_text} Min", min_value)
            self.update_parameter(f"{label_text} Max", max_value)

        range_slider.sliderReleased.connect(update_range)

        layout.addWidget(label)
        layout.addWidget(range_slider)

        self.update_parameter(f"{label_text} Min", default_min)
        self.update_parameter(f"{label_text} Max", default_max)

        return container




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

from qtrangeslider import QRangeSlider

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