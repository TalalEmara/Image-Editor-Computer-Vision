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
       
        widget.setStyleSheet(GENERAL_STYLE)
        
        for child in widget.findChildren(QGroupBox):
            child.setStyleSheet(GROUP_BOX_STYLE)
        
        for child in widget.findChildren(QLabel):
            child.setStyleSheet(LABEL_STYLE)
        
        for child in widget.findChildren(QComboBox):
            child.setStyleSheet(COMBO_BOX_STYLE)
        
        for child in widget.findChildren(QSlider):
            child.setStyleSheet(SLIDER_STYLE)
        
        for child in widget.findChildren(QSpinBox):
            child.setStyleSheet(SPIN_BOX_STYLE)
       
    
    def createSliderWithSpinBox(self, label_text, min_val, max_val):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        label = QLabel(label_text)
        label.setMinimumWidth(100)
        label.setStyleSheet(LABEL_STYLE)
        
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
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
