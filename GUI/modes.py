from PyQt5.QtWidgets import QVBoxLayout, QGroupBox, QPushButton, QWidget
from PyQt5.QtCore import pyqtSignal
from GUI.styles import BUTTON_STYLE, GROUP_BOX_STYLE, GENERAL_STYLE

class ModeSelector(QWidget):
    mode_selected = pyqtSignal(str)  

    def __init__(self):
        super().__init__()
        self.current_selected_button = None
        self.buttons = {}
      
        button_names = {
            "Gray": "Gray",
            "Noise and Filtering": "Noise & Filter",
            "Frequency Domain Filter": "Frequency Domain Filter",
            "Edge Detection": "Edge Detection",
            "Threshold": "Threshold",
            "Hybrid Images": "Hybrid Images",
            "Histogram": "Histogram",
            "Equalization": "Equalization",
            "Normalize": "Normalize"
        }
        
        for display_name, signal_name in button_names.items():
            btn = QPushButton(display_name)
            btn.setCheckable(True)
            btn.setStyleSheet(BUTTON_STYLE)
            self.buttons[signal_name] = btn
            btn.clicked.connect(lambda checked, name=signal_name, button=btn: 
                                self.handle_button_selection(name, button))

    def handle_button_selection(self, mode_name, clicked_button):
        if self.current_selected_button and self.current_selected_button != clicked_button:
            self.current_selected_button.setChecked(False)
        if clicked_button.isChecked():
            self.current_selected_button = clicked_button
            self.mode_selected.emit(mode_name)
        else:
            clicked_button.setChecked(True)

    def createmodePanel(self):
        self.setStyleSheet(GENERAL_STYLE)
        Mode_panel = QVBoxLayout()

        operations_group = QGroupBox("Modes")
        operations_group.setStyleSheet(GROUP_BOX_STYLE)
        operations_layout = QVBoxLayout()
        operations_layout.setSpacing(8)
        operations_layout.setContentsMargins(10, 15, 10, 10)

        for mode_name in self.buttons:
            operations_layout.addWidget(self.buttons[mode_name])

        operations_group.setLayout(operations_layout)
        Mode_panel.addWidget(operations_group)

        Mode_panel.addStretch()
        return Mode_panel

    def selectDefaultMode(self, Mode_name):
        if Mode_name in self.buttons:
            self.buttons[Mode_name].setChecked(True)
            self.current_selected_button = self.buttons[Mode_name]
            self.mode_selected.emit(Mode_name)
