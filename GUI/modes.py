from PyQt5.QtWidgets import QVBoxLayout, QGroupBox, QPushButton, QWidget, QGridLayout
from PyQt5.QtCore import pyqtSignal, Qt
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
            "Histogram": "Histogram",
            "Equalization": "Equalization",
            "Normalize": "Normalize",
            "Hybrid Images": "Hybrid Images",
            "Snake":"Snake"
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

        # operations_layout = QVBoxLayout()
        operations_layout = QGridLayout()

        operations_layout.setSpacing(8)
        operations_layout.setContentsMargins(10, 15, 10, 10)

        # for mode_name in self.buttons:
        #     operations_layout.addWidget(self.buttons[mode_name])

        row, col = 0, 0
        counter = 0

        for mode_name in self.buttons:
            button = self.buttons[mode_name]

            if counter == len(self.buttons) - 1:  # If it's the last button, span 2 columns
                operations_layout.addWidget(button, row, 0, 1, 2)
            else:
                operations_layout.addWidget(button, row, col)
                col += 1
                if col > 1:
                    col = 0
                    row += 1

            counter += 1

        operations_group.setLayout(operations_layout)
        Mode_panel.addWidget(operations_group)

        # debug function
        # operations_group.setAttribute(Qt.WA_StyledBackground, True)
        # operations_group.setStyleSheet("background-color:#2D2D2D;")

        Mode_panel.addStretch()
        return Mode_panel

    def selectDefaultMode(self, Mode_name):
        if Mode_name in self.buttons:
            self.buttons[Mode_name].setChecked(True)
            self.current_selected_button = self.buttons[Mode_name]
            self.mode_selected.emit(Mode_name)
