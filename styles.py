GENERAL_STYLE = """
    QWidget {
        background-color: #1E1E2E;
        color: #D9E0EE;
        font-family: 'Segoe UI', sans-serif;
        font-size: 14px;
    }
   
"""

GROUP_BOX_STYLE = """
    QGroupBox {{
        border: 2px solid ;
        border-radius: 10px;
        margin-top: 10px;
        background-color: #d39232;
        padding: 10px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top center;
        padding: 0 3px;
         
        font-weight: bold;
    }}
"""

LABEL_STYLE = """
    QLabel {
        color: #D9E0EE;
        font-weight: bold;
    }
"""

COMBO_BOX_STYLE = """
     QComboBox {
        background-color: #2E2E3E;
        color: #D9E0EE;
        border: 1px solid #44475A;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 16px; /* Increase font size */
    }
    QComboBox:hover {
        border: 1px solid #BD93F9;
    }
    
    QComboBox QAbstractItemView {
        background-color: #2E2E3E;
        color: #D9E0EE;
        selection-background-color: #44475A;
        selection-color: #BD93F9;
        border: 1px solid #44475A;
    }
"""

SPIN_BOX_STYLE = """
    QSpinBox {
        background-color: #2E2E3E;
        color: #D9E0EE;
        border: 1px solid #44475A;
        border-radius: 4px;
        padding: 2px 6px;
    }
    QSpinBox:hover {
        border: 1px solid #BD93F9;
    }
   
"""

SLIDER_STYLE = """
    QSlider::groove:horizontal {
        background: #44475A;
        height: 6px;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        background: #BD93F9;
        border: 1px solid #6272A4;
        width: 14px;
        height: 14px;
        border-radius: 7px;
        margin: -4px 0;
    }
    QSlider::handle:horizontal:hover {
        background: #FF79C6;
        border: 1px solid #BD93F9;
    }
    QSlider::sub-page:horizontal {
        background: #BD93F9;
        border-radius: 3px;
        height: 6px;
    }
    QSlider::add-page:horizontal {
        background: #44475A;
        border-radius: 3px;
        height: 6px;
    }
"""


