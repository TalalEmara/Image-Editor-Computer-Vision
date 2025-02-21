GENERAL_STYLE = """
    QWidget {
        background-color: #1C1C28;
        color: #E0E6F0;
        font-family: 'Segoe UI', sans-serif;
        font-size: 14px;
    }
"""


GROUP_BOX_STYLE = """
    QGroupBox {{
        border: 2px solid #3A3A4E;
        border-radius: 10px;
        margin-top: 10px;
        background-color: #2A2A3C;
        padding: 10px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top center;
        padding: 0 3px;
        font-weight: bold;
        color: #A4B0C3;
        font-size: 15px;
    }}
"""


LABEL_STYLE = """
    QLabel {
        color: #E0E6F0;
        font-weight: bold;
    }
"""


COMBO_BOX_STYLE = """
    QComboBox {
        background-color: #2B2B3D;
        color: #E0E6F0;
        border: 1px solid #4E4E64;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 16px;
    }
    QComboBox:hover {
        border: 1px solid #6C9EFF;
    }
    
    QComboBox QAbstractItemView {
        background-color: #2B2B3D;
        color: #E0E6F0;
        selection-background-color: #3C4A69;
        selection-color: #A9D1FF;
        border: 1px solid #4E4E64;
    }
"""


SPIN_BOX_STYLE = """
    QSpinBox {
        background-color: #2B2B3D;
        color: #E0E6F0;
        border: 1px solid #4E4E64;
        border-radius: 4px;
        padding: 2px 6px;
    }
    QSpinBox:hover {
        border: 1px solid #6C9EFF;
    }
"""


SLIDER_STYLE = """
    QSlider::groove:horizontal {
        background: #4E4E64;
        height: 6px;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        background: #6C9EFF;
        border: 1px solid #3A527A;
        width: 14px;
        height: 14px;
        border-radius: 7px;
        margin: -4px 0;
    }
    QSlider::handle:horizontal:hover {
        background: #91BFFF;
        border: 1px solid #6C9EFF;
    }
    QSlider::sub-page:horizontal {
        background: #6C9EFF;
        border-radius: 3px;
        height: 6px;
    }
    QSlider::add-page:horizontal {
        background: #4E4E64;
        border-radius: 3px;
        height: 6px;
    }
"""


