BACKGROUND_COLOR = "#F5F5F0"   
WHITE_COLOR = "#1A1A1A"        
SELECTION_COLOR = "#E0E0D8"    
BOX_BACKGROUND = "#FFFFFF"     
HOVER_COLOR = "#C8C8C0"        
HOVER_HANDLE = "#A0A098"        
ACCENT_COLOR = "#546E7A"        
ACCENT_HOVER = "#455A64"       

GENERAL_STYLE = f"""
    QWidget {{
        background-color: {BACKGROUND_COLOR};
        color: {WHITE_COLOR};
        font-family: 'Segoe UI', sans-serif;
        font-size: 14px;
        font-weight: bold;
    }}
"""

GROUP_BOX_STYLE = f"""
    QGroupBox {{
        border: 2px solid {ACCENT_COLOR};
        border-radius: 10px;
        margin-top: 10px;
        background-color: {BACKGROUND_COLOR};
        padding: 10px;
        font-weight: bold;
        font-size: 14px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top center;
        padding: 0 3px;
        color: {ACCENT_COLOR};
        font-weight: bold;
        font-size: 14px;
    }}
"""

LABEL_STYLE = f"""
    QLabel {{
        color: {ACCENT_COLOR};
        font-weight: bold;
        font-size:16px;
    }}
"""

COMBO_BOX_STYLE = f"""
     QComboBox {{
        background-color: {BOX_BACKGROUND};
        color: {WHITE_COLOR};
        border: 2px solid {ACCENT_COLOR};
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 14px; 
    }}
    QComboBox:hover {{
        border: 2px solid {ACCENT_HOVER};
    }}
    
    QComboBox QAbstractItemView {{
        background-color: {BOX_BACKGROUND};
        color: {WHITE_COLOR};
        selection-background-color: {ACCENT_COLOR};
        selection-color: {BOX_BACKGROUND};
        border: 2px solid {ACCENT_COLOR};
    }}
"""

SPIN_BOX_STYLE = f"""
    QSpinBox ,QDoubleSpinBox{{
        background-color: {BOX_BACKGROUND};
        color: {WHITE_COLOR};
        border: 2px solid {ACCENT_COLOR};
        border-radius: 4px;
        padding: 2px 6px;
    }}
    QSpinBox:hover, QDoubleSpinBox:hover {{
        border: 2px solid {ACCENT_HOVER};
    }}
"""

SLIDER_STYLE = f"""
    QSlider::groove:horizontal, QRangeSlider::groove:horizontal{{
        background: {SELECTION_COLOR};
        height: 6px;
        border-radius: 3px;
    }}
    QSlider::handle:horizontal ,QRangeSlider::handle:horizontal{{
        background: {ACCENT_COLOR};
        border: 1px solid {ACCENT_HOVER};
        width: 14px;
        height: 14px;
        border-radius: 7px;
        margin: -4px 0;
    }}
    QSlider::handle:horizontal:hover,QRangeSlider::handle:horizontal:hover {{
        background: {ACCENT_HOVER};
        border: 2px solid {ACCENT_COLOR};
    }}
    QSlider::sub-page:horizontal,QRangeSlider::sub-page:horizontal {{
        background: {ACCENT_COLOR};
        border-radius: 3px;
        height: 6px;
    }}
    QSlider::add-page:horizontal,QRangeSlider::add-page:horizontal {{
        background: {SELECTION_COLOR};
        border-radius: 3px;
        height: 6px;
    }}
"""

BUTTON_STYLE = f"""
    QPushButton {{
        background-color: {BOX_BACKGROUND};
        color: {ACCENT_COLOR};
        border: 2px solid {ACCENT_COLOR};
        border-radius: 4px;
        padding: 5px 10px;
        font-weight: bold;
    }}
    QPushButton:hover {{
        background-color: {ACCENT_COLOR};
        border: 2px solid {ACCENT_HOVER};
        color: {BOX_BACKGROUND};
    }}
    QPushButton:pressed {{
        background-color: {ACCENT_HOVER};
        color: {BOX_BACKGROUND};
        border: 2px solid {ACCENT_COLOR};
    }}
"""