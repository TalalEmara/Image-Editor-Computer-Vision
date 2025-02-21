BACKGROUND_COLOR = "#1E1E2E"
WHITE_COLOR = "#D9E0EE"
SELECTION_COLOR="#44475A"
BOX_BACKGROUND="#2E2E3E"
HOVER_COLOR="#BD93F9"
HOVER_HANDLE="#FF79C6"
PURPLE="#6272A4"

GENERAL_STYLE = f"""
    QWidget {{
        background-color: {BACKGROUND_COLOR};
        color: {WHITE_COLOR};
        font-family: 'Segoe UI', sans-serif;
        font-size: 14px;
    }}
"""

GROUP_BOX_STYLE = """
    QGroupBox {{
        border: 2px solid ;
        border-radius: 10px;
        margin-top: 10px;
        background-color: {BACKGROUND_COLOR};
        padding: 10px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top center;
        padding: 0 3px;
         
        font-weight: bold;
    }}
"""

LABEL_STYLE = f"""
    QLabel {{
        color: {WHITE_COLOR};
        font-weight: bold;
    }}
"""

COMBO_BOX_STYLE =f"""
     QComboBox {{
        background-color: {BOX_BACKGROUND};
        color: {WHITE_COLOR};
        border: 1px solid {SELECTION_COLOR};
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 16px; 
    }}
    QComboBox:hover {{
        border: 1px solid {HOVER_COLOR};
    }}
    
    QComboBox QAbstractItemView {{
        background-color: {BOX_BACKGROUND};
        color: {WHITE_COLOR};
        selection-background-color: {SELECTION_COLOR};
        selection-color: {HOVER_COLOR};
        border: 1px solid {SELECTION_COLOR};
    }}
"""

SPIN_BOX_STYLE = f"""
    QSpinBox {{
        background-color: {BOX_BACKGROUND};
        color: {WHITE_COLOR};
        border: 1px solid {SELECTION_COLOR};
        border-radius: 4px;
        padding: 2px 6px;
    }}
    QSpinBox:hover {{
        border: 1px solid {HOVER_COLOR};
    }}
   
"""

SLIDER_STYLE = f"""
    QSlider::groove:horizontal {{
        background: {SELECTION_COLOR};
        height: 6px;
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {HOVER_COLOR};
        border: 1px solid {PURPLE};
        width: 14px;
        height: 14px;
        border-radius: 7px;
        margin: -4px 0;
    }}
    QSlider::handle:horizontal:hover {{
        background: {HOVER_HANDLE};
        border: 1px solid {HOVER_COLOR};
    }}
    QSlider::sub-page:horizontal {{
        background: {HOVER_COLOR};
        border-radius: 3px;
        height: 6px;
    }}
    QSlider::add-page:horizontal {{
        background: {SELECTION_COLOR};
        border-radius: 3px;
        height: 6px;
    }}
"""


