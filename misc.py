import platform
from PyQt5 import QtWidgets, QtGui, QtCore

if platform.system() == 'Windows':
    font_size = 14
else:
    font_size = 17

windowstyle = f"""
* {{
    background-color: white;
    alternate-background-color: #f3f3f3;
    selection-color: black;
    selection-background-color: #e0e0e0;
    font-size: {font_size}px;
}}
QLineEdit {{
    border: 1px solid gray;
    border-radius: 3px;
}}
QPushButton {{
    background-color: white;
    border: 1px solid gray;
    border-radius: 4px;
    border-style: outset;
}}
QPushButton:pressed {{
    background-color: #f0f0f0;
    border-style: inset;
}}
QHeaderView:section {{
    background-color: white;
}}
QComboBox {{
    background-color: #f0f0f0;
}}

"""

menustyle = """
* {
    background-color: #f3f3f3;
}
QMenu::item {
    background-color: #f3f3f3;
    color: black;
}
QMenu::item::selected {
    background-color: gray;
    color: white;
}
"""

class Message(QtWidgets.QWidget):
    def __init__(self, parent, txt, title):
        super(QtWidgets.QWidget, self).__init__(parent)
        msg = QtWidgets.QMessageBox()
        msg.setStyleSheet(windowstyle+"""
            QPushButton {
                padding-top: 5px;
                padding-bottom: 5px;
                padding-right: 8px;
                padding-left: 8px;
            }""")
        if title == 'Error':
            msg.setIcon(QtWidgets.QMessageBox.Critical)
        else:
            msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setWindowTitle(title)
        msg.setText(txt)
        msg.exec()