import sys
from PySide6.QtWidgets import QApplication
from mvps_studio.gui.app import MVPSStudio

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MVPSStudio()
    window.show()
    sys.exit(app.exec())
