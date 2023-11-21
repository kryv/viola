import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk

from numba import jit, njit
from matplotlib.figure import Figure
from matplotlib.backends.qt_compat import QtCore, QtWidgets, QtGui
if QtCore.qVersion() >= "5.":
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

menustyle = """
QMenu::item
{
    background-color: #e6e6e6;
    color: black;
}
QMenu::item::selected
{
    background-color: gray;
    color: white;
}
"""

def lcnvf(i, f, w):
    """
    i = initial pixel
    f = final pixel
    w = width in mm
    """
    m = (i+f)/2.0
    return lambda p: (p-m)*w/(f-i)

def gauss(xmean, ymean, xxcov, yycov, xycov, xx, yy, xy, xx2, yy2):
    ecov = xxcov*yycov-xycov*xycov
    src = (xx2 - 2.0*xx*xmean + xmean**2)*(yycov/ecov) \
        + (yy2 - 2.0*yy*ymean + ymean**2)*(xxcov/ecov) \
        - 2.0*(xy - xx*ymean - yy*xmean + xmean*ymean)*(xycov/ecov)
    return src

def finite(src, peak, bkgr, nois, satr):
    ys, xs = src.shape
    err = nois*np.random.randn(ys, xs) + bkgr
    err = err.astype(np.int32)
    data = np.array(np.exp(-src/2.0)*peak)
    data = data.astype(np.int32)
    data += err
    data[data<0] = 0
    data[data>int(satr)] = int(satr)
    return data.astype(np.int16)

class DistGen(QtCore.QObject):
    x = 10.0
    y = -10.0
    a = 30.0
    b = 10.0
    p = 25.0
    dt = 0.5
    data = None
    signal = QtCore.pyqtSignal()

    peak = 20000.0
    bkgr = 3000.0
    nois = 2000.0
    satr = 30000.0

    def update(self):
        a = self.a
        b = self.b
        p = self.p

        sn = np.sin(p*np.pi/180.0)
        cs = np.cos(p*np.pi/180.0)

        xx = a*a*cs*cs + b*b*sn*sn
        yy = b*b*cs*cs + a*a*sn*sn
        xy = (a*a-b*b)*sn*cs

        self.xx = xx
        self.yy = yy
        self.xy = xy
        self.xr = np.sqrt(xx)
        self.yr = np.sqrt(yy)

    @QtCore.pyqtSlot()
    def image_gen(self):
        ys, xs = (1200, 1920)
        xcnv = lcnvf(660.0, 1260.0, 200.0)
        ycnv = lcnvf(300.0, 900.0, -200.0)
        xi = xcnv(np.arange(xs))
        yi = ycnv(np.arange(ys))
        xi = xi.astype(np.float32)
        yi = yi.astype(np.float32)
        xx, yy = np.meshgrid(xi, yi)
        xy = xx*yy
        xx2, yy2 = np.meshgrid(xi*xi, yi*yi)

        while True:
            t0 = time.time()
            xmean = self.x
            ymean = self.y
            xxcov = self.xx
            yycov = self.yy
            xycov = self.xy

            peak = self.peak
            bkgr = self.bkgr
            nois = self.nois
            satr = self.satr

            src = gauss(xmean, ymean, xxcov, yycov, xycov, xx, yy, xy, xx2, yy2)
            self.data = finite(src, peak, bkgr, nois, satr)

            imgout = self.data.flatten()
            with open('dummy_image.pkl', 'wb') as f:
                pk.dump(imgout, f)

            t1 = time.time() - t0

            if self.dt > t1:
                print('remain time: {}'.format(self.dt - t1))
                time.sleep(self.dt - t1)
            else:
                print('over time: {}'.format(t1 - self.dt))

            time.sleep(self.dt)
            self.signal.emit()

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.image = True
        self.cnt = 0

        self._main = QtWidgets.QWidget()
        self.setStyleSheet("""
            background-color: white;
            alternate-background-color: #e6e6e6;
            font-size: 17px;
            """)

        self.setCentralWidget(self._main)
        self.setWindowTitle('Dummy Plug')

        layout = QtWidgets.QHBoxLayout(self._main)
        if self.image:
            canvas = FigureCanvas(Figure(tight_layout=True))
            self.ax = canvas.figure.add_subplot(111)
            layout.addWidget(canvas)

        settab = QtWidgets.QVBoxLayout()
        self.dist = DistGen()
        self.dist.update()

        tabtitle = QtWidgets.QLabel('Ellipse Parameters', self)
        xlay, self.ell_x = self.textline(' X    :', '[mm]')
        ylay, self.ell_y = self.textline(' Y    :', '[mm]')
        alay, self.ell_a = self.textline(' A    :', '[mm]')
        blay, self.ell_b = self.textline(' B    :', '[mm]')
        play, self.ell_p = self.textline(' Angle:', '[deg]')
        split = QtWidgets.QLabel('', self)
        xrlay, self.ell_xr = self.textline(' Xrms :', '[mm]')
        yrlay, self.ell_yr = self.textline(' Yrms :', '[mm]')
        xylay, self.ell_xy = self.textline(' X-Y  :', '[1]')
        self.ell_xr.setEnabled(False)
        self.ell_yr.setEnabled(False)
        self.ell_xy.setEnabled(False)

        pklay, self.peak = self.textline(' Peak Amp.  :', '', width=[150, 100, 0])
        bklay, self.bkgr = self.textline(' Background :', '', width=[150, 100, 0])
        nslay, self.nois = self.textline(' Nois Level :', '', width=[150, 100, 0])
        stlay, self.satr = self.textline(' Saturation :', '', width=[150, 100, 0])

        self.ell_x.textChanged.connect(self.update_dist)
        self.ell_y.textChanged.connect(self.update_dist)
        self.ell_a.textChanged.connect(self.update_dist)
        self.ell_b.textChanged.connect(self.update_dist)
        self.ell_p.textChanged.connect(self.update_dist)
        self.peak.textChanged.connect(self.update_dist)
        self.bkgr.textChanged.connect(self.update_dist)
        self.nois.textChanged.connect(self.update_dist)
        self.satr.textChanged.connect(self.update_dist)

        self.sldx = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.sldy = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slda = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.sldb = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.sldp = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.sldpk = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.sldbk = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.sldns = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.sldst = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)

        self.sldx.setRange(-200, 200)
        self.sldy.setRange(-150, 150)
        self.slda.setRange(1, 100)
        self.sldb.setRange(1, 100)
        self.sldp.setRange(-180, 180)
        self.sldpk.setRange(1, 50000)
        self.sldbk.setRange(0, 10000)
        self.sldns.setRange(0, 10000)
        self.sldst.setRange(1, 30000)

        self.sldx.valueChanged.connect(self.update_sld)
        self.sldy.valueChanged.connect(self.update_sld)
        self.slda.valueChanged.connect(self.update_sld)
        self.sldb.valueChanged.connect(self.update_sld)
        self.sldp.valueChanged.connect(self.update_sld)
        self.sldpk.valueChanged.connect(self.update_sld)
        self.sldbk.valueChanged.connect(self.update_sld)
        self.sldns.valueChanged.connect(self.update_sld)
        self.sldst.valueChanged.connect(self.update_sld)

        self.load_dist()
        self.load_sld()

        settab.addWidget(tabtitle)
        settab.addLayout(xlay)
        settab.addWidget(self.sldx)
        settab.addLayout(ylay)
        settab.addWidget(self.sldy)
        settab.addLayout(alay)
        settab.addWidget(self.slda)
        settab.addLayout(blay)
        settab.addWidget(self.sldb)
        settab.addLayout(play)
        settab.addWidget(self.sldp)
        settab.addWidget(split)
        settab.addLayout(xrlay)
        settab.addLayout(yrlay)
        settab.addLayout(xylay)
        settab.addWidget(split)
        settab.addLayout(pklay)
        settab.addWidget(self.sldpk)
        settab.addLayout(bklay)
        settab.addWidget(self.sldbk)
        settab.addLayout(nslay)
        settab.addWidget(self.sldns)
        settab.addLayout(stlay)
        settab.addWidget(self.sldst)
        settab.setAlignment(QtCore.Qt.AlignTop)
        layout.addLayout(settab)

        self.thread = QtCore.QThread(self)
        self.dist.moveToThread(self.thread)

        if self.image:
            self.dist.signal.connect(self.callback)

        self.thread.started.connect(self.dist.image_gen)
        self.thread.start()

    @QtCore.pyqtSlot()
    def callback(self):
        if self.cnt == 0:
            self.d = self.ax.imshow(self.dist.data)
            self.cnt = 1
        else:
            self.d.set_data(self.dist.data)

        self.ax.figure.canvas.draw()

    def load_dist(self):
        self.ell_x.setText('{}'.format(self.dist.x))
        self.ell_y.setText('{}'.format(self.dist.y))
        self.ell_a.setText('{}'.format(self.dist.a))
        self.ell_b.setText('{}'.format(self.dist.b))
        self.ell_p.setText('{}'.format(self.dist.p))
        self.peak.setText('{}'.format(self.dist.peak))
        self.bkgr.setText('{}'.format(self.dist.bkgr))
        self.nois.setText('{}'.format(self.dist.nois))
        self.satr.setText('{}'.format(self.dist.satr))

    def load_sld(self):
        self.sldx.setValue(self.dist.x)
        self.sldy.setValue(self.dist.y)
        self.slda.setValue(self.dist.a)
        self.sldb.setValue(self.dist.b)
        self.sldp.setValue(self.dist.p)
        self.sldpk.setValue(self.dist.peak)
        self.sldbk.setValue(self.dist.bkgr)
        self.sldns.setValue(self.dist.nois)
        self.sldst.setValue(self.dist.satr)

    def update_dist(self):
        try:
            self.dist.x = float(self.ell_x.text())
            self.dist.y = float(self.ell_y.text())
            self.dist.a = float(self.ell_a.text())
            self.dist.b = float(self.ell_b.text())
            self.dist.p = float(self.ell_p.text())

            self.dist.peak = float(self.peak.text())
            self.dist.bkgr = float(self.bkgr.text())
            self.dist.nois = float(self.nois.text())
            self.dist.satr = float(self.satr.text())

            self.load_sld()
            self.dist.update()
            self.ell_xr.setText('{:.4f}'.format(self.dist.xr))
            self.ell_yr.setText('{:.4f}'.format(self.dist.yr))
            self.ell_xy.setText('{:.4f}'.format(self.dist.xy/self.dist.xr/self.dist.yr))
        except:
            pass

    def update_sld(self):
        self.dist.x = self.sldx.value()
        self.dist.y = self.sldy.value()
        self.dist.a = self.slda.value()
        self.dist.b = self.sldb.value()
        self.dist.p = self.sldp.value()
        self.dist.peak = self.sldpk.value()
        self.dist.bkgr = self.sldbk.value()
        self.dist.nois = self.sldns.value()
        self.dist.satr = self.sldst.value()
        self.load_dist()

    def textline(self, label, unit='', width=[100, 100, 100]):
        layout = QtWidgets.QHBoxLayout()
        txt1 = QtWidgets.QLabel(label, self)
        txt1.setFixedWidth(width[0])
        txt2 = QtWidgets.QLineEdit(self)
        txt2.setFixedWidth(width[1])
        txt3 = QtWidgets.QLabel(unit, self)
        txt3.setFixedWidth(width[2])
        layout.addWidget(txt1)
        layout.addWidget(txt2)
        layout.addWidget(txt3)
        return layout, txt2

if __name__ == "__main__":
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    app = ApplicationWindow()
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec_()
