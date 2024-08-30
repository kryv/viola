import os
from pathlib import Path
from PyQt5 import QtWidgets, QtGui, QtCore

try:
    from epics import caput
except ModuleNotFoundError as e:
    caput = lambda *kw, **arg: None

try:
    from epics import caget
except ModuleNotFoundError as e:
    caget = lambda x, timeout=1.0: None

from misc import Message

path = os.path.dirname(os.path.abspath(__file__))
pathl = Path(path)

def update_pv(**arg):
    name = arg['name']
    info = arg['info']
    base = 'image1:ArrayData'
    key = [['HCEN_CSET', 'X centroid'],
           ['VCEN_CSET', 'Y centroid'],
           ['HRMS_CSET', 'X RMS'],
           ['VRMS_CSET', 'Y RMS'],
           ['CHV_CSET', 'XY corelation'],
           ['INTEN_CSET', 'Total count']]

    if base in name:
        stat = []
        for k in key:
            s = caput(name.replace(base, k[0]), info[k[1]], timeout=1)
            stat.append(s == None)
        if all(stat):
            ret = False
        else:
            ret = True
    else:
        ret = False

    return ret

FSEE_KEY = 'FS1_FSEE'
uniformity_pv = 'FS1_SEE:DCTL_N0001:UNIFORMITY'
class FSEE_PV(QtWidgets.QWidget):
    def __init__(self, parent, model):
        super(QtWidgets.QWidget, self).__init__(parent)
        self.model = model
        layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel('- FSEE uniformity PV update')
        self.pvtxt = QtWidgets.QLineEdit(uniformity_pv, self)
        pvbtn = QtWidgets.QPushButton(self)
        pvbtn.setIcon(QtGui.QIcon(str(pathl/'icons'/'upload.svg')))
        pvbtn.setIconSize(QtCore.QSize(15, 15))
        pvbtn.setFixedSize(25, 25)
        pvbtn.clicked.connect(self.update)
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.pvtxt)
        hbox.addWidget(pvbtn)
        layout.addWidget(label)
        layout.addLayout(hbox)
        self.setLayout(layout)

    def update(self):
        ls = self.model.findItems('RMS error')
        if len(ls) != 0:
            row = ls[-1].row()
            itm = self.model.item(row, 1)
            try:
                val = float(itm.text())
                pv = self.pvtxt.text()
                s = caput(pv, val, timeout=1)
                if s is None:
                    raise OSError('Channel access to {} timed out.'.format(pv))
            except ValueError as e:
                msg = Message(self, str(e), title='Error')
            except OSError as e:
                msg = Message(self, str(e), title='Error')

CAMERA_STATE = True

def get_bkg_info(pv, bkg_dir):
    global CAMERA_STATE
    fld = bkg_dir if bkg_dir.exists() else pathl.home()/'Pictures'
    gain_name = pv.replace(':image1', ':cam1').replace(':ArrayData', ':Gain')
    expt_name = pv.replace(':image1', ':cam1').replace(':ArrayData', ':AcquireTime')
    if CAMERA_STATE:
        gain = caget(gain_name, timeout=0.5)
        expt = caget(expt_name, timeout=0.5)
    else:
        gain = 'XX'
        expt = 'XX'
    if gain is None or expt is None:
        CAMERA_STATE = False
        gain = 'XX'
        expt = 'XX'

    optname = '_bkg_g{}_t{}'.format(gain, expt)
    return fld, optname

def get_pixel_info(pv):
    bit = caget(pv.replace(':ArrayData', ':DataType_RBV'), timeout=0.5)
    xsize = caget(pv.replace(':ArrayData', ':ArraySize0_RBV'), timeout=0.5)
    ysize = caget(pv.replace(':ArrayData', ':ArraySize1_RBV'), timeout=0.5)
    return bit, xsize, ysize