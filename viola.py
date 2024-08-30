import os
import sys
import time
import cv2
import json
import argparse
import tempfile
import logging
import urllib.request
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle as pk

from scipy import ndimage
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from datetime import datetime
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib.offsetbox import (AnnotationBbox, DrawingArea)
#from matplotlib.backends.qt_compat import (QtCore, QtWidgets, QtGui)
from PyQt5 import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

__version__ = '1.5.0'

mpl.use('QT5Agg')
try: QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
except: pass

try: QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
except: pass

try: QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_DontUseNativeDialogs, True)
except: pass

try: QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_DontUseNativeMenuBar, True)
except: pass

logging.basicConfig(level=logging.INFO)

try:
    from epics import caget
except ModuleNotFoundError as e:
    logging.info('use dummy "caget" function')
    caget = lambda x: -15000*np.eye(400, dtype=np.int16).flatten()

use_numba = False
if use_numba:
    try:
        from func_numba import (get_cov, gen_mask, bkgr_error)
    except ModuleNotFoundError as e:
        logging.info(e)
        logging.info('use numpy base functions')
        from func import (get_cov, gen_mask, bkgr_error)
else:
    from func import (get_cov, gen_mask, bkgr_error)

from func import (lcnvf, coodinate_image, get_stat, c_cent, c_mrgr, c_pts2, get_ellipse)
from func import (NoIndent, NoIndentEncoder)
from misc import Message, windowstyle, menustyle

path = os.path.dirname(os.path.abspath(__file__))
pathl = Path(path)

frib_ftc = True
if frib_ftc:
    from frib_device import update_pv, FSEE_KEY, FSEE_PV
    from frib_device import get_bkg_info
    from frib_device import get_pixel_info
    extfunc = update_pv
    extkey1 = FSEE_KEY
    extbtn1 = FSEE_PV
    extbkgf = get_bkg_info
    extpixl = get_pixel_info
else:
    extfunc = None
    extkey1 = ''
    extbtn1 = None
    extbkgf = None
    extpixl = None

if hasattr(Image, "UnidentifiedImageError"):
    UIError = Image.UnidentifiedImageError
else:
    UIError = OSError

cmaps = plt.colormaps()
linecolors = ['w', 'w', 'lime', 'orange', 'red']

sunit = {'X centroid': 'mm',
            'Y centroid': 'mm',
            'X RMS': 'mm',
            'Y RMS': 'mm',
            'XY corelation': ''}

Qivalp = QtGui.QIntValidator()
Qivalp.setBottom(0)

Qdvalp = QtGui.QDoubleValidator()
Qdvalp.setBottom(0.0)

def ellipse(cen, v, ang, facecolor='none', **kws):
    ell = mpl.patches.Ellipse(cen, v[0], v[1], angle=180+ang, facecolor=facecolor, **kws)
    return ell

def plot_trans(ax, tlbr, pts1, mrgn, sz, vpts1, lclr):
    f0 = 0.20
    f1 = 1-f0
    f2 = 4
    trgp = np.array([[[0, -1], [ 1, 0]],
                     [[0, -1], [-1, 0]],
                     [[0,  1], [ 1, 0]],
                     [[0,  1], [-1, 0]]])*sz

    arrow = mpl.patches.ArrowStyle.CurveFilledAB(head_width=0.3)

    rpp = c_mrgr(pts1, mrgn)[[0, 1, 3, 2]]
    rpg = mpl.patches.Polygon(rpp, lw=1.5, fill=False, ec=lclr[3], ls='--')
    ax.add_patch(rpg)

    pga = []
    for p, t in zip(pts1, trgp):
        da = DrawingArea(20, 20, 10, 10)
        dots = [[0.0, 0.0], t[0], f0*t[1]+f1*t[0], f0*(t[0]+t[1]), f0*t[0]+f1*t[1], t[1]]
        pg = mpl.patches.Polygon(dots, lw=1.5, fill=False, fc=lclr[4], ec=lclr[4])
        pc = mpl.patches.Circle(0.6*(t[0]+t[1]), 3, lw=1.5, fill=True, fc=lclr[4], ec=lclr[4])
        pc.set_visible(False)
        da.add_artist(pg)
        da.add_artist(pc)
        ab = AnnotationBbox(da, p, frameon=False, pad=0.0)
        ax.add_artist(ab)
        pga.append([ab, pg, pc])

    clp = c_cent(pts1)
    cly = mpl.patches.FancyArrowPatch(clp[0], clp[1],
            arrowstyle=arrow, mutation_scale=25, fc=lclr[4], ec=lclr[4], lw=2)
    clx = mpl.patches.FancyArrowPatch(clp[2], clp[3],
            arrowstyle=arrow, mutation_scale=25, fc=lclr[4], ec=lclr[4], lw=2)
    ax.add_patch(cly)
    ax.add_patch(clx)
    ax.grid(True)
    return pga, rpg, [cly, clx]

class ImageAnalysis():
    def __init__(self, xsize, ysize):
        self.have_main = False
        self.have_hist = False
        self.have_stat = False

        self.xwidth = 200
        self.yheight = 200
        self.init_pts1(xsize, ysize)
        self.mrgn = np.float32([0, 0, 0, 0])
        self.xangle = 0.0
        self.yangle = 0.0
        self.xofs = 0.0
        self.yofs = 0.0

        self.fltr = False
        self.fltr_mtd = 'Gaussian'
        self.fltr_size = 1

        self.hcmap = 'viridis'
        self.lclr = [mpl.colors.to_hex(c) for c in linecolors]
        self.lvis = True

    def init_pts1(self, xsize, ysize):
        self.pts1 = np.float32([[xsize*0.0, ysize*0.0],
                                [xsize*1.0, ysize*0.0],
                                [xsize*0.0, ysize*1.0],
                                [xsize*1.0, ysize*1.0]])

    def update_coord(self):
        self.mm, self.dsize = c_pts2(self.pts1, self.mrgn, self.yheight/self.xwidth)
        xs, ys = self.dsize

        xinfo = [self.mrgn[2], xs-self.mrgn[3], self.xwidth*np.cos(self.xangle*np.pi/180)]
        yinfo = [self.mrgn[0], ys-self.mrgn[1], -self.yheight*np.cos(self.yangle*np.pi/180)]

        x0i, self.x1i, self.x2i = coodinate_image(xs, ys, xinfo, yinfo)
        self.xi = x0i[0] + self.xofs
        self.ximax = self.xi.max()
        self.ximin = self.xi.min()
        self.xw = self.ximax - self.ximin
        self.xc = self.xi.min() + self.xw/2
        self.yi = x0i[1] + self.yofs
        self.yimax = self.yi.max()
        self.yimin = self.yi.min()
        self.yw = self.yimax - self.yimin
        self.yc = self.yi.min() + self.yw/2

    def load_img(self, im0):
        self.im = cv2.warpPerspective(im0, self.mm, self.dsize)
        if self.fltr:
            if self.fltr_mtd == 'Mean':
                fsize = np.abs(int(self.fltr_size))
                self.im = cv2.blur(self.im, (fsize, fsize))
            elif self.fltr_mtd == 'Gaussian':
                fsize = np.abs(2*int(int(self.fltr_size)/2)) + 1
                self.im = cv2.GaussianBlur(self.im, (fsize, fsize), 0)

        tmp = self.im.flatten()[::3]
        self.ist = np.sort(tmp)[::-1]
        self.pxs = 3*np.arange(len(self.ist))+1

    def pre_analysis(self, thl, bkgr):
        self.thl = thl
        self.bkgr = bkgr
        imp = self.im.copy()
        imp[imp<thl] = bkgr
        self.imp = imp - bkgr

    def analysis(self, s1=3, s2=6, itr=5):

        self.s2 = s2
        ys, xs = self.imp.shape
        cen, cov, tot = get_cov(self.imp, self.x1i, self.x2i)

        for i in range(itr):
            msk = np.zeros([ys, xs], dtype=bool)
            msk = gen_mask(msk, cen, cov, s1, self.xi, self.yi, self.x1i, self.x2i)
            cen, cov, tot = get_cov(self.imp, self.x1i, self.x2i, msk)

        msk = np.zeros([ys, xs], dtype=bool)
        msk = gen_mask(msk, cen, cov, s2, self.xi, self.yi, self.x1i, self.x2i)
        self.cen, self.cov, self.total = get_cov(self.imp, self.x1i, self.x2i, msk)
        self.cen += np.array([self.xofs, self.yofs])
        self.xprj = np.sum(self.imp*msk, axis=0)
        self.yprj = np.sum(self.imp*msk, axis=1)
        self.bkgrlvl, self.bkgrrms = bkgr_error(self.im, msk)

    def plot_main(self, ax, status, rescale=False, nsig=3):
        if not self.have_main:
            self.ims = ax.imshow(self.im, cmap = self.hcmap,
                extent=(self.xi.min(), self.xi.max(), self.yi.min(), self.yi.max()))
            self.im0i = self.im.min()
            self.im1i = self.im.max()

            ev = get_ellipse(self.cen, self.cov)
            self.el1 = ax.add_patch(ellipse(ev[0], ev[1]*nsig, ev[2],
                            edgecolor=self.lclr[1], lw=1.5))
            self.el2 = ax.add_patch(ellipse(ev[0], ev[1]*self.s2, ev[2],
                            edgecolor=self.lclr[2], lw=1.5))
            self.cbr = plt.colorbar(self.ims, ax = ax, extend='both')
            self.have_main = True
        else:
            if status == 2:
                self.ims.set_data(self.im)
                self.ims.set_extent((self.xi.min(), self.xi.max(), self.yi.min(), self.yi.max()))
                if rescale:
                    self.im0i = self.im.min()
                    self.im1i = self.im.max()
                self.ims.set_clim(self.im0i, self.im1i)
                self.ims.set_cmap(self.hcmap)
                if mpl.__version__ < '3.5':
                    self.cbr.draw_all()
                else:
                    ax.figure.draw_without_rendering()
            ev = get_ellipse(self.cen, self.cov)
            self.el1.set_center(ev[0])
            self.el1.set_edgecolor(self.lclr[1])
            self.el1.width = ev[1][0]*nsig
            self.el1.height = ev[1][1]*nsig
            self.el1.angle = ev[2]
            self.el2.set_center(ev[0])
            self.el2.set_edgecolor(self.lclr[2])
            self.el2.width = ev[1][0]*self.s2
            self.el2.height = ev[1][1]*self.s2
            self.el2.angle = ev[2]

    def plot_hist(self, ax, xscale, yscale, centering):

        if centering and ~np.isnan(self.cen).any():
            xi0 = self.cen[0] - self.xw*xscale/2
            xi1 = self.cen[0] + self.xw*xscale/2
            yi0 = self.cen[1] - self.yw*yscale/2
            yi1 = self.cen[1] + self.yw*yscale/2
        else:
            xi0 = self.xc - self.xw*xscale/2
            xi1 = self.xc + self.xw*xscale/2
            yi0 = self.yc - self.yw*yscale/2
            yi1 = self.yc + self.yw*yscale/2

        xi0 = self.ximin if xi0 < self.ximin else xi0
        xi1 = self.ximax if xi1 > self.ximax else xi1
        yi0 = self.yimin if yi0 < self.yimin else yi0
        yi1 = self.yimax if yi1 > self.yimax else yi1
        xoff = (xi1-xi0)/400
        yoff = (yi1-yi0)/400

        vxmax = self.xprj.max()/((yi1-yi0)/6.0)
        vymax = self.yprj.max()/((xi1-xi0)/6.0)
        vxmax = 1.0 if vxmax == 0.0 else vxmax
        vymax = 1.0 if vymax == 0.0 else vymax

        ax.set_xlim(xi0, xi1)
        ax.set_ylim(yi0, yi1)

        if not self.have_hist:
            self.xhst = ax.plot(self.xi, self.xprj/vxmax + yi0 + yoff, self.lclr[3])[0]
            self.yhst = ax.plot(self.yprj/vymax + xi0 + xoff, self.yi, self.lclr[3])[0]
            self.xcen = ax.axvline(self.cen[0], color=self.lclr[0])
            self.ycen = ax.axhline(self.cen[1], color=self.lclr[0])
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('y [mm]')
            self.have_hist = True
        else:
            self.xhst.set_xdata(self.xi)
            self.xhst.set_ydata(self.xprj/vxmax + yi0 + yoff)
            self.xhst.set_color(self.lclr[3])
            self.yhst.set_ydata(self.yi)
            self.yhst.set_xdata(self.yprj/vymax + xi0 + xoff)
            self.yhst.set_color(self.lclr[3])
            self.xcen.set_xdata([self.cen[0]]*2)
            self.ycen.set_ydata([self.cen[1]]*2)
            self.xcen.set_color(self.lclr[0])
            self.ycen.set_color(self.lclr[0])

    def plot_stat(self, ax):
        ymax = np.max([self.ist.max(), self.thl])
        ax.set_ylim(-ymax*0.05, ymax*1.05)
        ax.set_xlim(1, self.pxs[-1])

        if not self.have_stat:
            self.psrt = ax.plot(self.pxs, self.ist, '-', color='blue')[0]
            self.athl = ax.annotate('Threshold', xy=(2, self.thl+0.02*ymax),
                            color='r', fontsize='large', fontweight='bold')
            self.abkgr = ax.annotate('Background', xy=(2, self.bkgr-0.07*ymax),
                            color='k', fontsize='large', fontweight='bold')
            self.pthl = ax.axhline(self.thl, color='r', lw=1)
            self.pbkgr = ax.axhline(self.bkgr, color='k', lw=1)
            ax.set_xlabel('Pixel Number')
            ax.set_ylabel('Count')
            self.have_stat = True
        else:
            self.psrt.set_xdata(self.pxs)
            self.psrt.set_ydata(self.ist)
            self.athl.set_position((2, self.thl+0.02*ymax))
            self.abkgr.set_position((2, self.bkgr-0.07*ymax))
            self.pthl.set_ydata([self.thl]*2)
            self.pbkgr.set_ydata([self.bkgr]*2)

    def mouse_on_stat(self, event):
        if self.have_stat:
            if event.inaxes:
                ap = np.abs(self.pthl.get_ydata()[0] - event.ydata)
                bp = np.abs(self.pbkgr.get_ydata()[0] - event.ydata)
                if ap < bp:
                    self.pthl.set_lw(2)
                    self.pbkgr.set_lw(1)
                else:
                    self.pthl.set_lw(1)
                    self.pbkgr.set_lw(2)

                if event.button == 1:
                    if ap < bp:
                        self.pthl.set_ydata([event.ydata]*2)
                    else:
                        self.pbkgr.set_ydata([event.ydata]*2)
                    return self.pthl.get_ydata()[0], self.pbkgr.get_ydata()[0]

    def set_line_visible(self):
        state = self.lvis
        self.el1.set_visible(state)
        self.el2.set_visible(state)
        self.xhst.set_visible(state)
        self.yhst.set_visible(state)
        self.xcen.set_visible(state)
        self.ycen.set_visible(state)

class ImageStorage(QtCore.QObject):
    signal = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()
    run = False
    store = []
    maxsize = 100
    dt = 1.0
    xsize = 1920
    ysize = 1200
    norm = False
    nfac = 30000.0
    ovrf = 'absolute'
    cmave = False
    cmsav = np.array([])
    cmprv = np.array([])
    cmcnt = 0
    #imas = ImageAnalysis(xsize, ysize)
    target = ''
    def load_source(self):
        try:
            root, ext = os.path.splitext(self.target)
            if root == 'dummy_image':
                with open('dummy_image.pkl', 'rb') as f:
                    imgout = pk.load(f).astype(np.float32)
                    self.im0 = imgout.reshape(self.ysize, self.xsize)
            elif root[0:8] == 'https://' or root[0:7] == 'http://':
                strm = urllib.request.urlopen(self.target)
                byt = bytes()
                for i in range(5000):
                    byt += strm.read(1024)
                    b = byt.find(b'\xff\xd9')
                    if b != -1:
                        break
                a = byt.find(b'\xff\xd8')
                raw = cv2.imdecode(np.frombuffer(byt[a:b+2], dtype=np.uint8), -1)
                self.im0 = np.array(Image.fromarray(raw).convert('F'), dtype=np.float32)
                self.ysize, self.xsize = self.im0.shape

            elif ext == '':
                imgout = caget(root).astype(np.uint16).astype(np.float32)
                if len(imgout) == 0:
                    raise ValueError('Length of "{}" is zero. Camera is supposed to be OFF.'.format(root))
                if (extpixl is not None) and (len(imgout) != self.xsize*self.ysize):
                    bit, self.xsize, self.ysize = extpixl(root)
                self.im0 = imgout.reshape(self.ysize, self.xsize)
                if self.ovrf == 'absolute':
                    self.im0 = np.abs(self.im0)
                elif self.ovrf == 'zero':
                    self.im0[self.im0 < 0] = 0
            else:
                raw = Image.open(self.target).convert('F')
                self.im0 = np.array(raw, dtype=np.float32)
                self.ysize, self.xsize = self.im0.shape

            if self.cmave and (self.im0.shape == self.cmsav.shape == self.cmprv.shape):
                if not np.all(self.im0 == self.cmprv):
                    self.cmcnt += 1
                    c = self.cmcnt
                    self.cmprv = self.im0
                    self.im0 = self.cmsav*(c-1)/c + self.im0/c
                else:
                    self.im0 = self.cmsav
            else:
                self.cmcnt = 1
                self.cmprv = self.im0

            self.cmsav = self.im0
            self.imtime = datetime.now()
            return 0
        except FileNotFoundError as e:
            return e
        except ValueError as e:
            return e
        except AttributeError as e:
            return e
        except UIError as e:
            return e
        except:
            import traceback
            return traceback.format_exc()

    @QtCore.pyqtSlot()
    def load_cycle(self):
        while self.run:
            t0 = time.time()
            s = self.load_source()
            self.signal.emit()
            t1 = time.time() - t0

            if self.dt > t1:
                time.sleep(self.dt - t1)

        self.finished.emit()

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.new = True
        self.status = False
        self.saveset_file = None
        self.liveview = False
        self.def_ns1 = 3.0
        self.def_ns2 = 6.0
        self.def_nsi = 5
        self.def_thrs = 10000
        self.def_bkgr = 3000
        self.tf = tempfile.NamedTemporaryFile(prefix='viola_')
        self.dump = np.memmap(self.tf, dtype='float64', mode='w+', shape=6)
        self.rselect_pos = None
        self.use_bkgimg = False
        self.fp_bkgimg = ''
        self.bkgimg = 0
        self.call_extfunc = False
        self.tpool = ThreadPoolExecutor(max_workers=3)
        self.prev_img = pathl
        self.prev_set = pathl
        self.prev_res = pathl.home()/'Pictures'
        self.prev_bkg = pathl.home()/'Pictures'
        self.bkg_dir = pathl.home()/'Pictures'

        self._main = QtWidgets.QWidget()
        self.setStyleSheet(windowstyle)
        self.setCentralWidget(self._main)
        self.setWindowTitle('Viola')

        # Define menue bar
        menu = self.menuBar()
        filemenu = self.init_filemenu()
        menu.addMenu(filemenu)
        optmenu = self.init_option()
        menu.addMenu(optmenu)

        # Define status bar
        self.stts = self.statusBar()
        self.stts.showMessage('Status: Stop')

        self.ddate = QtWidgets.QLabel('Data-date: ')
        self.stts.addPermanentWidget(self.ddate)

        layout = QtWidgets.QHBoxLayout(self._main)

        # Define fisrt column
        maintab = QtWidgets.QVBoxLayout()

        ctrl = self.init_ctrlb()

        canv1 = FigureCanvas(Figure(tight_layout=True))
        self.ax = canv1.figure.add_subplot(111)
        tlbr1 = NavigationToolbar(canv1, self)
        tlbr1.setStyleSheet("background-color: white;")

        self.cmave = QtWidgets.QCheckBox('Cumulative Average: n=0')
        tvbar = QtWidgets.QHBoxLayout()
        tvbar.addStretch()
        tvbar.addWidget(self.cmave)

        sldxy = self.init_sldxy()
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine|QtWidgets.QFrame.Sunken)
        nsmask = self.init_nsmask()

        maintab.addLayout(ctrl)
        maintab.addLayout(tvbar)
        maintab.addWidget(tlbr1)
        maintab.addWidget(canv1)
        maintab.addLayout(sldxy)
        maintab.addWidget(separator)
        maintab.addLayout(nsmask)
        layout.addLayout(maintab)

        # Define second column
        subtab = QtWidgets.QVBoxLayout()

        tree, self.tmodel = self.init_ptree()
        tree.setMinimumHeight(250)
        tree.clicked.connect(lambda i: self.copy_item(self.tmodel, i))
        header = tree.header()
        header.setSectionsClickable(True)
        header.setSectionsMovable(False)
        header.sectionClicked.connect(lambda i: self.copy_ptree(self.tmodel, i))
        ##subtab.addWidget(tree)

        canv2 = FigureCanvas(Figure(tight_layout=True))
        canv2.setMinimumHeight(200)
        self.bx = canv2.figure.add_subplot(111)
        sldns = self.init_sldns()
        ##subtab.addWidget(canv2)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split.addWidget(tree)
        split.addWidget(canv2)
        split.setCollapsible(0, False)
        split.setCollapsible(1, False)
        subtab.addWidget(split)
        subtab.addLayout(sldns)

        boxl = QtWidgets.QTableWidget()
        boxl.setLayout(subtab)
        boxl.setFixedWidth(420)
        layout.addWidget(boxl)

        self.storage = ImageStorage()
        self.ima = ImageAnalysis(self.storage.xsize, self.storage.ysize)
        #canv2.mpl_connect('motion_notify_event', self.mouse_on_stat)

        self.ima.update_coord()

        self.thread = QtCore.QThread(self)
        self.storage.moveToThread(self.thread)
        self.storage.signal.connect(lambda: self.update(2))
        self.storage.finished.connect(self.live_stop)
        self.storage.finished.connect(self.thread.quit)
        self.thread.started.connect(self.storage.load_cycle)

        self.cmave.setChecked(self.storage.cmave)
        self.cmave.stateChanged.connect(self.set_cmave)

    @QtCore.pyqtSlot()
    def update(self, mode):
        t0 = time.time()

        if self.new:
            mode = 0
            self.new = False

        if mode != 1:
            if not hasattr(self.storage, 'im0'):
                status = self.storage.load_source()
                if status != 0:
                    self.new = True
                    self.errormsg(str(status))
                    return None

            imt = self.storage.im0
            if self.use_bkgimg and self.fp_bkgimg != '':
                shape = self.bkgimg.shape if hasattr(self.bkgimg, 'shape') else (1, 1)
                if imt.shape == shape:
                    imt = imt - self.bkgimg
                else:
                    self.errormsg(
                        'Input image shape does not match with the background image.')
                    self.new = True
                    return None

            if self.storage.norm:
                maxv = np.max([1, imt.max()])
                imt = imt/maxv*self.storage.nfac

            self.ima.load_img(imt)
            self.update_status()

        if not self.status:
            return None

        try:
            valt = float(self.valt.text())
            valb = float(self.valb.text())
            self.ima.pre_analysis(valt, valb)
            ns1 = float(self.ns1.text())
            ns2 = float(self.ns2.text())
            nsi = int(self.nsi.text())
            self.ima.analysis(s1=ns1, s2=ns2, itr=nsi)
            t1 = time.time()
            self.ima.plot_main(self.ax, mode, rescale=self.ckrsc.isChecked())
            sldx = self.sldx.value()/100.0
            sldy = self.sldy.value()/100.0
            self.ima.plot_hist(self.ax, sldx, sldy, self.ckcen.isChecked())
            self.ima.set_line_visible()
            self.ax.set_aspect('auto')
            self.ax.figure.canvas.draw_idle()
            #self.ax.figure.canvas.blit(self.ax.figure.canvas.figure.bbox)
            self.ima.plot_stat(self.bx)
            self.bx.set_xscale('log')
            self.bx.grid(True)
            self.bx.autoscale_view()
            self.bx.figure.canvas.draw_idle()
            #self.bx.figure.canvas.blit(self.bx.figure.canvas.figure.bbox)
            t2 = time.time()
            self.update_ptree()
            logging.info('update calculation time [s]: {}'.format(t2-t0))
        except:
            pass

    def update_status(self):
        self.ddate.setText('Data-date: ' + self.storage.imtime.strftime('%Y/%m/%d %H:%M:%S'))
        self.cmave.setText('Cumulative Average: n=' + str(self.storage.cmcnt))

    def set_cmave(self):
        self.storage.cmave = self.cmave.isChecked()

    def mouse_on_stat(self, event):
        ret = self.ima.mouse_on_stat(event)
        self.bx.figure.canvas.draw_idle()
        if ret is not None:
            self.valt.setText('{:.2f}'.format(ret[0]))
            self.valb.setText('{:.2f}'.format(ret[1]))
            self.update(1)

    def init_filemenu(self):
        filemenu = QtWidgets.QMenu("&File", self)

        open_i = QtWidgets.QAction('Open Image', self)
        open_i.triggered.connect(self.openfile)
        open_s = QtWidgets.QAction('Open Setting', self)
        open_s.triggered.connect(self.openset)

        saveset = QtWidgets.QAction('Save Setting', self)
        saveset.triggered.connect(lambda: self.saveset(False))

        savesetas = QtWidgets.QAction('Save Setting As...', self)
        savesetas.triggered.connect(lambda: self.saveset(True))

        saveresult = QtWidgets.QAction('Save Result', self)
        saveresult.triggered.connect(self.saveresult)

        actexit = QtWidgets.QAction('Exit', self)
        actexit.triggered.connect(QtWidgets.qApp.quit)

        filemenu.addAction(open_i)
        filemenu.addAction(open_s)
        filemenu.addSeparator()
        filemenu.addAction(saveset)
        filemenu.addAction(savesetas)
        filemenu.addAction(saveresult)
        filemenu.addSeparator()
        filemenu.addAction(actexit)
        filemenu.setStyleSheet(menustyle)

        return filemenu

    def openfile(self):
        self.live_stop()
        title = 'Open Image File'
        exts = 'Images (*.png *.jpg *.tif *.tiff)'
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, title, str(self.prev_img), exts)
        if filename != '':
            self.etarget.setText(filename)
            self.storage.target = filename
            self.prev_img = Path(filename).absolute().parent
            self.one_play()

    def openset(self):
        self.live_stop()
        title = 'Open Setting File'
        exts = 'Json File (*.json);;All Files (*)'
        cwd = os.path.split(os.path.abspath( __file__))[0]
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, title, str(self.prev_set), exts)
        if filename != '':
            self.prev_set = Path(filename).absolute().parent
        self.loadset(filename)

    def loadset(self, filename):
        self.live_stop()
        if filename != '':
            try:
                self.setWindowTitle('Viola:'+filename)
                self.saveset_file = filename

                with open(self.saveset_file, 'r') as f:
                    d = OrderedDict(json.load(f))

                d1 = d['Source']
                self.storage.target = d1['Target']
                self.etarget.setText(d1['Target'])
                self.storage.xsize  = d1['X pixel']
                self.storage.ysize  = d1['Y pixel']
                self.storage.norm   = d1['Normilized']
                self.storage.nfac   = d1['Norm. factor']
                if 'Overflow handling' in d1:
                    self.storage.ovrf = d1['Overflow handling']
                if 'Call external function' in d1:
                    self.call_extfunc = d1['Call external function']

                d2 = d['Fiducial']
                self.ima.xwidth = d2['X width']
                self.ima.yheight = d2['Y height']
                self.ima.pts1[0] = d2['Top-Left']
                self.ima.pts1[1] = d2['Top-Right']
                self.ima.pts1[2] = d2['Bottom-Left']
                self.ima.pts1[3] = d2['Bottom-Right']
                self.ima.xangle = d2['X angle of incident']
                self.ima.yangle = d2['Y angle of incident']
                if 'X offset' in d2:
                    self.ima.xofs = d2['X offset']
                else:
                    self.ima.xofs = 0.0
                if 'Y offset' in d2:
                    self.ima.yofs = d2['Y offset']
                else:
                    self.ima.yofs = 0.0

                d3 = d['Margin']
                self.ima.mrgn[0] = d3['Top']
                self.ima.mrgn[1] = d3['Bottom']
                self.ima.mrgn[2] = d3['Left']
                self.ima.mrgn[3] = d3['Right']

                d4 = d['Analysis']
                self.sldt.setValue(2**16)
                self.sldb.setValue(0)
                self.valt.setText('{}'.format(int(d4['Threshold'])))
                self.valb.setText('{}'.format(int(d4['Background'])))
                self.ns1.setText('{}'.format(d4['N-sigma 1']))
                self.ns2.setText('{}'.format(d4['N-sigma 2']))
                self.nsi.setText('{}'.format(int(d4['N-sigma iteration'])))

                if 'Smoothing filter' in d4:
                    self.ima.fltr = d4['Smoothing filter']
                if 'Filter type' in d4:
                    self.ima.fltr_mtd = d4['Filter type']
                if 'Filter size' in d4:
                    self.ima.fltr_size = d4['Filter size']

                if 'Background subtraction' in d4:
                    self.use_bkgimg = d4['Background subtraction']
                if 'Background image file' in d4:
                    self.fp_bkgimg = d4['Background image file']

                    try:
                        raw = Image.open(self.fp_bkgimg).convert('F')
                        self.bkgimg = np.array(raw, dtype=np.float32)
                    except FileNotFoundError as e:
                        self.errormsg(str(e))
                    except:
                        import traceback
                        self.errormsg(str(traceback.format_exc()))

                self.ima.update_coord()
                self.one_play()
                if d1['X pixel'] != self.storage.xsize or d1['Y pixel'] != self.storage.ysize:
                    Message(self, 'X-Y pixel sizes are updated automatilcally.\nRecalibation of the transformation will be required.', 'Warning')
                self.valns_changed('t')
            except:
                self.setWindowTitle('Viola')
                self.saveset_file = None
                import traceback
                self.errormsg(traceback.format_exc())

    def get_setting(self):
        d1 = OrderedDict()
        d1['Target'] = self.storage.target
        d1['X pixel'] = self.storage.xsize
        d1['Y pixel'] = self.storage.ysize
        d1['Normilized'] = self.storage.norm
        d1['Norm. factor'] = self.storage.nfac
        d1['Overflow handling'] = self.storage.ovrf
        d1['Call external function'] = self.call_extfunc
        d1['Timestamp'] = self.storage.imtime.strftime('%Y/%m/%d %H:%M:%S')

        d2 = OrderedDict()
        d2['X width'] = self.ima.xwidth
        d2['Y height'] = self.ima.yheight
        d2['Top-Left'] = NoIndent(self.ima.pts1[0].tolist())
        d2['Top-Right'] = NoIndent(self.ima.pts1[1].tolist())
        d2['Bottom-Left'] = NoIndent(self.ima.pts1[2].tolist())
        d2['Bottom-Right'] = NoIndent(self.ima.pts1[3].tolist())
        d2['X angle of incident'] = self.ima.xangle
        d2['Y angle of incident'] = self.ima.yangle
        d2['X offset'] = self.ima.xofs
        d2['Y offset'] = self.ima.yofs
        d2['Transformed pixel size'] = NoIndent(self.ima.dsize)
        d2['X min to max'] = NoIndent([self.ima.ximin, self.ima.ximax])
        d2['Y min to max'] = NoIndent([self.ima.yimin, self.ima.yimax])

        d3 = OrderedDict()
        d3['Top'] = float(self.ima.mrgn[0])
        d3['Bottom'] = float(self.ima.mrgn[1])
        d3['Left'] = float(self.ima.mrgn[2])
        d3['Right'] = float(self.ima.mrgn[3])

        d4 = OrderedDict()
        d4['Threshold'] = float(self.valt.text())
        d4['Background'] = float(self.valb.text())
        d4['N-sigma 1'] = float(self.ns1.text())
        d4['N-sigma 2'] = float(self.ns2.text())
        d4['N-sigma iteration'] = int(self.nsi.text())

        d4['Smoothing filter'] = self.ima.fltr
        d4['Filter type'] = self.ima.fltr_mtd
        d4['Filter size'] = int(self.ima.fltr_size)

        d4['Background subtraction'] = self.use_bkgimg
        if self.use_bkgimg:
            d4['Background image file'] = self.fp_bkgimg

        d = OrderedDict()
        d['Source'] = d1
        d['Fiducial'] = d2
        d['Margin'] = d3
        d['Analysis'] = d4

        return d

    def get_keyname(self, opt = ''):
        base = os.path.split(self.storage.target)[1]
        name, _ = os.path.splitext(base)
        key = name.replace(':image1', '').replace(':ArrayData', '').replace(':', '_')
        now = datetime.now()
        key += opt
        key += now.strftime('_%Y%m%d_%H%M%S')
        return key

    def saveset(self, new):
        if not hasattr(self.storage, 'im0'):
            self.errormsg('No Data to Save')
            return None

        if (self.saveset_file is None) or new:
            title = 'Save Setting File'
            exts = 'Json File (*.json)'
            defname = str(self.prev_set/self.get_keyname())
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, defname, exts)
            if filename != '':
                root, ext = os.path.splitext(filename)
                if ext == '':
                    filename += '.json'
            else:
                return None
        else:
            filename = self.saveset_file

        try:
            with open(filename, 'w') as f:
                f.write(json.dumps(self.get_setting(), indent=4, cls=NoIndentEncoder))
            self.prev_set = Path(filename).absolute().parent
            self.setWindowTitle('Viola:'+filename)
            self.saveset_file = filename
        except:
            import traceback
            self.errormsg(traceback.format_exc())

    def saveresult(self):
        if not hasattr(self.storage, 'im0'):
            self.errormsg('No Image to Save')
            return None

        title = 'Save Result File'
        ext0 = 'Raw + Screen + Json File (*.tiff + *.png + *.json)'
        ext1 = 'Raw Image (*.tiff)'
        ext2 = 'Screenshot (*.png)'
        ext3 = 'Json File (*.json)'
        ext4 = 'Transformed Image Data (*.npy)'
        exts = ext0+';;'+ext1+';;'+ext2+';;'+ext3+';;'+ext4

        defname = str(self.prev_res/self.get_keyname())
        filename, ext = QtWidgets.QFileDialog.getSaveFileName(self, title, defname, exts)

        if filename != '':
            self.prev_res = Path(filename).absolute().parent
            try:
                root, fext = os.path.splitext(filename)
                if ext in [ext0, ext1]:
                    iout = Image.fromarray(self.storage.im0.astype(np.uint16))
                    iout.save(root+'.tiff')
                if ext in [ext0, ext2]:
                    screen = QtWidgets.QWidget.grab(self)
                    screen.save(root+'.png', 'png')
                if ext in [ext0, ext3]:
                    d0 = get_stat(self.ima.cen, self.ima.cov)
                    d0['unit'] = 'mm'
                    d0['Total count'] = float(self.ima.total)
                    d0['Background level'] = float(self.ima.bkgrlvl)
                    d0['Background RMS'] = float(self.ima.bkgrrms)

                    d = self.get_setting()
                    if ext in [ext0]:
                        d['Source']['Origin'] = d['Source']['Target']
                        d['Source']['Target'] = root+'.tiff'
                    d['Result'] = d0
                    with open(root+'.json', 'w') as f:
                        f.write(json.dumps(d, indent=4, cls=NoIndentEncoder))
                if ext in [ext4]:
                    np.save(root+'.npy', self.ima.im)
            except:
                import traceback
                self.errormsg(str(traceback.format_exc()))
                #self.errormsg('Failed to save file.')

    def init_option(self):
        setmenu = QtWidgets.QMenu("&Option", self)
        acttran = QtWidgets.QAction('Transform && Trim', self)
        actslic = QtWidgets.QAction('Slice View', self)
        actpref = QtWidgets.QAction('Preference', self)

        acttran.triggered.connect(self.diag_tran)
        actslic.triggered.connect(self.diag_slic)
        actpref.triggered.connect(self.diag_pref)

        setmenu.addAction(acttran)
        setmenu.addAction(actslic)
        setmenu.addAction(actpref)
        setmenu.setStyleSheet(menustyle)

        return setmenu

    def init_ctrlb(self):
        row = QtWidgets.QHBoxLayout()

        txt1 = QtWidgets.QLabel('Source: ', self)
        txt1.setFixedWidth(70)
        self.etarget = QtWidgets.QLineEdit(self)
        self.etarget.setPlaceholderText('PV name or Image file path')
        self.etarget.editingFinished.connect(self.set_target)
        txt3 = QtWidgets.QLabel('', self)
        txt3.setFixedWidth(20)

        row.addWidget(txt1)
        row.addWidget(self.etarget)
        row.addWidget(txt3)

        self.live = QtWidgets.QPushButton('- Live', self)

        stop = QtWidgets.QPushButton(self)
        #prev = QtWidgets.QPushButton(self)
        updt = QtWidgets.QPushButton(self)
        #frwd = QtWidgets.QPushButton(self)
        snap = QtWidgets.QPushButton(self)

        stop.setIcon(QtGui.QIcon(str(pathl/'icons'/'stop.svg')))
        #prev.setIcon(QtGui.QIcon(path+'/icons/prev.svg'))
        updt.setIcon(QtGui.QIcon(str(pathl/'icons'/'play_pause.svg')))
        #frwd.setIcon(QtGui.QIcon(path+'/icons/next.svg'))
        snap.setIcon(QtGui.QIcon(str(pathl/'icons'/'snap.svg')))

        self.live.clicked.connect(self.live_start)
        stop.clicked.connect(self.live_stop)

        updt.clicked.connect(self.one_play)
        snap.clicked.connect(lambda: self.take_screenshot('Viola_', self))
        #frwd.clicked.connect(self.tmp)

        self.live.setFixedSize(70, 27)
        self.live.setStyleSheet('color: black;')
        row.addWidget(self.live)

        stop.setIconSize(QtCore.QSize(11, 11))
        #for btn in [stop, prev, updt, frwd, snap]:
        for btn in [stop, updt, snap]:
            btn.setFixedSize(27, 27)
            row.addWidget(btn)
        row.setAlignment(QtCore.Qt.AlignRight)
        return row

    def set_target(self):
        self.storage.target = self.etarget.text()

    def live_start(self):
        self.liveview = True
        self.live.setText('\u2022 Live')
        self.live.setStyleSheet('color: red;')
        self.stts.showMessage('Status: Live')
        self.storage.run = True
        self.thread.start()

    def live_stop(self):
        self.liveview = False
        self.storage.run = False
        self.live.setText('- Live')
        self.live.setStyleSheet('color: black;')
        self.stts.showMessage('Status: Stop')

    def one_play(self):
        status = self.storage.load_source()
        if status == 0:
            self.status = True
            self.update(2)
        else:
            #self.new = True
            self.status = False
            self.errormsg(str(status))

    def tmp(self):
        logging.info('thread isRunning : '+str(self.thread.isRunning()))
        logging.info('thread isFinished: '+str(self.thread.isFinished()))

    def init_sldxy(self):
        row = QtWidgets.QHBoxLayout()
        txt0 = QtWidgets.QLabel(' ')
        txt1 = QtWidgets.QLabel('X')
        txt2 = QtWidgets.QLabel('Y')
        self.sldx = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.sldy = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.sldx.setRange(1, 100)
        self.sldy.setRange(1, 100)
        self.sldx.setValue(100)
        self.sldy.setValue(100)
        self.sldx.valueChanged.connect(lambda: self.sldxy_changed('x'))
        self.sldy.valueChanged.connect(lambda: self.sldxy_changed('y'))
        self.cksyc = QtWidgets.QCheckBox('Sync')
        self.ckcen = QtWidgets.QCheckBox('Centering')
        self.ckrsc = QtWidgets.QCheckBox('Rescaling')
        self.ckrsc.setChecked(True)
        self.cksyc.clicked.connect(lambda: self.sldxy_changed('x'))
        self.ckcen.clicked.connect(lambda: self.sldxy_changed(None))
        self.ckrsc.clicked.connect(self.ckrsc_changed)

        row.addWidget(txt0)
        row.addWidget(txt1)
        row.addWidget(self.sldx)
        row.addWidget(txt0)
        row.addWidget(txt2)
        row.addWidget(self.sldy)
        row.addWidget(txt0)
        row.addWidget(self.cksyc)
        row.addWidget(self.ckcen)
        row.addWidget(self.ckrsc)

        return row

    def init_sldns(self):
        col = QtWidgets.QVBoxLayout()

        row1 = QtWidgets.QHBoxLayout()
        txt1 = QtWidgets.QLabel('Threshold: ')
        self.valt = QtWidgets.QLineEdit(self)
        self.valt.setText('{}'.format(self.def_thrs))
        self.valt.setValidator(Qivalp)
        self.valt.editingFinished.connect(lambda: self.valns_changed('t'))
        row1.addWidget(txt1)
        row1.addWidget(self.valt)
        self.sldt = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.sldt.setRange(0, 2**16)
        self.sldt.setValue(self.def_thrs)
        self.sldt.sliderReleased.connect(lambda: self.sldns_changed('t', False))
        #self.sldt.valueChanged.connect(lambda: self.sldns_changed('t', False))

        row2 = QtWidgets.QHBoxLayout()
        txt2 = QtWidgets.QLabel('Background: ')
        self.valb = QtWidgets.QLineEdit(self)
        self.valb.setText('{}'.format(self.def_bkgr))
        self.valb.setValidator(Qivalp)
        self.valb.editingFinished.connect(lambda: self.valns_changed('b'))
        row2.addWidget(txt2)
        row2.addWidget(self.valb)
        self.sldb = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.sldb.setRange(0, 2**16)
        self.sldb.setValue(self.def_bkgr)
        self.sldb.sliderReleased.connect(lambda: self.sldns_changed('b', False))
        #self.sldb.valueChanged.connect(lambda: self.sldns_changed('b', False))

        col.addLayout(row1)
        col.addWidget(self.sldt)

        col.addLayout(row2)
        col.addWidget(self.sldb)

        return col

    def init_nsmask(self):
        row = QtWidgets.QHBoxLayout()
        txt1 = QtWidgets.QLabel('N-\u03C3 masking: ')
        txt2 = QtWidgets.QLabel('N-\u03C31 =')
        self.ns1 = QtWidgets.QLineEdit(self)
        self.ns1.setText('{}'.format(self.def_ns1))
        self.ns1.setValidator(Qdvalp)
        txt3 = QtWidgets.QLabel(', N-\u03C32 =')
        self.ns2 = QtWidgets.QLineEdit(self)
        self.ns2.setText('{}'.format(self.def_ns2))
        self.ns2.setValidator(Qdvalp)
        txt4 = QtWidgets.QLabel(', iteration =')
        self.nsi = QtWidgets.QLineEdit(self)
        self.nsi.setText('{}'.format(self.def_nsi))
        self.nsi.setValidator(Qivalp)

        for ns in [self.ns1, self.ns2, self.nsi]:
            ns.setFixedWidth(40)
            ns.setAlignment(QtCore.Qt.AlignCenter)
            ns.editingFinished.connect(self.nsmask_changed)

        row.addStretch()
        row.addWidget(txt1)
        row.addWidget(txt2)
        row.addWidget(self.ns1)
        row.addWidget(txt3)
        row.addWidget(self.ns2)
        row.addWidget(txt4)
        row.addWidget(self.nsi)
        return row

    def init_ptree(self):
        tree = QtWidgets.QTreeView(self)
        tmodel = QtGui.QStandardItemModel()
        tmodel.setHorizontalHeaderLabels(['Parameter', 'Value', 'Unit'])
        tree.setModel(tmodel)
        tree.setColumnWidth(0, 200)
        tree.setColumnWidth(1, 120)
        tree.setColumnWidth(2, 70)
        tree.setAlternatingRowColors(True)
        return tree, tmodel

    def update_ptree(self):
        self.tmodel.removeRows(0, self.tmodel.rowCount())
        root = self.tmodel.invisibleRootItem()

        def qi(x=None):
            qsi = QtGui.QStandardItem(x)
            qsi.setEditable(False)
            return qsi

        d = get_stat(self.ima.cen, self.ima.cov)

        for i, name in enumerate(d.keys()):
            value = '{:.4f}'.format(d[name])
            root.appendRow([qi(name), qi(value), qi(sunit[name])])
            self.dump[i] = d[name]

        value = '{:.4e}'.format(self.ima.total)
        root.appendRow([qi('Total count'), qi(value), qi('')])
        value = '{:.2f}'.format(self.ima.bkgrlvl)
        root.appendRow([qi('Background level'), qi(value), qi('(est.)')])
        value = '{:.2f}'.format(self.ima.bkgrrms)
        root.appendRow([qi('Background RMS'), qi(value), qi('(est.)')])

        d['Total count'] = self.ima.total
        if self.call_extfunc and extfunc is not None:
            def call_extfunc_status(f):
                self.call_extfunc = f.result()

            job = self.tpool.submit(extfunc, name=self.storage.target, info=d)
            job.add_done_callback(call_extfunc_status)

        self.dump[-1] = self.ima.total

    def copy_ptree(self, model, idx):
        row = model.rowCount()
        if row > 0:
            text = ''
            for i in range(row):
                text += model.item(i, idx).text()
                text += '\n'

            cb = QtWidgets.QApplication.clipboard()
            cb.clear(mode=cb.Clipboard)
            cb.setText(text, mode=cb.Clipboard)

    def copy_item(self, model, qmi):
        r = qmi.row()
        c = qmi.column()
        text = model.item(r, c).text()
        if text is None:
            text = ''
        cb = QtWidgets.QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        cb.setText(text, mode=cb.Clipboard)

    def sldxy_changed(self, crd):
        if self.cksyc.isChecked():
            if crd == 'x':
                self.sldy.setValue(self.sldx.value())
            elif crd == 'y':
                self.sldx.setValue(self.sldy.value())

        if not self.new:
            sldx = self.sldx.value()/100.0
            sldy = self.sldy.value()/100.0
            self.ima.plot_hist(self.ax, sldx, sldy, self.ckcen.isChecked())
            self.ax.figure.canvas.draw_idle()

    def ckrsc_changed(self):
        if not self.new:
            self.ima.plot_main(self.ax, 2, rescale=self.ckrsc.isChecked())
            self.ax.figure.canvas.draw_idle()

    def sldns_changed(self, src, released):
        valt = self.sldt.value()
        valb = self.sldb.value()
        if valt < valb:
            if src == 't':
                valt = valb
                self.sldt.setValue(valt)
            elif src == 'b':
                valb = valt
                self.sldb.setValue(valb)
        self.valt.setText('{}'.format(valt))
        self.valb.setText('{}'.format(valb))
        opt = False
        if released and self.liveview :
            opt = True
        elif not released and not self.liveview:
            opt = True
        if not self.new and opt:
            self.update(1)

    def valns_changed(self, src):
        valt = int(self.valt.text())
        valb = int(self.valb.text())
        if valt < valb:
            if src == 't':
                valt = valb
                self.valt.setText('{}'.format(int(valt)))
            elif src == 'b':
                valb = valt
                self.valb.setText('{}'.format(int(valb)))
        self.sldt.setValue(valt)
        self.sldb.setValue(valb)

    def nsmask_changed(self):
        if not self.new:
            self.update(1)

    def diag_tran(self):
        self.whopt = QtWidgets.QDialog()
        self.whopt.setWindowTitle('Transform & Trim')
        self.whopt.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.Drawer |\
                                  QtCore.Qt.WindowStaysOnTopHint)
        self.whopt.setStyleSheet(windowstyle)

        layout = QtWidgets.QHBoxLayout()

        fcol = QtWidgets.QVBoxLayout()
        canv = FigureCanvas(Figure(tight_layout=True))
        cx = canv.figure.add_subplot(111)
        tlbr = NavigationToolbar(canv, self)
        tlbr.setStyleSheet("background-color: white;")

        fcol.addWidget(tlbr)
        fcol.addWidget(canv)

        scol = QtWidgets.QVBoxLayout()
        box1 = QtWidgets.QGroupBox('- Image')
        box2 = QtWidgets.QGroupBox('- Fiducials')
        box3 = QtWidgets.QGroupBox('- Margins')
        box1.setFixedWidth(350)
        box2.setFixedWidth(350)
        box3.setFixedWidth(350)

        lb1 = QtWidgets.QVBoxLayout()
        lxpx, txpx = self.textline('X size:', 'pixel', width=[80, 100, 80])
        lypx, typx = self.textline('Y size:', 'pixel', width=[80, 100, 80])
        txpx.setValidator(Qivalp)
        typx.setValidator(Qivalp)
        lb1.addLayout(lxpx)
        lb1.addLayout(lypx)
        box1.setLayout(lb1)

        lb2 = QtWidgets.QVBoxLayout()
        lfdw, tfdw = self.textline('Width :', unit='mm', width=[80, 100, 50])
        lfdh, tfdh = self.textline('Height:', unit='mm', width=[80, 100, 50])
        tfdw.setValidator(QtGui.QDoubleValidator())
        tfdh.setValidator(QtGui.QDoubleValidator())
        lb2.addLayout(lfdw)
        lb2.addLayout(lfdh)
        vpts1 = []
        for s in ['Top-Left  ', 'Top-Right ', 'Bot-Left  ', 'Bot-Right ']:
            lt, ttx, tty = self.textline2(s+'pixel (x, y):', width=[220, 100, 100])
            ttx.setValidator(QtGui.QDoubleValidator())
            tty.setValidator(QtGui.QDoubleValidator())
            vpts1.append([ttx, tty])
            lb2.addLayout(lt)

        lphi, ttxp, ttyp = \
            self.textline2('Angle of incidence (x, y) [deg]:', width=[310, 100, 100])
        ttxp.setValidator(QtGui.QDoubleValidator())
        ttyp.setValidator(QtGui.QDoubleValidator())

        lofs, txof, tyof = \
            self.textline2('Center offset (x, y) [mm]:', width=[310, 100, 100])
        txof.setValidator(QtGui.QDoubleValidator())
        tyof.setValidator(QtGui.QDoubleValidator())

        lb2.addLayout(lphi)
        lb2.addLayout(lofs)
        box2.setLayout(lb2)
        lb3 = QtWidgets.QVBoxLayout()
        vmrgn = []
        for s in ['Top    :', 'Bottom :', 'Left   :', 'Right  :']:
            lt, tt = self.textline(s, 'pixel', width=[80, 100, 50])
            tt.setValidator(QtGui.QDoubleValidator())
            vmrgn.append(tt)
            lb3.addLayout(lt)
        box3.setLayout(lb3)

        bts = QtWidgets.QGridLayout()
        rld =  QtWidgets.QPushButton('Reload', self)
        rst =  QtWidgets.QPushButton('Reset', self)
        dft =  QtWidgets.QPushButton('Default', self)
        apl =  QtWidgets.QPushButton('Apply', self)
        okk =  QtWidgets.QPushButton('OK', self)
        cnl =  QtWidgets.QPushButton('Cancel', self)

        for b in [rld, rst, dft, apl, okk, cnl]:
            b.setAutoDefault(False)
            b.setFixedHeight(25)

        bts.addWidget(rld, 1, 1)
        bts.addWidget(rst, 1, 2)
        bts.addWidget(dft, 1, 3)
        bts.addWidget(apl, 2, 1)
        bts.addWidget(okk, 2, 2)
        bts.addWidget(cnl, 2, 3)

        scol.addWidget(box1)
        scol.addWidget(box2)
        scol.addWidget(box3)
        scol.addWidget(QtWidgets.QLabel(''))
        scol.addLayout(bts)
        scol.addStretch()

        layout.addLayout(fcol)
        layout.addLayout(scol)

        ima = self.ima
        self.pts1 = ima.pts1.copy()
        self.mrgn = ima.mrgn.copy()
        self.xangle = ima.xangle*1.0
        self.yangle = ima.yangle*1.0
        self.cximg = None

        def on_mouse_move(event):
            if  self.cximg != None:
                if event.inaxes and tlbr.mode == '':
                    xy = np.array([event.xdata, event.ydata])
                    ar = np.zeros(4)
                    for i, pg in enumerate(self.pga):
                        ar[i] = np.sum((pg[0].xybox - xy)**2)
                        pg[1].set_fill(False)
                        pg[2].set_visible(False)

                    idx = ar.argmin()
                    lim = 1000*1000

                    if ar[idx]<lim:
                        self.pga[idx][1].set_fill(True)
                        self.pga[idx][2].set_visible(True)

                    if event.button == 1 and ar[idx]<lim:
                        self.pga[idx][0].xybox = xy
                        self.pts1[idx] = xy
                        clp = c_cent(self.pts1)
                        self.cln[0].set_positions(clp[0], clp[1])
                        self.cln[1].set_positions(clp[2], clp[3])
                        r = c_mrgr(self.pts1, self.mrgn)[[0, 1, 3, 2]]
                        rt = np.concatenate([r, [r[0]]])
                        self.rpg.set_xy(rt)
                        vpts1[idx][0].setText('{:.1f}'.format(xy[0]))
                        vpts1[idx][1].setText('{:.1f}'.format(xy[1]))
                    cx.figure.canvas.draw_idle()
                else:
                    for i, pg in enumerate(self.pga):
                        pg[1].set_fill(False)
                        pg[2].set_visible(False)
                    cx.figure.canvas.draw_idle()

        def size_changed():
            if txpx.text() in ['', '0']:
                txpx.setText('{}'.format(self.storage.xsize))
            if typx.text() in ['', '0']:
                typx.setText('{}'.format(self.storage.ysize))

            self.storage.xsize = int(txpx.text())
            self.storage.ysize = int(typx.text())

        def xyhw_changed():
            if tfdw.text() in ['']:
                tfdw.setText('{}'.format(ima.xwidth))
            elif float(tfdw.text()) == 0.0:
                tfdw.setText('{}'.format(ima.xwidth))

            if tfdh.text() in ['']:
                tfdh.setText('{}'.format(ima.yheight))
            elif float(tfdh.text()) == 0.0:
                tfdh.setText('{}'.format(ima.yheight))

        def angle_limit(ttvp):
            if ttvp.text() == '':
                ttvp.setText('{:.1f}'.format(0.0))
            elif float(ttvp.text()) > 89.0:
                ttvp.setText('{:.1f}'.format(89.0))
            elif float(ttvp.text()) < -89.0:
                ttvp.setText('{:.1f}'.format(-89.0))

        def pts1_changed(i, j):
            self.pts1[i][j] = float(vpts1[i][j].text())
            if hasattr(self, 'pga'):
                xy = self.pts1[i]
                self.pga[i][0].xybox = xy
                clp = c_cent(self.pts1)
                self.cln[0].set_positions(clp[0], clp[1])
                self.cln[1].set_positions(clp[2], clp[3])
                r = c_mrgr(self.pts1, self.mrgn)[[0, 1, 3, 2]]
                rt = np.concatenate([r, [r[0]]])
                self.rpg.set_xy(rt)
                cx.figure.canvas.draw_idle()

        def mrgn_changed():
            for i, tt in enumerate(vmrgn):
                try:
                    self.mrgn[i] = float(tt.text())
                except:
                    pass
                if hasattr(self, 'rpg'):
                    r = c_mrgr(self.pts1, self.mrgn)[[0, 1, 3, 2]]
                    rt = np.concatenate([r, [r[0]]])
                    self.rpg.set_xy(rt)
                    cx.figure.canvas.draw_idle()

        def cmd_rld():
            status = self.storage.load_source()
            if status == 0:
                if self.cximg == None:
                    self.cximg = cx.imshow(self.storage.im0, ima.hcmap)
                    self.pga, self.rpg, self.cln = \
                        plot_trans(cx, tlbr, self.pts1, self.mrgn, 20, vpts1, ima.lclr)
                else:
                    im0 = self.storage.im0
                    self.cximg.set_data(im0)
                    self.cximg.set_clim(im0.min(), im0.max())
                    self.cximg.set_cmap(ima.hcmap)

                #cx.set_aspect('auto')
                cx.autoscale()
                cx.figure.canvas.draw_idle()
            else:
                self.errormsg(str(status))

        def cmd_rst(mode):
            if mode == 1:
                ima.init_pts1(self.storage.xsize, self.storage.ysize)
                ima.mrgn = np.float32([0, 0, 0, 0])
            self.pts1 = ima.pts1.copy()
            self.mrgn = ima.mrgn.copy()
            txpx.setText('{}'.format(self.storage.xsize))
            typx.setText('{}'.format(self.storage.ysize))
            tfdw.setText('{}'.format(ima.xwidth))
            tfdh.setText('{}'.format(ima.yheight))
            for i, tt in enumerate(vpts1):
                tt[0].setText('{:.1f}'.format(ima.pts1[i][0]))
                tt[1].setText('{:.1f}'.format(ima.pts1[i][1]))
            for i, tt in enumerate(vmrgn):
                tt.setText('{:.1f}'.format(ima.mrgn[i]))

            ttxp.setText('{:.1f}'.format(ima.xangle))
            ttyp.setText('{:.1f}'.format(ima.yangle))
            txof.setText('{:.3f}'.format(ima.xofs))
            tyof.setText('{:.3f}'.format(ima.yofs))

            if hasattr(self, 'pga'):
                for i in range(len(self.pts1)):
                    xy = self.pts1[i]
                    self.pga[i][0].xybox = xy
                clp = c_cent(self.pts1)
                self.cln[0].set_positions(clp[0], clp[1])
                self.cln[1].set_positions(clp[2], clp[3])
                r = c_mrgr(self.pts1, self.mrgn)[[0, 1, 3, 2]]
                rt = np.concatenate([r, [r[0]]])
                self.rpg.set_xy(rt)
                cx.figure.canvas.draw_idle()

        def cmd_apl():
            size_changed()
            xyhw_changed()
            ima.xwidth = float(tfdw.text())
            ima.yheight = float(tfdh.text())
            ima.pts1 = self.pts1.copy()
            ima.mrgn = self.mrgn.copy()
            ima.xangle = float(ttxp.text())
            ima.yangle = float(ttyp.text())
            ima.xofs = float(txof.text())
            ima.yofs = float(tyof.text())
            ima.update_coord()
            self.update(2)

        def cmd_okk():
            cmd_apl()
            self.whopt.accept()

        cx.figure.canvas.mpl_connect('motion_notify_event', on_mouse_move)
        txpx.editingFinished.connect(size_changed)
        typx.editingFinished.connect(size_changed)
        tfdw.editingFinished.connect(xyhw_changed)
        tfdh.editingFinished.connect(xyhw_changed)
        ttxp.editingFinished.connect(lambda: angle_limit(ttxp))
        ttyp.editingFinished.connect(lambda: angle_limit(ttyp))

        for i, tt in enumerate(vpts1):
            tt[0].editingFinished.connect(lambda i=i: pts1_changed(i, 0))
            tt[1].editingFinished.connect(lambda i=i: pts1_changed(i, 1))

        for tt in vmrgn:
            tt.editingFinished.connect(mrgn_changed)

        rld.clicked.connect(cmd_rld)
        rst.clicked.connect(lambda: cmd_rst(0))
        dft.clicked.connect(lambda: cmd_rst(1))
        apl.clicked.connect(cmd_apl)
        okk.clicked.connect(cmd_okk)
        cnl.clicked.connect(lambda: self.whopt.reject())

        if hasattr(self.storage, 'im0'):
            cmd_rld()
        cmd_rst(0)
        self.whopt.setLayout(layout)
        self.whopt.show()

    def diag_slic(self):
        self.wslic = QtWidgets.QDialog()
        self.wslic.setWindowTitle('Slice View')
        self.wslic.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.Drawer |\
                                  QtCore.Qt.WindowStaysOnTopHint)
        self.wslic.setStyleSheet(windowstyle)

        layout = QtWidgets.QHBoxLayout()

        fcol = QtWidgets.QVBoxLayout()

        canv = FigureCanvas(Figure(tight_layout=True))
        dx = canv.figure.add_subplot(111)
        tlbr = NavigationToolbar(canv, self)
        tlbr.setStyleSheet("background-color: white;")

        rld = QtWidgets.QPushButton(self)
        rld.setIcon(QtGui.QIcon(str(pathl/'icons'/'reload.svg')))
        rld.setIconSize(QtCore.QSize(20, 20))
        rld.setFixedSize(30, 30)

        snp = QtWidgets.QPushButton(self)
        snp.setIcon(QtGui.QIcon(str(pathl/'icons'/'snap.svg')))
        snp.setIconSize(QtCore.QSize(20, 20))
        snp.setFixedSize(30, 30)

        brbox = QtWidgets.QHBoxLayout()
        brbox.addWidget(tlbr)
        brbox.addSpacing(20)
        brbox.addWidget(rld)
        brbox.addWidget(snp)
        brbox.addSpacing(10)

        fcol.addLayout(brbox)
        fcol.addWidget(canv)

        gcol = QtWidgets.QTableWidget()
        pbox = QtWidgets.QVBoxLayout()

        tfbox = QtWidgets.QGroupBox('Smoothing Filter')
        tfbox.setCheckable(True)
        tfbox.setChecked(self.ima.fltr)
        tfelm = QtWidgets.QVBoxLayout()

        t1box = QtWidgets.QHBoxLayout()
        txt1 = QtWidgets.QLabel('Smoothing Filter:')
        fmtd = QtWidgets.QComboBox(self)
        fmtd.addItems(['Mean', 'Gaussian'])
        fmtd.setCurrentText(self.ima.fltr_mtd)
        t1box.addWidget(txt1)
        t1box.addWidget(fmtd)
        t1box.addStretch()

        t2box = QtWidgets.QHBoxLayout()
        txt2 = QtWidgets.QLabel('Filter size [pixel]:')
        cavr = QtWidgets.QLineEdit()
        cavr.setText(str(self.ima.fltr_size))
        cavr.setValidator(Qivalp)
        t2box.addWidget(txt2)
        t2box.addWidget(cavr)

        t4box = QtWidgets.QHBoxLayout()
        txt4 = QtWidgets.QLabel('Image size [pixel]: ')
        csiz = QtWidgets.QLabel('')
        t4box.addWidget(txt4)
        t4box.addWidget(csiz)
        tfelm.addLayout(t1box)
        tfelm.addLayout(t2box)
        tfelm.addLayout(t4box)
        tfbox.setLayout(tfelm)

        tgbox = QtWidgets.QGroupBox()
        tgelm = QtWidgets.QVBoxLayout()

        t3box = QtWidgets.QHBoxLayout()
        txt3 = QtWidgets.QLabel('Evaluation:')
        ccrd = QtWidgets.QComboBox(self)
        ccrd.addItems(['X slice', 'Y slice', 'Rectangle'])
        t3box.addWidget(txt3)
        t3box.addWidget(ccrd)
        tgelm.addLayout(t3box)
        tgbox.setLayout(tgelm)

        tree = QtWidgets.QTreeView(self)
        tmodel = QtGui.QStandardItemModel()
        tmodel.setHorizontalHeaderLabels(['Parameter', 'Value', 'Unit'])
        tree.setModel(tmodel)
        tree.setColumnWidth(0, 160)
        tree.setColumnWidth(1, 120)
        tree.setColumnWidth(2, 40)
        tree.setFixedHeight(230)
        tree.setAlternatingRowColors(True)
        tree.clicked.connect(lambda i: self.copy_item(tmodel, i))
        header = tree.header()
        header.setSectionsClickable(True)
        header.setSectionsMovable(False)
        header.sectionClicked.connect(lambda i: self.copy_ptree(tmodel, i))
        tgelm.addWidget(tree)

        if (extkey1 is not None and extkey1 in self.storage.target)\
            or self.call_extfunc:
            addbtn = extbtn1(self, tmodel)
            tgelm.addWidget(addbtn)

        tgelm.addStretch()

        pbox.addWidget(tfbox)
        pbox.addWidget(tgbox)
        pbox.addStretch()
        gcol.setLayout(pbox)
        gcol.setFixedWidth(380)

        layout.addLayout(fcol)
        layout.addWidget(gcol)

        self.dximg = None
        self.dxxbase = None
        self.dxxline = None
        self.dxybase = None
        self.dxyline = None

        ima = self.ima

        def update_tmodel(params):
            # xc, yc, xw, yh, avr, rms, rmsp, m0, m1
            names = ['X centroid', 'Y centroid', 'X width', 'Y height', 'Total count',
                     'Average count', 'RMS error', 'RMS error', 'Minimun', 'Maximum']
            units = ['mm', 'mm', 'mm', 'mm', '', '', '', '%', '', '']
            tmodel.removeRows(0, tmodel.rowCount())
            root = tmodel.invisibleRootItem()

            def qi(x=None):
                qsi = QtGui.QStandardItem(x)
                qsi.setEditable(False)
                return qsi

            for n, v, u in zip(names, params, units):
                if v is not None:
                    value = '{:.2f}'.format(v) if n != 'Total count' else '{:.3e}'.format(v)
                    root.appendRow([qi(n), qi(value), qi(u)])

        def snap(raw, lin):
            return np.argmin(np.abs(lin - raw))

        def get_idx(p, pi):
            pn = len(pi)
            idx = snap(p, pi)
            idx = 0 if idx < 0 else idx
            idx = pn-1 if idx > pn-1 else idx
            return idx

        def draw_line(c, crd, updt=True):
            if crd == 'X':
                if self.dxxbase == None:
                    idx = get_idx(c, ima.yi)
                    lhst = ima.im[idx, :]
                    self.dxxbase = dx.axhline(ima.yi[idx], ls='--', color=ima.lclr[4])
                    self.dxxline = dx.plot(ima.xi, ima.yi[idx] + lhst/ima.im1i*ima.yw/5, color=ima.lclr[4])[0]
                else:
                    c = c if updt else self.dxxbase.get_ydata()[0]
                    idx = get_idx(c, ima.yi)
                    lhst = ima.im[idx, :]
                    self.dxxbase.set_ydata([ima.yi[idx], ima.yi[idx]])
                    self.dxxline.set_ydata(ima.yi[idx] + lhst/ima.im1i*ima.yw/5)
                xc = None
                yc = ima.yi[idx]
            elif crd == 'Y':
                if self.dxybase == None:
                    idx = get_idx(c, ima.xi)
                    lhst = ima.im[:, idx]
                    self.dxybase = dx.axvline(ima.xi[idx], ls='--', color=ima.lclr[4])
                    self.dxyline = dx.plot(ima.xi[idx] + lhst/ima.im1i*ima.xw/5, ima.yi, color=ima.lclr[4])[0]
                else:
                    c = c if updt else self.dxybase.get_xdata()[0]
                    idx = get_idx(c, ima.xi)
                    lhst = ima.im[:, idx]
                    self.dxybase.set_xdata([ima.xi[idx], ima.xi[idx]])
                    self.dxyline.set_xdata(ima.xi[idx] + lhst/ima.im1i*ima.yw/5)
                xc = ima.xi[idx]
                yc = None

            tot = np.sum(lhst)
            avr = np.mean(lhst)
            rms = np.sqrt(ndimage.variance(lhst))
            m0, m1 = np.min(lhst), np.max(lhst)
            update_tmodel([xc, yc, None, None, tot, avr, rms, 100*rms/avr, m0, m1])

        def update_visible():
            if self.dxxbase != None:
                if ccrd.currentText() == 'X slice':
                    self.dxxbase.set_visible(True)
                    self.dxxline.set_visible(True)
                    self.dxybase.set_visible(False)
                    self.dxyline.set_visible(False)
                    self.rselect.set_visible(False)
                    self.rselect.set_active(False)
                elif ccrd.currentText() == 'Y slice':
                    self.dxxbase.set_visible(False)
                    self.dxxline.set_visible(False)
                    self.dxybase.set_visible(True)
                    self.dxyline.set_visible(True)
                    self.rselect.set_visible(False)
                    self.rselect.set_active(False)
                elif ccrd.currentText() == 'Rectangle':
                    self.dxxbase.set_visible(False)
                    self.dxxline.set_visible(False)
                    self.dxybase.set_visible(False)
                    self.dxyline.set_visible(False)
                    self.rselect.set_visible(True)
                    self.rselect.set_active(True)
                try:
                    dx.figure.canvas.draw_idle()
                except:
                    pass

        def cmd_rld():
            if hasattr(ima, 'im') and self.status:
                ima.fltr = tfbox.isChecked()
                csiz.setText(str(ima.dsize))
                #if fmtd.currentText() == 'Mean':
                #    ima.fltr_mtd = 0
                #    ima.fltr_size = np.abs(int(cavr.text()))
                #elif fmtd.currentText() == 'Gaussian':
                #    ima.fltr_mtd = 1
                #    ima.fltr_size = np.abs(int(cavr.text()))

                ima.fltr_mtd = fmtd.currentText()
                ima.fltr_size = int(np.abs(int(cavr.text())))
                self.update(2)

                if self.dximg == None:
                    self.dximg = dx.imshow(ima.im, cmap=ima.hcmap,
                        extent=(ima.xi.min(), ima.xi.max(), ima.yi.min(), ima.yi.max()))
                    dx.set_aspect('auto')
                    dx.set_xlabel('x [mm]')
                    dx.set_ylabel('y [mm]')
                else:
                    self.dximg.set_data(ima.im)
                    self.dximg.set_clim(ima.im.min(), ima.im.max())
                    self.dximg.set_cmap(ima.hcmap)
                    self.dximg.set_extent((ima.xi.min(), ima.xi.max(), ima.yi.min(), ima.yi.max()))

                if self.dxxbase == None:
                    draw_line(ima.yc, 'X')
                if self.dxybase == None:
                    draw_line(ima.xc, 'Y')
                update_visible()

        def set_averaging():
            cmd_rld()
            if self.dxxbase != None:
                if ccrd.currentText() == 'X slice':
                    draw_line(ima.yc, 'X', False)
                elif ccrd.currentText() == 'Y slice':
                    draw_line(ima.xc, 'Y', False)
                elif ccrd.currentText() == 'Rectangle':
                    selected()
                try:
                    dx.figure.canvas.draw_idle()
                except:
                    pass

        def on_mouse_move(event):
            if  self.dximg != None:
                if event.inaxes and tlbr.mode == '':
                    if event.button == 1:
                        if ccrd.currentText() == 'X slice':
                            draw_line(event.ydata, 'X')
                            dx.figure.canvas.draw_idle()
                        elif ccrd.currentText() == 'Y slice':
                            draw_line(event.xdata, 'Y')
                            dx.figure.canvas.draw_idle()
                        elif ccrd.currentText() == 'Rectangle':
                            selected()
                        #canv.blit(canv.figure.bbox)

        def selected(ev1=None, ev2=None):
            self.rselect_pos = self.rselect.extents
            xy = self.rselect.extents
            xmin = snap(xy[0], ima.xi)
            xmax = snap(xy[1], ima.xi)
            ymin = snap(xy[3], ima.yi)
            ymax = snap(xy[2], ima.yi)

            evi = ima.im[ymin:ymax, xmin:xmax]
            xc, yc = self.rselect.center
            xw = xy[1] - xy[0]
            yh = xy[3] - xy[2]
            if evi.size != 0:
                tot = np.sum(evi)
                avr = np.mean(evi)
                rms = np.sqrt(ndimage.variance(evi))
                m0, m1 = np.min(evi), np.max(evi)
                pct = 100*rms/avr
            else:
                tot, avr, rms, m0, m1, pct = 0, 0, 0, 0, 0, 0
            update_tmodel([xc, yc, xw, yh, tot, avr, rms, pct, m0, m1])

        def callbackupdate(event):
            if self.rselect.active:
                self.rselect.update()

        if mpl.__version__ < '3.5':
            self.rselect = RectangleSelector(dx, selected, drawtype='box',
                useblit=False, button=[1], interactive=True, maxdist=20,
                rectprops={'fill':False, 'edgecolor':ima.lclr[4], 'alpha':1, 'lw':4, 'ls':':'})
        else:
            self.rselect = RectangleSelector(dx, selected,
                useblit=False, button=[1], interactive=True,
                props={'fill':False, 'edgecolor':ima.lclr[4], 'alpha':1, 'lw':4, 'ls':':'})

        dx.figure.canvas.mpl_connect('motion_notify_event', on_mouse_move)
        #dx.figure.canvas.mpl_connect('draw_event', callbackupdate)
        rld.clicked.connect(set_averaging)
        snp.clicked.connect(lambda: self.take_screenshot('Viola_', self.wslic))
        tfbox.toggled.connect(set_averaging)
        fmtd.currentIndexChanged.connect(set_averaging)
        ccrd.currentIndexChanged.connect(update_visible)
        ccrd.currentIndexChanged.connect(set_averaging)
        cavr.editingFinished.connect(set_averaging)
        set_averaging()

        #self.rselect.useblit = False
        if self.rselect_pos != None:
            self.rselect.extents = self.rselect_pos
        else:
            self.rselect.extents = (ima.xi.min()/2, ima.xi.max()/2, ima.yi.min()/2, ima.yi.max()/2)
        #self.rselect.useblit = False
        self.rselect.set_active(False)
        self.rselect.set_visible(False)

        self.wslic.setLayout(layout)
        self.wslic.show()

    def diag_pref(self):
        self.wpref = QtWidgets.QDialog()
        self.wpref.setWindowTitle('Preference')
        self.wpref.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.Drawer |\
                                  QtCore.Qt.WindowStaysOnTopHint)
        self.wpref.setStyleSheet(windowstyle)

        layout = QtWidgets.QVBoxLayout()

        gbox1 = QtWidgets.QGroupBox('Normalize Raw Data')
        gbox1.setCheckable(True)
        gbox1.setChecked(self.storage.norm)
        nrml = QtWidgets.QHBoxLayout()
        nrmt = QtWidgets.QLabel(' Normalizing Constant: ')
        nrmv = QtWidgets.QLineEdit(self)
        nrmv.setFixedWidth(100)
        nrmv.setText('{}'.format(self.storage.nfac))
        nrmv.setValidator(QtGui.QDoubleValidator())

        nrml.addWidget(nrmt)
        nrml.addWidget(nrmv)
        nrml.addStretch()
        gbox1.setLayout(nrml)

        gboxb = QtWidgets.QGroupBox('Subtract Background Image')
        gboxb.setCheckable(True)
        gboxb.setChecked(self.use_bkgimg)

        bkgl = QtWidgets.QHBoxLayout()
        bkgt = QtWidgets.QLabel(' File: ')
        bkgt.setFixedWidth(74)
        bkge = QtWidgets.QLineEdit(self)
        bkge.setPlaceholderText('Image file path')
        bkge.setText(self.fp_bkgimg)
        bkge.setReadOnly(True)
        bkge.setMinimumWidth(620)

        bkgb = QtWidgets.QPushButton(self)
        bkgb.setAutoDefault(False)
        bkgb.setFixedSize(25, 25)
        bkgb.setIcon(QtGui.QIcon(str(pathl/'icons'/'file.svg')))
        bkgb.setIconSize(QtCore.QSize(20, 20))
        bkgs = QtWidgets.QPushButton(self)
        bkgs.setAutoDefault(False)
        bkgs.setFixedSize(25, 25)
        bkgs.setIcon(QtGui.QIcon(str(pathl/'icons'/'snap.svg')))
        bkgs.setIconSize(QtCore.QSize(16, 16))

        bkgl.addWidget(bkgt)
        bkgl.addWidget(bkge)
        bkgl.addWidget(bkgb)
        bkgl.addWidget(bkgs)
        bkgl.addStretch()
        gboxb.setLayout(bkgl)

        gboxf = QtWidgets.QGroupBox('Smoothing Filter')
        gboxf.setCheckable(True)
        gboxf.setChecked(self.ima.fltr)

        fbox = QtWidgets.QVBoxLayout()
        f1box = QtWidgets.QHBoxLayout()
        ftxt1 = QtWidgets.QLabel('Smoothing Filter: ')
        fmtd = QtWidgets.QComboBox(self)
        fmtd.addItems(['Mean', 'Gaussian'])
        fmtd.setCurrentText(self.ima.fltr_mtd)
        f1box.addWidget(ftxt1)
        f1box.addWidget(fmtd)
        f1box.addStretch()

        f2box = QtWidgets.QHBoxLayout()
        ftxt2 = QtWidgets.QLabel('Filter size [pixel]: ')
        cavr = QtWidgets.QLineEdit(self)
        cavr.setText(str(self.ima.fltr_size))
        cavr.setValidator(Qivalp)
        cavr.setFixedWidth(100)
        f2box.addWidget(ftxt2)
        f2box.addWidget(cavr)
        f2box.addStretch()

        f3box = QtWidgets.QHBoxLayout()
        ftxt3 = QtWidgets.QLabel('Image size [pixel]: ')
        csiz = QtWidgets.QLabel('')
        if hasattr(self.storage, 'im0'):
            csiz.setText(str(self.ima.dsize))
        f3box.addWidget(ftxt3)
        f3box.addWidget(csiz)
        f3box.addStretch()

        fbox.addLayout(f1box)
        fbox.addLayout(f2box)
        fbox.addLayout(f3box)
        gboxf.setLayout(fbox)

        gbox2 = QtWidgets.QGroupBox('- Dump File: {}'.format(self.tf.name))
        fd1 = QtWidgets.QVBoxLayout()
        fd2 = QtWidgets.QHBoxLayout()
        fdn = QtWidgets.QLineEdit(self)

        fdh = QtWidgets.QLabel("Load data by")
        text = "numpy.memmap('{}', dtype='float64', mode='r', shape=6)".format(self.tf.name)
        fdn.setText(text)
        fdn.setReadOnly(True)
        fdn.setMinimumWidth(700)
        fdb = QtWidgets.QPushButton(self)
        fdb.setAutoDefault(False)
        fdb.setFixedSize(25, 25)
        fdb.setIcon(QtGui.QIcon(str(pathl/'icons'/'copy.svg')))
        fdb.setIconSize(QtCore.QSize(25, 25))
        fd2.addWidget(fdn)
        fd2.addWidget(fdb)
        fd2.addStretch()
        fdt = QtWidgets.QLabel(
        "Return array:\n"+\
        "[ X_centroid, Y_centroid, X_RMS, Y_RMS, XY_corelation, Total_count ]")
        fd1.addWidget(fdh)
        fd1.addLayout(fd2)
        fd1.addWidget(fdt)
        gbox2.setLayout(fd1)

        gbox3 = QtWidgets.QGroupBox('- Other Settings')

        t1col = QtWidgets.QVBoxLayout()

        cbext = QtWidgets.QCheckBox(' Update external information')
        cbext.setChecked(self.call_extfunc)

        t1box = QtWidgets.QHBoxLayout()
        txt1 = QtWidgets.QLabel('Overflow handling: ')
        ovrf = QtWidgets.QComboBox(self)
        ovrf.addItems(['absolute', 'zero'])
        ovrf.setCurrentText(self.storage.ovrf)
        t1box.addWidget(txt1)
        t1box.addWidget(ovrf)
        t1box.addStretch()

        t2box = QtWidgets.QHBoxLayout()
        txt2 = QtWidgets.QLabel('Color map: ')
        cmap = QtWidgets.QComboBox(self)
        cmap.addItems(cmaps)
        cmap.setCurrentText(self.ima.hcmap)
        t2box.addWidget(txt2)
        t2box.addWidget(cmap)
        t2box.addStretch()

        t3box = QtWidgets.QHBoxLayout()
        txt3 = QtWidgets.QCheckBox(' Line colors: ')
        txt3.setChecked(self.ima.lvis)
        t3box.addWidget(txt3)
        cboxs = []
        for lc in self.ima.lclr:
            cbox = QtWidgets.QPushButton()
            cbox.setAutoDefault(False)
            cbox.setStyleSheet("background-color: "+lc)
            t3box.addWidget(cbox)
            cboxs.append(cbox)
        t3box.addStretch()

        t1col.addWidget(cbext)
        t1col.addLayout(t1box)
        t1col.addLayout(t2box)
        t1col.addLayout(t3box)
        gbox3.setLayout(t1col)

        layout.addWidget(gbox1)
        layout.addWidget(gboxb)
        layout.addWidget(gboxf)
        layout.addWidget(gbox2)
        layout.addWidget(gbox3)
        layout.addStretch()

        ima = self.ima

        def norm():
            self.storage.norm = gbox1.isChecked()
            self.storage.nfac = float(nrmv.text())
            if hasattr(self.storage, 'im0'):
                self.update(2)

        def ccbd():
            cb = QtWidgets.QApplication.clipboard()
            cb.clear(mode=cb.Clipboard)
            cb.setText(text, mode=cb.Clipboard)

        def covrf():
            self.storage.ovrf = ovrf.currentText()

        def ccmap():
            ima.hcmap = cmap.currentText()
            if ima.have_main:
                ima.ims.set_cmap(ima.hcmap)
                ima.cbr.draw_all()
                self.ax.figure.canvas.draw_idle()

        def open_cbox(self, i):
            icol = ima.lclr[i]
            cdiag = QtWidgets.QColorDialog()
            cdiag.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.Drawer |\
                                 QtCore.Qt.WindowStaysOnTopHint)
            cdiag.setStyleSheet(windowstyle)
            color = cdiag.getColor(initial=QtGui.QColor(icol))
            if color.isValid():
                ncol = color.name()
                ima.lclr[i] = ncol
                cboxs[i].setStyleSheet("background-color: "+ncol)
                if ima.have_main:
                    ima.el1.set_edgecolor(ima.lclr[1])
                    ima.el2.set_edgecolor(ima.lclr[2])
                    ima.xhst.set_color(ima.lclr[3])
                    ima.yhst.set_color(ima.lclr[3])
                    ima.xcen.set_color(ima.lclr[0])
                    ima.ycen.set_color(ima.lclr[0])
                    self.ax.figure.canvas.draw_idle()

        def line_check():
            ima.lvis = txt3.isChecked()
            if ima.have_main:
                ima.set_line_visible()
                self.ax.figure.canvas.draw_idle()

        def cbkgimg():
            self.use_bkgimg = gboxb.isChecked()
            if hasattr(self.storage, 'im0'):
                self.update(2)

        def set_bkgimg(filename):
            if filename != '':
                try:
                    raw = Image.open(filename).convert('F')
                    self.bkgimg = np.array(raw, dtype=np.float32)
                    bkge.setText(filename)
                    self.fp_bkgimg = filename
                    if hasattr(self.storage, 'im0'):
                        self.update(2)
                except FileNotFoundError as e:
                    self.errormsg(str(e))
                except:
                    import traceback
                    self.errormsg(str(traceback.format_exc()))

        def open_bkgimg():
            title = 'Open Image File'
            exts = 'Images (*.png *.jpg *.tif *.tiff)'
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(self.wpref, title, str(self.prev_bkg), exts)
            if filename != '':
                self.prev_bkg = Path(filename).absolute().parent
                set_bkgimg(filename)

        def snap_bkgimg():
            filename = self.take_background()
            set_bkgimg(filename)

        def set_fltr():
            self.ima.fltr = gboxf.isChecked()
            self.ima.fltr_mtd = fmtd.currentText()
            self.ima.fltr_size = int(np.abs(int(cavr.text())))
            if hasattr(self.storage, 'im0'):
                csiz.setText(str(self.ima.dsize))
                self.update(2)

        def set_cbext():
            self.call_extfunc = cbext.isChecked()

        gbox1.toggled.connect(norm)
        nrmv.editingFinished.connect(norm)

        gboxb.toggled.connect(cbkgimg)
        bkgb.clicked.connect(open_bkgimg)
        bkgs.clicked.connect(snap_bkgimg)

        gboxf.toggled.connect(set_fltr)
        fmtd.currentIndexChanged.connect(set_fltr)
        cavr.editingFinished.connect(set_fltr)

        fdb.clicked.connect(ccbd)
        cbext.stateChanged.connect(set_cbext)
        ovrf.currentIndexChanged.connect(covrf)
        cmap.currentIndexChanged.connect(ccmap)
        txt3.stateChanged.connect(line_check)
        for i, cbox in enumerate(cboxs):
            cbox.clicked.connect(lambda s, x=i: open_cbox(self, x))

        self.wpref.setLayout(layout)
        self.wpref.show()

    def take_screenshot(self, head, obj):
        now = datetime.now()
        fld = pathl.home()/'Pictures'
        fname = head+now.strftime('%Y%m%d_%H%M%S')+'.png'
        screen = QtWidgets.QWidget.grab(obj)
        try:
            screen.save(str(fld/fname), 'png')
            self.errormsg('Screenshot saved to:\n'+str(fld)+fname, 'Success')
        except:
            import traceback
            self.errormsg(str(traceback.format_exc()))

    def take_background(self):
        if get_bkg_info is None:
            fld = self.bkg_dir
            optname = '_bkg'
        else:
            fld, optname = get_bkg_info(self.storage.target, self.bkg_dir)

        fname = self.get_keyname(opt = optname)
        if not hasattr(self.storage, 'im0'):
            self.errormsg('No data to use as background')
            return ''
        iout = Image.fromarray(self.storage.im0.astype(np.uint16))
        ret = str(fld/(fname + '.tiff'))
        try:
            iout.save(ret)
            self.errormsg('Background image saved to:\n'+ret, 'Success')
        except:
            import traceback
            self.errormsg(str(traceback.format_exc()))
            ret = ''
        return ret

    def errormsg(self, txt, title='Error'):
        self.live_stop()
        msg = Message(self, txt, title)

    def textline(self, label, unit='', width=[100, 100, 100]):
        layout = QtWidgets.QHBoxLayout()
        txt1 = QtWidgets.QLabel(label, self)
        txt1.setFixedWidth(width[0])
        txt2 = QtWidgets.QLineEdit(self)
        txt2.setFixedWidth(width[1])
        layout.addWidget(txt1)
        layout.addWidget(txt2)
        if unit != '':
            txt3 = QtWidgets.QLabel(unit, self)
            txt3.setFixedWidth(width[2])
            layout.addWidget(txt3)
        layout.addStretch()
        layout.setAlignment(QtCore.Qt.AlignLeft)
        return layout, txt2

    def textline2(self, label, width=[100, 100, 100]):
        layout = QtWidgets.QVBoxLayout()
        txt1 = QtWidgets.QLabel(label, self)
        txt1.setFixedWidth(width[0])

        row = QtWidgets.QHBoxLayout()
        txt2 = QtWidgets.QLineEdit(self)
        txt2.setFixedWidth(width[1])
        txt3 = QtWidgets.QLineEdit(self)
        txt3.setFixedWidth(width[2])

        layout.addWidget(txt1)
        row.addStretch()
        row.addWidget(txt2)
        row.addWidget(QtWidgets.QLabel(', '))
        row.addWidget(txt3)
        row.setAlignment(QtCore.Qt.AlignRight)
        layout.addLayout(row)
        layout.setAlignment(QtCore.Qt.AlignLeft)
        return layout, txt2, txt3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str, required=False, help='Launch with setting file')
    parser.add_argument('--set-dir', type=str, required=False, help='Default setting directory')
    parser.add_argument('--bkg-dir', type=str, required=False, help='Default background image dirctory')
    parser.add_argument('--size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'), required=False, help='Define window size')
    args = parser.parse_args()

    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    app = ApplicationWindow()
    app.show()
    #app.activateWindow()
    app.raise_()
    if args.setting is not None:
        app.loadset(args.setting)
    if args.set_dir is not None:
        ptmp = Path(args.set_dir)
        app.prev_set = ptmp.absolute()
    if args.bkg_dir is not None:
        ptmp = Path(args.bkg_dir)
        app.prev_bkg = ptmp.absolute()
        app.bkg_dir = ptmp.absolute()
    if args.size is not None:
        app.resize(*args.size)

    logging.info('Viola: version {}'.format(__version__))
    qapp.exec_()
