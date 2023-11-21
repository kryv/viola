import numpy as np
import cv2
import uuid
import json
from collections import OrderedDict

def get_cov(impr, xxi, xx2i, mask=None):
    if mask is not None:
        imp = impr*mask
    else:
        imp = impr
    xx, yy, xy = xxi
    xx2, yy2 = xx2i

    tot = np.sum(imp)
    if tot > 0:
        xmean = np.sum(xx*imp)/tot
        ymean = np.sum(yy*imp)/tot

        xycov = np.sum(xy*imp)/tot - xmean*ymean
        xxcov = np.sum(xx2*imp)/tot - xmean*xmean
        yycov = np.sum(yy2*imp)/tot - ymean*ymean
        cen = np.array([xmean, ymean])
        cov = np.array([[xxcov, xycov],[xycov, yycov]])
    else:
        cen = np.array([0.0, 0.0])
        cov = np.array([[0.0, 0.0], [0.0, 0.0]])

    return cen, cov, tot

def gen_mask(mask, cen, cov, nsig, xi, yi, xxi, xx2i):
    xx, yy, xy = xxi
    xx2, yy2 = xx2i
    xmean, ymean = cen
    xxcov = cov[0,0]
    yycov = cov[1,1]
    xycov = cov[0,1]
    ecov = xxcov*yycov-xycov*xycov
    ecov = 1.0 if ecov == 0.0 else ecov
    mask = (xx2 - 2.0*xx*xmean + xmean**2)*(yycov/ecov) \
         + (yy2 - 2.0*yy*ymean + ymean**2)*(xxcov/ecov) \
         - 2.0*(xy - xx*ymean - yy*xmean + xmean*ymean)*(xycov/ecov) < nsig*nsig
    return mask

def bkgr_error(im, mask):
    ys, xs = im.shape
    tot = np.sum((im != 0)*~mask)
    imm = im[~mask].astype(float)

    mean = np.sum(imm)/tot if tot != 0 else 0.0
    std  = np.sum(imm*imm)/tot if tot != 0 else 0.0
    rms = np.sqrt(np.absolute(std-mean**2))
    return mean, rms

def lcnvf(i, f, w):
    """
    i = initial pixel
    f = final pixel
    w = width in mm
    """
    f -= 1
    m = (i+f)/2.0
    return lambda p: (p-m)*(w)/(f-i)

def coodinate_image(xs, ys, xinfo, yinfo):
    # generate coordinate system of the image

    xcnv = lcnvf(xinfo[0], xinfo[1], xinfo[2])
    ycnv = lcnvf(yinfo[0], yinfo[1], yinfo[2])

    xi = xcnv(np.arange(xs))
    yi = ycnv(np.arange(ys))
    xx, yy = np.meshgrid(xi, yi)
    xy = xx*yy
    xx2, yy2 = np.meshgrid(xi*xi, yi*yi)

    x0i = [xi, yi]
    x1i = np.array([xx, yy, xy])
    x2i = np.array([xx2, yy2])

    return x0i, x1i, x2i

def get_stat(cen, cov):
    # get statistic values
    d = OrderedDict()
    d['X centroid'], d['Y centroid'] = cen
    d['X RMS'] = xr = np.sqrt(cov[0,0])
    d['Y RMS'] = yr = np.sqrt(cov[1,1])
    d['XY corelation'] = cov[0,1]/xr/yr if cov[0,1] != 0.0 else cov[0,1]
    return d

def c_cent(pts1):
    def csp(p11, p12, p21, p22):
        p1 = p12 - p11
        p2 = p22 - p21
        b = np.cross(p1, p2)
        if b == 0.0:
            return None
        p3 = p21 - p11
        r = -np.cross(p2, p3)/b
        return p11 + r*p1

    c0 = csp(*pts1[[0, 3, 1, 2]])
    cx = csp(*pts1[[0, 1, 2, 3]])
    cy = csp(*pts1[[0, 2, 1, 3]])
    if cx is None:
        cx = c0 + (pts1[1] - pts1[0])
    if cy is None:
        cy = c0 + (pts1[2] - pts1[0])

    tt = csp(c0, cy, pts1[0], pts1[1])
    bb = csp(c0, cy, pts1[2], pts1[3])
    ll = csp(c0, cx, pts1[0], pts1[2])
    rr = csp(c0, cx, pts1[1], pts1[3])

    return np.array([tt, bb, ll, rr])

def c_mrgr(pts1, mrgn):
    uv = lambda v: (v/np.linalg.norm(v), v)
    gl, vl = uv(pts1[0]-pts1[2])
    gr, vr = uv(pts1[1]-pts1[3])
    gt, vt = uv(pts1[0]-pts1[1])
    gb, vb = uv(pts1[2]-pts1[3])

    slt = np.cross(vl, vt)
    srt = np.cross(vr, vt)
    srb = np.cross(vl, vb)
    slb = np.cross(vr, vb)

    f = np.max([np.abs([slt, srt, srb, slb])])

    nlt = pts1[0] + (-gl*mrgn[0] - gt*mrgn[2])*slt/f
    nrt = pts1[1] + (-gr*mrgn[0] + gt*mrgn[3])*srt/f
    nlb = pts1[2] + ( gl*mrgn[1] - gb*mrgn[2])*slb/f
    nrb = pts1[3] + ( gr*mrgn[1] + gb*mrgn[3])*srb/f
    return np.array([nlt, nrt, nlb, nrb])

def c_pts2(pts1, mrgn, aspect):
    w = np.max([np.abs(pts1[0, 0]-pts1[1, 0]), np.abs(pts1[0, 0]-pts1[2, 0]),
                np.abs(pts1[0, 0]-pts1[3, 0]), np.abs(pts1[1, 0]-pts1[2, 0]),
                np.abs(pts1[1, 0]-pts1[3, 0]), np.abs(pts1[2, 0]-pts1[3, 0])])
    h = np.max([np.abs(pts1[0, 1]-pts1[2, 1]), np.abs(pts1[0, 1]-pts1[2, 1]),
                np.abs(pts1[0, 1]-pts1[3, 1]), np.abs(pts1[1, 1]-pts1[2, 1]),
                np.abs(pts1[1, 1]-pts1[3, 1]), np.abs(pts1[2, 1]-pts1[3, 1])])
    if w*aspect > h:
        h = w*aspect
    else:
        w = h/aspect

    msize = 100
    if w+mrgn[2]+mrgn[3] < msize:
        w = msize-mrgn[2]-mrgn[3]
        h = w*aspect
    if h+mrgn[0]+mrgn[1] < msize:
        h = msize-mrgn[0]-mrgn[1]
        w = h/aspect

    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]]) + np.float32([mrgn[2], mrgn[0]])
    mat = cv2.getPerspectiveTransform(pts1, pts2)
    dsize = (int(w+mrgn[2]+mrgn[3]), int(h+mrgn[0]+mrgn[1]))
    return mat, dsize

def get_ellipse(cen, cov):
    # calculate eigenequation to transform the ellipse
    v, w = np.linalg.eigh(cov)
    u = w[0]/np.linalg.norm(w[0])
    ang = np.arctan2(u[1], u[0])*180.0/np.pi
    v[v<0] = 0.0
    v = 2.0*np.sqrt(v)
    return cen, v, ang

class NoIndent(object):
    def __init__(self, value):
        self.value = value

class NoIndentEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super(NoIndentEncoder, self).__init__(*args, **kwargs)
        self.kwargs = dict(kwargs)
        del self.kwargs['indent']
        self._replacement_map = {}

    def default(self, o):
        if isinstance(o, NoIndent):
            key = uuid.uuid4().hex
            self._replacement_map[key] = json.dumps(o.value, **self.kwargs)
            return "@@%s@@" % (key,)
        else:
            return super(NoIndentEncoder, self).default(o)

    def encode(self, o):
        result = super(NoIndentEncoder, self).encode(o)
        for k, v in iter(self._replacement_map.items()):
            result = result.replace('"@@%s@@"' % (k,), v)
        return result
