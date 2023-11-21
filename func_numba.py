import numpy as np
from numba import jit, njit

@njit(fastmath=True)
def get_cov(imp, x1i, x2i, mask=None):
    # calculate centroid and covariance of the image
    xx, yy, xy = x1i
    xx2, yy2 = x2i
    ys, xs = imp.shape
    tot = 0
    xmean = 0.0
    ymean = 0.0
    xycov = 0.0
    xxcov = 0.0
    yycov = 0.0

    for i in range(ys):
        for j in range(xs):
            cnt = imp[i, j] if mask is None else imp[i, j]*mask[i, j]
            if cnt != 0:
                tot += cnt
                xmean += xx[i, j]*cnt
                ymean += yy[i, j]*cnt
                xycov += xy[i, j]*cnt
                xxcov += xx2[i, j]*cnt
                yycov += yy2[i, j]*cnt

    if tot > 0:
        xmean /= tot
        ymean /= tot
        xycov = xycov/tot - xmean*ymean
        xxcov = xxcov/tot - xmean*xmean
        yycov = yycov/tot - ymean*ymean

    cen = np.array([xmean, ymean])
    cov = np.array([[xxcov, xycov],[xycov, yycov]])
    return cen, cov, tot

@njit(fastmath=True)
def gen_mask(mask, cen, cov, nsig, xi, yi, x1i, x2i):
    # generate ellipse mask for the image
    xx, yy, xy = x1i
    xx2, yy2 = x2i
    xmean, ymean = cen
    xxcov = cov[0,0]
    yycov = cov[1,1]
    xycov = cov[0,1]
    ecov = xxcov*yycov-xycov*xycov
    ecov = 1.0 if ecov == 0.0 else ecov
    ys, xs = mask.shape

    xss = np.where(np.abs(xi-xmean) <= np.sqrt(xxcov)*nsig)[0]
    if len(xss) == 0:
        xsi = 0
        xsf = xs
    else:
        xsi = xss.min()
        xsf = xss.max()

    yss = np.where(np.abs(yi-ymean) <= np.sqrt(yycov)*nsig)[0]
    if len(yss) == 0:
        ysi = 0
        ysf = ys
    else:
        ysi = yss.min()
        ysf = yss.max()

    lim = nsig*nsig*ecov

    for i in range(ysi, ysf):
        for j in range(xsi, xsf):
            mask[i, j] = (xx2[i, j] - 2.0*xx[i, j]*xmean + xmean**2)*(yycov) \
                       + (yy2[i, j] - 2.0*yy[i, j]*ymean + ymean**2)*(xxcov) \
                       - 2.0*(xy[i, j] - xx[i, j]*ymean - yy[i, j]*xmean + xmean*ymean)*(xycov) <= lim
    return mask

@njit(fastmath=True)
def bkgr_error(im, mask):
    ys, xs = im.shape
    tot = 0
    mean = 0
    std = 0

    for i in range(ys):
        for j in range(xs):
            cnt = im[i, j]*~mask[i, j]
            if cnt != 0:
                tot += 1
                mean += im[i, j]
                std += im[i, j]**2

    if tot > 0:
        mean /= tot
        std /= tot
    rms = np.sqrt(np.abs(std-mean**2))
    return mean, rms