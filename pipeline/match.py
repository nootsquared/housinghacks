import cv2
import numpy as np

_sift = None


def _features(path, max_side=900):
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=4000)
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    s = max_side / max(im.shape)
    if s < 1:
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    im = cv2.createCLAHE(2.0, (8, 8)).apply(im)
    return _sift.detectAndCompute(im, None)


def inliers(fa, fb, ratio=0.75):
    (ka, da), (kb, db) = fa, fb
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return 0
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [a for a, b in (p for p in m if len(p) == 2) if a.distance < ratio * b.distance]
    if len(good) < 15:
        return len(good) // 3
    pa = np.float32([ka[g.queryIdx].pt for g in good])
    pb = np.float32([kb[g.trainIdx].pt for g in good])
    _, mask = cv2.findFundamentalMat(pa, pb, cv2.USAC_MAGSAC, 1.5, 0.999, 5000)
    return int(mask.sum()) if mask is not None else 0


def feature_cache(paths):
    return {p: _features(p) for p in paths}
