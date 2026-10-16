import os
"""Reference implementation of the raw-TFLite pipeline that v0.10 ports to Kotlin.
Rects are (cx, cy, size, rot) in upright-frame pixels, always square.
"""
import math, numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

M = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../') + '/models/'

def sig(x): return 1.0 / (1.0 + np.exp(-x))
def normrad(a): return a - 2 * math.pi * math.floor((a + math.pi) / (2 * math.pi))

class Net:
    def __init__(self, path):
        self.it = Interpreter(model_path=path); self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]['index']
        self.out = [d['index'] for d in self.it.get_output_details()]
    def __call__(self, x):
        self.it.set_tensor(self.inp, x[None].astype(np.float32)); self.it.invoke()
        return [self.it.get_tensor(i).reshape(-1) for i in self.out]

def warp(img, rect, S):
    """img: HxWx3 float array (0..1). Rotated square crop -> SxS."""
    cx, cy, s, r = rect
    c, sn = math.cos(r), math.sin(r)
    k = s / S
    # output (u,v) -> input: [cx,cy] + R*((u+.5)*k - s/2, (v+.5)*k - s/2)
    a = k * c; b = -k * sn; d = k * sn; e = k * c
    ox = cx + c * (0.5 * k - s / 2) - sn * (0.5 * k - s / 2)
    oy = cy + sn * (0.5 * k - s / 2) + c * (0.5 * k - s / 2)
    pil = Image.fromarray((img * 255).astype(np.uint8))
    out = pil.transform((S, S), Image.AFFINE, (a, b, ox - 0.5, d, e, oy - 0.5), resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    return np.asarray(out, np.float32) / 255.0

def unproject(lx, ly, rect):
    """crop-normalized -> frame pixels"""
    cx, cy, s, r = rect
    x = (lx - 0.5) * s; y = (ly - 0.5) * s
    c, sn = math.cos(r), math.sin(r)
    return cx + c * x - sn * y, cy + sn * x + c * y

def rot_world(p, r):
    c, sn = math.cos(r), math.sin(r)
    x, y = p[..., 0].copy(), p[..., 1].copy()
    p = p.copy(); p[..., 0] = c * x - sn * y; p[..., 1] = sn * x + c * y
    return p

def align_rect(x0, y0, x1, y1, scale, target_deg=90):
    """AlignmentPointsRects (center=p0, size=2*|p1-p0|) + RectTransformation(scale, square_long)"""
    size = 2 * math.hypot(x1 - x0, y1 - y0)
    rot = normrad(math.radians(target_deg) - math.atan2(-(y1 - y0), x1 - x0))
    return (x0, y0, size * scale, rot)

def transform(cx, cy, w, h, rot, scale, shift_y=0.0):
    cx += -h * shift_y * math.sin(rot); cy += h * shift_y * math.cos(rot)
    return (cx, cy, max(w, h) * scale, rot)

# ------------------------------------------------------------------ pose detector
def ssd_anchors():
    strides = [8, 16, 32, 32, 32]; out = []
    i = 0
    while i < len(strides):
        j = i
        while j < len(strides) and strides[j] == strides[i]: j += 1
        n = 2 * (j - i)
        fm = math.ceil(224 / strides[i])
        for y in range(fm):
            for x in range(fm):
                for _ in range(n): out.append(((x + 0.5) / fm, (y + 0.5) / fm))
        i = j
    return np.array(out, np.float32)
ANCH = ssd_anchors(); assert len(ANCH) == 2254

class PoseDet:
    def __init__(self): self.net = Net(M + 'pose_landmarker_full/pose_detector.tflite')
    def __call__(self, img):
        H, W = img.shape[:2]; L = max(H, W)
        rect = (W / 2, H / 2, L, 0.0)          # letterbox == square crop around center w/ zero border
        x = warp(img, rect, 224) * 2 - 1
        raw, score = self.net(x)
        raw = raw.reshape(2254, 12); sc = sig(np.clip(score, -100, 100))
        i = int(np.argmax(sc))
        if sc[i] < 0.5: return None, float(sc[i])
        # weighted NMS around best
        def box(k):
            a = ANCH[k]; r = raw[k] / 224
            return np.array([r[0] + a[0], r[1] + a[1], r[2], r[3]] + [v for t in range(4) for v in (r[4 + 2 * t] + a[0], r[5 + 2 * t] + a[1])])
        b0 = box(i)
        def iou(p, q):
            ax0, ay0, ax1, ay1 = p[0] - p[2] / 2, p[1] - p[3] / 2, p[0] + p[2] / 2, p[1] + p[3] / 2
            bx0, by0, bx1, by1 = q[0] - q[2] / 2, q[1] - q[3] / 2, q[0] + q[2] / 2, q[1] + q[3] / 2
            iw = max(0, min(ax1, bx1) - max(ax0, bx0)); ih = max(0, min(ay1, by1) - max(ay0, by0))
            inter = iw * ih; return inter / (p[2] * p[3] + q[2] * q[3] - inter + 1e-9)
        acc = np.zeros(12); ws = 0
        for k in np.where(sc >= 0.5)[0]:
            bk = box(k)
            if iou(b0, bk) > 0.3: acc += bk * sc[k]; ws += sc[k]
        b = acc / ws
        kp = [unproject(b[4 + 2 * t], b[5 + 2 * t], rect) for t in range(4)]
        return align_rect(kp[0][0], kp[0][1], kp[1][0], kp[1][1], 1.25), float(sc[i])

class PoseLmk:
    def __init__(self, path=None): self.net = Net(path or M + 'pose_landmarker_full/pose_landmarks_detector.tflite')
    def __call__(self, img, rect, refine=True):
        x = warp(img, rect, 256)
        lm, flag, seg, heat, world = self.net(x)
        if flag[0] < 0.5: return None
        lm = lm.reshape(39, 5).copy(); heat = heat.reshape(64, 64, 39)
        nx = lm[:, 0] / 256; ny = lm[:, 1] / 256
        if refine:
            for j in range(39):
                c, r = int(nx[j] * 64), int(ny[j] * 64)
                if not (0 <= c < 64 and 0 <= r < 64): continue
                c0, c1, r0, r1 = max(0, c - 3), min(64, c + 4), max(0, r - 3), min(64, r + 4)
                w = sig(heat[r0:r1, c0:c1, j])
                if w.max() >= 0.5:
                    cs, rs = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
                    nx[j] = (w * cs).sum() / 64 / w.sum(); ny[j] = (w * rs).sum() / 64 / w.sum()
        H, W = img.shape[:2]
        out = np.zeros((39, 5), np.float32)
        for j in range(39):
            px, py = unproject(nx[j], ny[j], rect)
            out[j] = (px / W, py / H, lm[j, 2] / 256 * rect[2] / W, sig(lm[j, 3]), sig(lm[j, 4]))
        wl = rot_world(world.reshape(39, 3)[:33], rect[3])
        nxt = align_rect(out[33, 0] * W, out[33, 1] * H, out[34, 0] * W, out[34, 1] * H, 1.25)
        return out[:33], wl, nxt, float(flag[0])

# ------------------------------------------------------------------ hands
HAND_SUB = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]
def hand_rect_from_lm(px):  # px: 21x2 pixels
    p = px[HAND_SUB]
    x0, y0 = p[0]; x1 = ((p[4, 0] + p[8, 0]) / 2 + p[6, 0]) / 2; y1 = ((p[4, 1] + p[8, 1]) / 2 + p[6, 1]) / 2
    rot = normrad(math.pi / 2 - math.atan2(-(y1 - y0), x1 - x0))
    ac = (p.max(0) + p.min(0)) / 2
    c, s = math.cos(-rot), math.sin(-rot)
    q = p - ac; pr = np.stack([q[:, 0] * c - q[:, 1] * s, q[:, 0] * s + q[:, 1] * c], 1)
    pc = (pr.max(0) + pr.min(0)) / 2; wh = pr.max(0) - pr.min(0)
    cx = pc[0] * math.cos(rot) - pc[1] * math.sin(rot) + ac[0]
    cy = pc[0] * math.sin(rot) + pc[1] * math.cos(rot) + ac[1]
    return transform(cx, cy, wh[0], wh[1], rot, 2.0, -0.1)

def hand_rect_from_pose(pl, W, H, side):
    w, pk, ix = (15, 17, 19) if side == 0 else (16, 18, 20)
    x0, y0 = pl[w, 0] * W, pl[w, 1] * H
    mx = (pl[pk, 0] + pl[ix, 0]) / 2 * W; my = (pl[pk, 1] + pl[ix, 1]) / 2 * H
    # holistic: center wrist, rotation wrist->middle(target 90), size 2*|..| then scale 2.7, shift_y -0.1
    size = 2 * math.hypot(mx - x0, my - y0)
    rot = normrad(math.pi / 2 - math.atan2(-(my - y0), mx - x0))
    return transform(x0, y0, size, size, rot, 2.7, -0.1)

class HandLmk:
    def __init__(self, path=None): self.net = Net(path or M + 'hand_landmarker/hand_landmarks_detector.tflite')
    def __call__(self, img, rect):
        x = warp(img, rect, 224)
        lm, pres, hand, world = self.net(x)
        if pres[0] < 0.5: return None
        H, W = img.shape[:2]
        lm = lm.reshape(21, 3); px = np.array([unproject(a / 224, b / 224, rect) for a, b, _ in lm])
        out = np.c_[px[:, 0] / W, px[:, 1] / H, lm[:, 2] / 224 * rect[2] / W]
        wl = rot_world(world.reshape(21, 3), rect[3])
        return out, wl, float(hand[0]), hand_rect_from_lm(px), float(pres[0])

# ------------------------------------------------------------------ face
BS_SUB = [0,1,4,5,6,7,8,10,13,14,17,21,33,37,39,40,46,52,53,54,55,58,61,63,65,66,67,70,78,80,81,82,84,87,88,91,93,95,103,105,107,109,127,132,133,136,144,145,146,148,149,150,152,153,154,155,157,158,159,160,161,162,163,168,172,173,176,178,181,185,191,195,197,234,246,249,251,263,267,269,270,276,282,283,284,285,288,291,293,295,296,297,300,308,310,311,312,314,317,318,321,323,324,332,334,336,338,356,361,362,365,373,374,375,377,378,379,380,381,382,384,385,386,387,388,389,390,397,398,400,402,405,409,415,454,466,468,469,470,471,472,473,474,475,476,477]

def face_rect_from_lm(px):
    x0, y0 = px[33]; x1, y1 = px[263]
    rot = normrad(0 - math.atan2(-(y1 - y0), x1 - x0))
    mn, mx = px.min(0), px.max(0)  # RectFromLandmarks: axis aligned bbox
    return transform((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, mx[0] - mn[0], mx[1] - mn[1], rot, 1.5)

def face_rect_from_pose(pl, W, H):
    # eyes 2 (left eye) / 5 (right eye), ears 7/8, nose 0
    le = np.array([pl[2, 0] * W, pl[2, 1] * H]); re = np.array([pl[5, 0] * W, pl[5, 1] * H])
    # subject's right eye is image-left in a non-mirrored frame; landmarks 33 (image-left) -> 263
    a, b = (re, le)
    rot = normrad(0 - math.atan2(-(b[1] - a[1]), b[0] - a[0]))
    ear = math.hypot((pl[7, 0] - pl[8, 0]) * W, (pl[7, 1] - pl[8, 1]) * H)
    eye = math.hypot(*(le - re))
    size = max(ear * 1.6, eye * 4.0)
    c = (le + re) / 2 * 0.6 + np.array([pl[0, 0] * W, pl[0, 1] * H]) * 0.4
    return (c[0], c[1], size, rot)

class FaceLmk:
    def __init__(self, path=None):
        self.net = Net(path or M + 'face_landmarker/face_landmarks_detector.tflite')
        self.bs = Net(M + 'face_landmarker/face_blendshapes.tflite')
    def __call__(self, img, rect):
        x = warp(img, rect, 256)
        lm, flag, _ = self.net(x)
        if sig(flag[0]) < 0.5: return None
        H, W = img.shape[:2]
        lm = lm.reshape(478, 3); px = np.array([unproject(a / 256, b / 256, rect) for a, b, _ in lm])
        out = np.c_[px[:, 0] / W, px[:, 1] / H, lm[:, 2] / 256 * rect[2] / W]
        bs = self.bs(px[BS_SUB].astype(np.float32))[0]
        return out, bs, face_rect_from_lm(px), float(sig(flag[0]))
