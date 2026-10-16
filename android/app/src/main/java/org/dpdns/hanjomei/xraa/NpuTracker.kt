package org.dpdns.hanjomei.xraa

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.os.SystemClock
import android.util.Log
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.Future
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * Raw TFLite pipeline (MediaPipe-equivalent pre/post processing) so the landmark models can run on the
 * Hexagon NPU through the QNN delegate. Reference implementation: work/npu/pipe/pipe.py.
 *
 * engine: 0 = NPU fp16, 1 = NPU int8, 2 = GPU, 3 = CPU
 * Rects are (cx, cy, size, rot) in upright-frame pixels, always square.
 */
class NpuTracker(private val ctx: Context) : BaseTracker() {
    companion object {
        const val TAG = "XRANpu"
        /** user-facing engines; NPU entries resolve to QNN (Snapdragon) or NNAPI (everything else) at init */
        val ENGINES = arrayOf("NPU Quality", "NPU Balance", "NPU Performance", "GPU", "CPU")
        val BACKENDS_SHORT = arrayOf("QNN", "QNN", "GPU", "CPU", "NNAPI", "NNAPI")
        /** internal backends (Net.backend): 0 QNN fp16, 1 QNN int8, 2 GPU, 3 CPU, 4 NNAPI fp16, 5 NNAPI int8 */
        const val PRESENCE_CHECK_EVERY = 5   // int8: presence heads don't survive quantization, verify with fp16
        const val ACQ = 0.5f        // presence to start tracking
        const val KEEP = 0.3f       // presence to keep tracking (hysteresis: hand over face etc.)
        const val HOLD_MS = 300L    // keep the last result this long through short dropouts

        val BS_SUB = intArrayOf(0,1,4,5,6,7,8,10,13,14,17,21,33,37,39,40,46,52,53,54,55,58,61,63,65,66,67,70,78,80,81,82,84,87,88,91,93,95,103,105,107,109,127,132,133,136,144,145,146,148,149,150,152,153,154,155,157,158,159,160,161,162,163,168,172,173,176,178,181,185,191,195,197,234,246,249,251,263,267,269,270,276,282,283,284,285,288,291,293,295,296,297,300,308,310,311,312,314,317,318,321,323,324,332,334,336,338,356,361,362,365,373,374,375,377,378,379,380,381,382,384,385,386,387,388,389,390,397,398,400,402,405,409,415,454,466,468,469,470,471,472,473,474,475,476,477)
        val BS_NAMES = arrayOf("_neutral","browDownLeft","browDownRight","browInnerUp","browOuterUpLeft","browOuterUpRight","cheekPuff","cheekSquintLeft","cheekSquintRight","eyeBlinkLeft","eyeBlinkRight","eyeLookDownLeft","eyeLookDownRight","eyeLookInLeft","eyeLookInRight","eyeLookOutLeft","eyeLookOutRight","eyeLookUpLeft","eyeLookUpRight","eyeSquintLeft","eyeSquintRight","eyeWideLeft","eyeWideRight","jawForward","jawLeft","jawOpen","jawRight","mouthClose","mouthDimpleLeft","mouthDimpleRight","mouthFrownLeft","mouthFrownRight","mouthFunnel","mouthLeft","mouthLowerDownLeft","mouthLowerDownRight","mouthPressLeft","mouthPressRight","mouthPucker","mouthRight","mouthRollLower","mouthRollUpper","mouthShrugLower","mouthShrugUpper","mouthSmileLeft","mouthSmileRight","mouthStretchLeft","mouthStretchRight","mouthUpperUpLeft","mouthUpperUpRight","noseSneerLeft","noseSneerRight")
        val BS_NAMES_JSON = BS_NAMES.joinToString(",", "[", "]") { "\"" + it + "\"" }
        val HAND_SUB = intArrayOf(0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18)

        fun sig(x: Float) = (1.0 / (1.0 + exp(-x.toDouble()))).toFloat()
        fun normRad(a: Float): Float = (a - 2 * PI * floor((a + PI) / (2 * PI))).toFloat()
    }

    class R4(val cx: Float, val cy: Float, val s: Float, val rot: Float)

    @Volatile var engine = 0; private set
    private val pool = Executors.newFixedThreadPool(3)
    private var nets = ArrayList<Net>()

    private var poseDet: Net? = null
    private var poseLmk: Net? = null
    private var poseChk: Net? = null
    private val handLmk = arrayOfNulls<Net>(2)
    private val handChk = arrayOfNulls<Net>(2)
    private var faceLmk: Net? = null
    private var faceChk: Net? = null
    private var blend: Net? = null

    // per-thread scratch: 0 pose, 1 hand A, 2 hand B, 3 face
    private inner class Scratch(val S: Int) {
        val bmp: Bitmap = Bitmap.createBitmap(S, S, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        val px = IntArray(S * S)
        val m = Matrix(); val inv = Matrix()
    }
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val scDet by lazy { Scratch(224) }
    private val scPose by lazy { Scratch(256) }
    private val scHand = arrayOf(lazy { Scratch(224) }, lazy { Scratch(224) })
    private val scFace by lazy { Scratch(256) }

    // tracking state
    private var poseRect: R4? = null
    private val handRect = arrayOfNulls<R4>(2)
    private val handPresent = BooleanArray(2)
    private val handHanded = FloatArray(2)
    private var faceRect: R4? = null
    private var facePresent = false
    private val handLastJson = arrayOfNulls<String>(2)
    private val handLastOk = LongArray(2)
    /** last result was a hold (presence dropped below KEEP), not a fresh detection */
    private val handHeld = BooleanArray(2)
    /** per-frame motion of each hand's ROI centre (px), for predicting the next ROI */
    private val handVx = FloatArray(2); private val handVy = FloatArray(2)
    private var faceLastJson: String? = null
    private var faceLastOk = 0L
    private var frameNo = 0L
    private val anchors = ssdAnchors()

    // smoothing
    private val fPose = LandmarkFilter(33, 0.05f, 80f, 1f)
    private val fWorld = LandmarkFilter(33, 0.1f, 40f, 1f)
    private val visSmooth = FloatArray(33)
    private var visInit = false

    @Volatile private var handLastMs = 0f
    @Volatile private var lastDet = -1f
    @Volatile private var lastFlag = -1f
    @Volatile private var lastErr = ""
    private var lastHandFlag = FloatArray(2) { -1f }
    @Volatile private var lastFaceFlag = -1f
    @Volatile private var faceLastMs = 0f

    fun init(e: Int) {
        close()
        engine = e
        absorbPendingCrash()
        // QNN only exists on Snapdragon; elsewhere (MediaTek, Exynos, Tensor) the NPU entries go through NNAPI.
        // A tier is (pose, hand, face) backends. Balance = pose fp16 (int8 pose jitters), hands int8 (2 instances,
        // 2x faster, 1.26% err), face fp16 (int8 face was slower on HTP).
        val qcom = isQualcomm()
        val npu: (Int, Int, Int) -> List<IntArray> = { p, h, f ->
            val l = ArrayList<IntArray>()
            if (qcom) l.add(intArrayOf(p, h, f))
            l.add(intArrayOf(p + 4, h + 4, f + 4)); l
        }
        val tiers = ArrayList<IntArray>()
        when (e) {
            0 -> tiers.addAll(npu(0, 0, 0))
            1 -> tiers.addAll(npu(0, 1, 0))
            2 -> tiers.addAll(npu(1, 1, 1))
        }
        if (e <= 3) tiers.add(intArrayOf(2, 2, 2))
        tiers.add(intArrayOf(3, 3, 3))
        var err: Throwable? = null
        for (t in tiers) {
            try {
                build(t[0], t[1], t[2])
                val be = t[0]
                val fb = ctx.getString(R.string.fallback)
                modeLabel = when {
                    e <= 2 && be < 2 -> ENGINES[e] + " · QNN"
                    e <= 2 && be >= 4 -> ENGINES[e] + " · NNAPI" + assign
                    e <= 2 -> BACKENDS_SHORT[be] + fb
                    e == 3 -> if (be == 2) "GPU" else "CPU$fb"
                    else -> "CPU"
                }
                return
            } catch (x: Throwable) {
                Log.w(TAG, "tier ${t.joinToString()} init failed", x); err = x
                close()
            }
        }
        throw err ?: IllegalStateException("no engine")
    }

    private fun isQualcomm(): Boolean {
        val m = if (android.os.Build.VERSION.SDK_INT >= 31) android.os.Build.SOC_MANUFACTURER else android.os.Build.HARDWARE
        Log.i(TAG, "soc: ${if (android.os.Build.VERSION.SDK_INT >= 31) android.os.Build.SOC_MODEL else ""} / $m / ${android.os.Build.HARDWARE}")
        return m.contains("qcom", true) || m.contains("qualcomm", true) || android.os.Build.HARDWARE.contains("qcom", true)
    }

    private fun mk(asset: String, be: Int): Net = Net(ctx, asset, be).also { nets.add(it) }

    // ---- NNAPI crash guard: some vendor drivers SIGSEGV while compiling a model (D8100: face_f on fp16).
    // A native crash can't be caught, so mark the model before compiling; if the marker survives to the
    // next init, that model+backend is blacklisted on this device and the slot falls back.
    private val gp by lazy { ctx.getSharedPreferences("nnapi_guard", Context.MODE_PRIVATE) }
    private fun guardKey(asset: String, be: Int) = "$asset@$be@${android.os.Build.FINGERPRINT}"
    private fun absorbPendingCrash() {
        val p = gp.getString("pending", null) ?: return
        val bad = HashSet(gp.getStringSet("bad", emptySet())!!); bad.add(p)
        gp.edit().putStringSet("bad", bad).remove("pending").commit()
        Log.w(TAG, "nnapi crash during compile last run, blacklisted: $p")
    }
    private fun isBad(asset: String, be: Int) = gp.getStringSet("bad", emptySet())!!.contains(guardKey(asset, be))
    fun resetNnapiBlacklist() { gp.edit().clear().commit() }

    private fun mkGuarded(asset: String, be: Int): Net {
        if (be < 4) return mk(asset, be)
        if (isBad(asset, be)) throw IllegalStateException("blacklisted $asset@$be")
        gp.edit().putString("pending", guardKey(asset, be)).commit()
        try { return mk(asset, be) } finally { gp.edit().remove("pending").commit() }
    }

    private val assign = StringBuilder()
    private fun beTag(be: Int) = when (be) { 0 -> "Q16"; 1 -> "Q8"; 2 -> "G"; 3 -> "C"; 4 -> "N16"; else -> "N8" }

    /** Per-slot chain for NNAPI engines: one model failing only demotes that slot. Returns (net, isInt8). */
    private fun slot(name: String, be: Int): Pair<Net, Boolean> {
        val chain = ArrayList<Pair<String, Int>>()
        if (be == 5) chain.add("m/${name}_i8.tflite" to 5)
        chain.add("m/${name}_f.tflite" to 4); chain.add("m/${name}_f.tflite" to 2); chain.add("m/${name}_f.tflite" to 3)
        var err: Throwable? = null
        for ((a, b) in chain) {
            try { return mkGuarded(a, b).also { Log.i(TAG, "slot $name -> $a @ ${beTag(b)}") } to (b == 5) }
            catch (t: Throwable) { Log.w(TAG, "slot $name: $a @ ${beTag(b)} failed: ${t.message}"); err = t }
        }
        throw err ?: IllegalStateException("slot $name")
    }

    private fun build(pBe: Int, hBe: Int, fBe: Int) {
        assign.setLength(0)
        poseDet = mk("m/pose_det.tflite", 3)   // detector only runs on track loss; CPU is the safe path
        if (pBe >= 4) {
            val tags = ArrayList<String>()
            val (pn, pq) = slot("pose", pBe); poseLmk = pn
            if (pq) poseChk = slot("pose", 4).first
            tags.add("P:" + beTag(pn.backend))
            for (k in 0..1) {
                val (hn, hq) = slot("hand", hBe); handLmk[k] = hn
                if (hq) handChk[k] = slot("hand", 4).first
                if (k == 0) tags.add("H:" + beTag(hn.backend))
            }
            val (fn, fq) = slot("face", fBe); faceLmk = fn
            if (fq) faceChk = slot("face", 4).first
            tags.add("F:" + beTag(fn.backend))
            assign.append(" (").append(tags.joinToString(" ")).append(")")
        } else {
            val sfx = { be: Int -> if (be == 1) "_i8" else "_f" }
            poseLmk = mk("m/pose${sfx(pBe)}.tflite", pBe)
            for (k in 0..1) handLmk[k] = mk("m/hand${sfx(hBe)}.tflite", hBe)
            faceLmk = mk("m/face${sfx(fBe)}.tflite", fBe)
            if (pBe == 1) poseChk = mk("m/pose_f.tflite", 0)
            if (hBe == 1) for (k in 0..1) handChk[k] = mk("m/hand_f.tflite", 0)
            if (fBe == 1) faceChk = mk("m/face_f.tflite", 0)
        }
        blend = mk("m/blend.tflite", 3)
        resetTracking()
    }

    @Volatile var reseeds = 0L
    private fun resetTracking() {
        poseRect = null; faceRect = null; facePresent = false
        handRect.fill(null); handPresent.fill(false); handLastJson.fill(null); handHeld.fill(false)
        handVx.fill(0f); handVy.fill(0f)
        faceLastJson = null
        fPose.reset(); fWorld.reset(); visInit = false
    }

    // ---------------------------------------------------------------- geometry
    private class Frame(val raw: Bitmap, rot: Int) {
        val rw = raw.width; val rh = raw.height
        val w = if (rot == 90 || rot == 270) rh else rw
        val h = if (rot == 90 || rot == 270) rw else rh
        /** upright -> raw */
        val up2raw = Matrix().apply {
            setValues(when (rot) {
                90 -> floatArrayOf(0f, 1f, 0f, -1f, 0f, rh.toFloat(), 0f, 0f, 1f)
                180 -> floatArrayOf(-1f, 0f, rw.toFloat(), 0f, -1f, rh.toFloat(), 0f, 0f, 1f)
                270 -> floatArrayOf(0f, -1f, rw.toFloat(), 1f, 0f, 0f, 0f, 0f, 1f)
                else -> floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
            })
        }
    }

    /** rotated square crop of the upright frame -> sc.px (ARGB, S x S) */
    private fun warp(fr: Frame, r: R4, sc: Scratch) {
        val S = sc.S.toFloat()
        val k = r.s / S
        val m = sc.m
        m.setTranslate(-S / 2f, -S / 2f)
        m.postScale(k, k)
        m.postRotate(Math.toDegrees(r.rot.toDouble()).toFloat())
        m.postTranslate(r.cx, r.cy)
        m.postConcat(fr.up2raw)                    // out -> raw
        m.invert(sc.inv)                           // raw -> out
        sc.canvas.drawColor(Color.BLACK)
        sc.canvas.drawBitmap(fr.raw, sc.inv, paint)
        sc.bmp.getPixels(sc.px, 0, sc.S, 0, 0, sc.S, sc.S)
    }

    /** crop-normalized -> frame pixels, writes into out[o], out[o+1] */
    private fun unproject(lx: Float, ly: Float, r: R4, out: FloatArray, o: Int) {
        val x = (lx - 0.5f) * r.s; val y = (ly - 0.5f) * r.s
        val c = cos(r.rot); val s = sin(r.rot)
        out[o] = r.cx + c * x - s * y; out[o + 1] = r.cy + s * x + c * y
    }

    private fun alignRect(x0: Float, y0: Float, x1: Float, y1: Float, scale: Float, targetRad: Float = (PI / 2).toFloat()): R4 {
        val size = 2 * hypot(x1 - x0, y1 - y0)
        val rot = normRad(targetRad - atan2(-(y1 - y0), x1 - x0))
        return R4(x0, y0, size * scale, rot)
    }

    private fun transform(cx0: Float, cy0: Float, w: Float, h: Float, rot: Float, scale: Float, shiftY: Float = 0f): R4 {
        val cx = cx0 - h * shiftY * sin(rot); val cy = cy0 + h * shiftY * cos(rot)
        return R4(cx, cy, max(w, h) * scale, rot)
    }

    private fun ssdAnchors(): FloatArray {
        val strides = intArrayOf(8, 16, 32, 32, 32)
        val out = ArrayList<Float>()
        var i = 0
        while (i < strides.size) {
            var j = i
            while (j < strides.size && strides[j] == strides[i]) j++
            val n = 2 * (j - i)
            val fm = Math.ceil(224.0 / strides[i]).toInt()
            for (y in 0 until fm) for (x in 0 until fm) for (q in 0 until n) { out.add((x + 0.5f) / fm); out.add((y + 0.5f) / fm) }
            i = j
        }
        return out.toFloatArray()
    }

    // ---------------------------------------------------------------- pose
    private val detRaw = FloatArray(2254 * 12)
    private val detScore = FloatArray(2254)

    private fun detectPose(fr: Frame): R4? {
        val net = poseDet ?: return null
        val L = max(fr.w, fr.h).toFloat()
        val lr = R4(fr.w / 2f, fr.h / 2f, L, 0f)
        warp(fr, lr, scDet)
        net.setPixels(scDet.px, -1f, 1f)
        net.run()
        net.out(0, detRaw); net.out(1, detScore)
        var best = -1; var bs = 0f
        for (i in 0 until 2254) {
            val s = sig(detScore[i].coerceIn(-100f, 100f)); detScore[i] = s
            if (s > bs) { bs = s; best = i }
        }
        lastDet = bs
        if (best < 0 || bs < 0.5f) return null
        val b0 = box(best)
        val acc = FloatArray(12); var ws = 0f
        for (i in 0 until 2254) {
            if (detScore[i] < 0.5f) continue
            val bk = box(i)
            if (iou(b0, bk) > 0.3f) { for (t in 0 until 12) acc[t] += bk[t] * detScore[i]; ws += detScore[i] }
        }
        for (t in 0 until 12) acc[t] /= ws
        val p = FloatArray(4)
        unproject(acc[4], acc[5], lr, p, 0); unproject(acc[6], acc[7], lr, p, 2)
        return alignRect(p[0], p[1], p[2], p[3], 1.25f)
    }

    private fun box(k: Int): FloatArray {
        val ax = anchors[2 * k]; val ay = anchors[2 * k + 1]; val o = k * 12
        val b = FloatArray(12)
        b[0] = detRaw[o] / 224f + ax; b[1] = detRaw[o + 1] / 224f + ay
        b[2] = detRaw[o + 2] / 224f; b[3] = detRaw[o + 3] / 224f
        for (t in 0 until 4) { b[4 + 2 * t] = detRaw[o + 4 + 2 * t] / 224f + ax; b[5 + 2 * t] = detRaw[o + 5 + 2 * t] / 224f + ay }
        return b
    }

    private fun iou(p: FloatArray, q: FloatArray): Float {
        val iw = max(0f, min(p[0] + p[2] / 2, q[0] + q[2] / 2) - max(p[0] - p[2] / 2, q[0] - q[2] / 2))
        val ih = max(0f, min(p[1] + p[3] / 2, q[1] + q[3] / 2) - max(p[1] - p[3] / 2, q[1] - q[3] / 2))
        val inter = iw * ih
        return inter / (p[2] * p[3] + q[2] * q[3] - inter + 1e-9f)
    }

    private val pLm = FloatArray(195)
    private val pHeat = FloatArray(64 * 64 * 39)
    private val pWorld = FloatArray(117)

    /** pose: 33 x (x,y,z,vis,pres) normalized + world 33x3; returns next rect or null */
    private class PoseOut(val lm: FloatArray, val world: FloatArray, val next: R4)

    private fun runPose(fr: Frame, r: R4, check: Boolean, thr: Float = ACQ): PoseOut? {
        val net = poseLmk ?: return null
        warp(fr, r, scPose)
        net.setPixels(scPose.px, 0f, 1f)
        net.run()
        val flag = if (poseChk != null) {
            if (!check) 1f else { val c = poseChk!!; c.setPixels(scPose.px, 0f, 1f); c.run(); c.out0(1) }
        } else net.out0(1)
        lastFlag = flag
        if (flag < thr) return null
        net.out(0, pLm); net.out(3, pHeat); net.out(4, pWorld)
        val W = fr.w.toFloat(); val H = fr.h.toFloat()
        val all = FloatArray(39 * 5)
        val tmp = FloatArray(2)
        for (j in 0 until 39) {
            var nx = pLm[j * 5] / 256f; var ny = pLm[j * 5 + 1] / 256f
            // heatmap refinement (kernel 7)
            val c = (nx * 64).toInt(); val rr = (ny * 64).toInt()
            if (c in 0..63 && rr in 0..63) {
                var sum = 0f; var wc = 0f; var wr = 0f; var mx = 0f
                for (row in max(0, rr - 3) until min(64, rr + 4)) for (col in max(0, c - 3) until min(64, c + 4)) {
                    val v = sig(pHeat[(row * 64 + col) * 39 + j])
                    sum += v; wc += col * v; wr += row * v; if (v > mx) mx = v
                }
                if (mx >= 0.5f && sum > 0) { nx = wc / 64f / sum; ny = wr / 64f / sum }
            }
            unproject(nx, ny, r, tmp, 0)
            all[j * 5] = tmp[0] / W; all[j * 5 + 1] = tmp[1] / H
            all[j * 5 + 2] = pLm[j * 5 + 2] / 256f * r.s / W
            all[j * 5 + 3] = sig(pLm[j * 5 + 3]); all[j * 5 + 4] = sig(pLm[j * 5 + 4])
        }
        val world = FloatArray(99)
        val cr = cos(r.rot); val sr = sin(r.rot)
        for (j in 0 until 33) {
            val x = pWorld[j * 3]; val y = pWorld[j * 3 + 1]
            world[j * 3] = cr * x - sr * y; world[j * 3 + 1] = sr * x + cr * y; world[j * 3 + 2] = pWorld[j * 3 + 2]
        }
        val next = alignRect(all[33 * 5] * W, all[33 * 5 + 1] * H, all[34 * 5] * W, all[34 * 5 + 1] * H, 1.25f)
        return PoseOut(all.copyOf(33 * 5), world, next)
    }

    // ---------------------------------------------------------------- hands
    private fun handRectFromLm(px: FloatArray): R4 {   // px: 21*2 pixels
        val x0 = px[0]; val y0 = px[1]
        val x1 = ((px[5 * 2] + px[13 * 2]) / 2 + px[9 * 2]) / 2
        val y1 = ((px[5 * 2 + 1] + px[13 * 2 + 1]) / 2 + px[9 * 2 + 1]) / 2
        val rot = normRad((PI / 2).toFloat() - atan2(-(y1 - y0), x1 - x0))
        var mnx = Float.MAX_VALUE; var mny = Float.MAX_VALUE; var mxx = -Float.MAX_VALUE; var mxy = -Float.MAX_VALUE
        for (i in HAND_SUB) { mnx = min(mnx, px[i * 2]); mxx = max(mxx, px[i * 2]); mny = min(mny, px[i * 2 + 1]); mxy = max(mxy, px[i * 2 + 1]) }
        val acx = (mnx + mxx) / 2; val acy = (mny + mxy) / 2
        val c = cos(-rot); val s = sin(-rot)
        mnx = Float.MAX_VALUE; mny = Float.MAX_VALUE; mxx = -Float.MAX_VALUE; mxy = -Float.MAX_VALUE
        for (i in HAND_SUB) {
            val qx = px[i * 2] - acx; val qy = px[i * 2 + 1] - acy
            val rx = qx * c - qy * s; val ry = qx * s + qy * c
            mnx = min(mnx, rx); mxx = max(mxx, rx); mny = min(mny, ry); mxy = max(mxy, ry)
        }
        val pcx = (mnx + mxx) / 2; val pcy = (mny + mxy) / 2
        val cx = pcx * cos(rot) - pcy * sin(rot) + acx
        val cy = pcx * sin(rot) + pcy * cos(rot) + acy
        return transform(cx, cy, mxx - mnx, mxy - mny, rot, 2.0f, -0.1f)
    }

    private fun handRectFromPose(pl: FloatArray, W: Float, H: Float, side: Int): R4 {
        val w = if (side == 0) 15 else 16; val pk = w + 2; val ix = w + 4
        val x0 = pl[w * 5] * W; val y0 = pl[w * 5 + 1] * H
        val mx = (pl[pk * 5] + pl[ix * 5]) / 2 * W; val my = (pl[pk * 5 + 1] + pl[ix * 5 + 1]) / 2 * H
        val size = 2 * hypot(mx - x0, my - y0)
        val rot = normRad((PI / 2).toFloat() - atan2(-(my - y0), mx - x0))
        return transform(x0, y0, size, size, rot, 2.7f, -0.1f)
    }

    private fun runHand(fr: Frame, side: Int, r: R4, check: Boolean): String? {
        val t = SystemClock.elapsedRealtimeNanos()
        try {
            val net = handLmk[side] ?: return null
            val sc = scHand[side].value
            warp(fr, r, sc)
            net.setPixels(sc.px, 0f, 1f)
            net.run()
            val chk = handChk[side]
            var pres = 1f
            if (chk != null) {
                if (check || !handPresent[side]) { chk.setPixels(sc.px, 0f, 1f); chk.run(); pres = chk.out0(1); handHanded[side] = chk.out0(2) }
            } else { pres = net.out0(1); handHanded[side] = net.out0(2) }
            if (chk == null || check || !handPresent[side]) lastHandFlag[side] = pres
            val now = SystemClock.uptimeMillis()
            if (pres < (if (handPresent[side]) KEEP else ACQ)) {
                val last = handLastJson[side]
                if (handPresent[side] && last != null && now - handLastOk[side] < HOLD_MS) { handHeld[side] = true; return last }   // short dropout: hold, keep the rect
                handHeld[side] = false
                handPresent[side] = false; handRect[side] = null; handLastJson[side] = null; handVx[side] = 0f; handVy[side] = 0f; return null
            }
            val wasFresh = handPresent[side] && !handHeld[side]
            handPresent[side] = true
            val lm = net.out(0); val world = net.out(3)
            val handed = handHanded[side]
            val W = fr.w.toFloat(); val H = fr.h.toFloat()
            val px = FloatArray(42)
            for (j in 0 until 21) unproject(lm[j * 3] / 224f, lm[j * 3 + 1] / 224f, r, px, j * 2)
            val nr = handRectFromLm(px)
            val prev = handRect[side]
            if (prev != null && wasFresh) {
                // smoothed ROI velocity (only from consecutive fresh detections)
                handVx[side] = 0.5f * handVx[side] + 0.5f * (nr.cx - prev.cx)
                handVy[side] = 0.5f * handVy[side] + 0.5f * (nr.cy - prev.cy)
            } else { handVx[side] = 0f; handVy[side] = 0f }
            handRect[side] = nr
            handHeld[side] = false
            val right = handed > 0.5f
            val label = if (right) "Right" else "Left"
            val score = if (right) handed else 1f - handed
            val cr = cos(r.rot); val sr = sin(r.rot)
            val s = StringBuilder(2600)
            s.append("{\"label\":\"").append(label).append("\",\"score\":"); num(s, score); s.append(",\"lm\":[")
            for (j in 0 until 21) {
                if (j > 0) s.append(',')
                s.append('['); num(s, px[j * 2] / W); s.append(','); num(s, px[j * 2 + 1] / H); s.append(','); num(s, lm[j * 3 + 2] / 224f * r.s / W); s.append(']')
            }
            s.append("],\"world\":[")
            for (j in 0 until 21) {
                if (j > 0) s.append(',')
                val x = world[j * 3]; val y = world[j * 3 + 1]
                s.append('['); num(s, cr * x - sr * y); s.append(','); num(s, sr * x + cr * y); s.append(','); num(s, world[j * 3 + 2]); s.append(']')
            }
            s.append("]}")
            val out = s.toString()
            handLastJson[side] = out; handLastOk[side] = now
            return out
        } finally { handLastMs = (SystemClock.elapsedRealtimeNanos() - t) / 1e6f }
    }

    // ---------------------------------------------------------------- face
    private fun faceRectFromLm(px: FloatArray): R4 {   // 478*2 pixels
        val x0 = px[33 * 2]; val y0 = px[33 * 2 + 1]; val x1 = px[263 * 2]; val y1 = px[263 * 2 + 1]
        val rot = normRad(0f - atan2(-(y1 - y0), x1 - x0))
        var mnx = Float.MAX_VALUE; var mny = Float.MAX_VALUE; var mxx = -Float.MAX_VALUE; var mxy = -Float.MAX_VALUE
        for (i in 0 until 478) { mnx = min(mnx, px[i * 2]); mxx = max(mxx, px[i * 2]); mny = min(mny, px[i * 2 + 1]); mxy = max(mxy, px[i * 2 + 1]) }
        return transform((mnx + mxx) / 2, (mny + mxy) / 2, mxx - mnx, mxy - mny, rot, 1.5f)
    }

    private fun faceRectFromPose(pl: FloatArray, W: Float, H: Float): R4 {
        val lex = pl[2 * 5] * W; val ley = pl[2 * 5 + 1] * H; val rex = pl[5 * 5] * W; val rey = pl[5 * 5 + 1] * H
        val rot = normRad(0f - atan2(-(ley - rey), lex - rex))
        val ear = hypot((pl[7 * 5] - pl[8 * 5]) * W, (pl[7 * 5 + 1] - pl[8 * 5 + 1]) * H)
        val eye = hypot(lex - rex, ley - rey)
        val size = max(ear * 1.6f, eye * 4.0f)
        val cx = (lex + rex) / 2 * 0.6f + pl[0] * W * 0.4f
        val cy = (ley + rey) / 2 * 0.6f + pl[1] * H * 0.4f
        return R4(cx, cy, size, rot)
    }

    private fun runFace(fr: Frame, r: R4, check: Boolean): String? {
        val t = SystemClock.elapsedRealtimeNanos()
        try {
            val net = faceLmk ?: return null
            warp(fr, r, scFace)
            net.setPixels(scFace.px, 0f, 1f)
            net.run()
            val chk = faceChk
            val flag = if (chk != null) {
                if (check || !facePresent) { chk.setPixels(scFace.px, 0f, 1f); chk.run(); sig(chk.out0(1)) } else 1f
            } else sig(net.out0(1))
            lastFaceFlag = flag
            val now = SystemClock.uptimeMillis()
            if (flag < (if (facePresent) KEEP else ACQ)) {
                val last = faceLastJson
                if (facePresent && last != null && now - faceLastOk < HOLD_MS) return last
                facePresent = false; faceRect = null; faceLastJson = null; return null
            }
            facePresent = true
            val lm = net.out(0)
            val W = fr.w.toFloat(); val H = fr.h.toFloat()
            val px = FloatArray(478 * 2)
            for (j in 0 until 478) unproject(lm[j * 3] / 256f, lm[j * 3 + 1] / 256f, r, px, j * 2)
            faceRect = faceRectFromLm(px)
            val s = StringBuilder(16000)
            s.append(",\"face\":{\"lm\":[")
            for (j in 0 until 478) {
                if (j > 0) s.append(',')
                s.append('['); num(s, px[j * 2] / W); s.append(','); num(s, px[j * 2 + 1] / H); s.append(','); num(s, lm[j * 3 + 2] / 256f * r.s / W); s.append(']')
            }
            s.append(']')
            val b = blend
            if (b != null) {
                val inp = FloatArray(146 * 2)
                for (i in 0 until 146) { inp[i * 2] = px[BS_SUB[i] * 2]; inp[i * 2 + 1] = px[BS_SUB[i] * 2 + 1] }
                b.setFloats(inp); b.run()
                val bs = b.out(0)
                s.append(",\"bs\":[")
                for (i in 0 until 52) { if (i > 0) s.append(','); num(s, bs[i]) }
                s.append(']')
            }
            s.append('}')
            s.append(",\"bs_names\":").append(BS_NAMES_JSON)
            val out = s.toString()
            faceLastJson = out; faceLastOk = now
            return out
        } finally { faceLastMs = (SystemClock.elapsedRealtimeNanos() - t) / 1e6f }
    }

    // ---------------------------------------------------------------- main
    override fun process(raw: Bitmap, rot: Int, convStartNs: Long) {
        try { process0(raw, rot, convStartNs) } catch (t: Throwable) {
            lastErr = t.javaClass.simpleName + ": " + (t.message ?: "").take(60)
            Log.e(TAG, "process failed", t)
        }
        diag = ctx.getString(R.string.diag, lastDet, lastFlag, lastHandFlag[0], lastHandFlag[1], lastFaceFlag,
            if (lastErr.isEmpty()) "" else ctx.getString(R.string.diag_err, lastErr)) + " rs$reseeds"
    }

    private fun process0(raw: Bitmap, rot: Int, convStartNs: Long) {
        if (poseLmk == null) return
        val fr = Frame(raw, rot)
        val W = fr.w.toFloat(); val H = fr.h.toFloat()
        frameW = fr.w; frameH = fr.h
        convMs = (SystemClock.elapsedRealtimeNanos() - convStartNs) / 1e6f
        frameNo++
        val check = frameNo % PRESENCE_CHECK_EVERY == 0L
        val tsMs = SystemClock.uptimeMillis()

        // ---- hands / face that are already tracked start now, on their own last-frame ROI,
        //      in parallel with pose (MediaPipe tracking style). New ones are seeded from pose below.
        val faceWanted = SystemClock.uptimeMillis() < faceWantedUntil
        val handJobs = arrayOfNulls<Future<String?>>(2)
        val handTracked = BooleanArray(2)
        for (side in 0..1) {
            val r0 = handRect[side]
            if (handPresent[side] && r0 != null) {
                // constant-velocity prediction; grow the ROI with speed so a fast hand stays inside it
                val sp = hypot(handVx[side], handVy[side])
                val r = R4(r0.cx + handVx[side], r0.cy + handVy[side], r0.s * (1f + min(sp / r0.s, 0.5f)), r0.rot)
                handTracked[side] = true
                handJobs[side] = pool.submit(Callable { runHand(fr, side, r, check) })
            }
        }
        var faceJob: Future<String?>? = null
        val fr0 = faceRect
        if (faceWanted && facePresent && fr0 != null) faceJob = pool.submit(Callable { runFace(fr, fr0, check) })

        // ---- pose
        var t0 = SystemClock.elapsedRealtimeNanos()
        var po: PoseOut? = null
        val tracked = poseRect
        if (tracked != null) po = runPose(fr, tracked, check, KEEP)
        if (po == null) {
            val d = detectPose(fr)
            if (d != null) po = runPose(fr, d, true)
        }
        poseRect = po?.next
        poseMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6f

        val sb = StringBuilder(26000)
        sb.append("{\"ok\":1,\"ts\":").append(tsMs).append(",\"w\":").append(fr.w).append(",\"h\":").append(fr.h)
        if (po == null) {
            t0 = SystemClock.elapsedRealtimeNanos()
            for (j in handJobs) runCatching { j?.get() }       // never leave a job reading the frame
            runCatching { faceJob?.get() }
            waitMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6f
            resetTracking()
            sb.append('}')
            latestJson = sb.toString()
            handsMs = 0f; faceMs = 0f
            totalMs = (SystemClock.elapsedRealtimeNanos() - convStartNs) / 1e6f
            tickFps(); return
        }

        // ---- smoothing (MediaPipe pose landmarker equivalent)
        val lm = po.lm
        val scale = objectScale(lm, W, H, 5)
        fPose.apply(lm, 5, tsMs, scale, W, H)
        fWorld.apply(po.world, 3, tsMs, objectScaleWorld(po.world), 1f, 1f)
        for (j in 0 until 33) {
            val v = lm[j * 5 + 3]
            visSmooth[j] = if (!visInit) v else 0.1f * v + 0.9f * visSmooth[j]
            lm[j * 5 + 3] = visSmooth[j]
        }
        visInit = true

        // ---- seed hands / face that aren't tracked yet from the current pose
        t0 = SystemClock.elapsedRealtimeNanos()
        val handFromPose = arrayOfNulls<R4>(2)
        for (side in 0..1) {
            val wi = if (side == 0) 15 else 16
            handFromPose[side] = handRectFromPose(lm, W, H, side)
            if (handJobs[side] != null || lm[wi * 5 + 3] < 0.3f) continue
            val r = handFromPose[side]!!
            handJobs[side] = pool.submit(Callable { runHand(fr, side, r, check) })
        }
        if (faceWanted && faceJob == null && lm[3] > 0.3f) {
            val r = faceRectFromPose(lm, W, H)
            faceJob = pool.submit(Callable { runFace(fr, r, check) })
        } else if (!faceWanted) { faceRect = null; facePresent = false; faceLastJson = null }

        val hands = ArrayList<String>(2); val labels = ArrayList<String>(2); val scores = ArrayList<Float>(2)
        var anyHand = false
        for (side in 0..1) {
            val j = handJobs[side] ?: continue
            anyHand = true
            var s0 = try { j.get() } catch (e: Exception) { Log.w(TAG, "hand job", e); null }
            // a tracked hand lost (or only held) on its predicted ROI: retry this frame from the pose wrist
            val wi = if (side == 0) 15 else 16
            if (handTracked[side] && (s0 == null || handHeld[side]) && lm[wi * 5 + 3] >= 0.3f) {
                val pr = handFromPose[side]
                if (pr != null) {
                    val s1 = try { runHand(fr, side, pr, true) } catch (e: Exception) { null }
                    if (s1 != null) { s0 = s1; if (!handHeld[side]) reseeds++ }
                }
            }
            val s = s0 ?: continue
            val label = s.substringAfter("\"label\":\"").substringBefore('"')
            val score = s.substringAfter("\"score\":").substringBefore(',').toFloatOrNull() ?: 0f
            val dup = labels.indexOf(label)
            if (dup >= 0) { if (score > scores[dup]) { hands[dup] = s; scores[dup] = score } }
            else { hands.add(s); labels.add(label); scores.add(score) }
        }
        handsMs = if (anyHand) handLastMs else 0f
        val faceStr = if (faceJob != null) (try { faceJob.get() } catch (e: Exception) { Log.w(TAG, "face job", e); null }) else null
        faceMs = if (faceJob != null) faceLastMs else 0f
        waitMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6f

        // ---- drift guards for self-tracked hands (re-seed from pose next frame)
        for (side in 0..1) {
            val r = handRect[side] ?: continue
            val p = handFromPose[side] ?: continue
            val wi = if (side == 0) 15 else 16
            if (lm[wi * 5 + 3] > 0.5f && hypot(r.cx - p.cx, r.cy - p.cy) > 0.75f * max(r.s, p.s)) { handRect[side] = null; handPresent[side] = false; handLastJson[side] = null }
        }
        val r0 = handRect[0]; val r1 = handRect[1]
        if (r0 != null && r1 != null && hypot(r0.cx - r1.cx, r0.cy - r1.cy) < 0.3f * min(r0.s, r1.s)) {
            // both trackers collapsed onto one hand: drop the one farther from its own wrist
            val d0 = handFromPose[0]?.let { hypot(r0.cx - it.cx, r0.cy - it.cy) } ?: 0f
            val d1 = handFromPose[1]?.let { hypot(r1.cx - it.cx, r1.cy - it.cy) } ?: 0f
            val k = if (d0 > d1) 0 else 1
            handRect[k] = null; handPresent[k] = false; handLastJson[k] = null
        }

        sb.append(",\"pose\":{\"lm\":[")
        for (j in 0 until 33) {
            if (j > 0) sb.append(',')
            sb.append('['); num(sb, lm[j * 5]); sb.append(','); num(sb, lm[j * 5 + 1]); sb.append(','); num(sb, lm[j * 5 + 2]); sb.append(','); num(sb, lm[j * 5 + 3]); sb.append(']')
        }
        sb.append("],\"world\":[")
        val wl = po.world
        for (j in 0 until 33) {
            if (j > 0) sb.append(',')
            sb.append('['); num(sb, wl[j * 3]); sb.append(','); num(sb, wl[j * 3 + 1]); sb.append(','); num(sb, wl[j * 3 + 2]); sb.append(','); num(sb, lm[j * 5 + 3]); sb.append(']')
        }
        sb.append("]}")
        sb.append(",\"hands\":[").append(hands.joinToString(",")).append(']')
        if (faceStr != null) sb.append(faceStr)
        sb.append('}')
        latestJson = sb.toString()
        totalMs = (SystemClock.elapsedRealtimeNanos() - convStartNs) / 1e6f
        tickFps()
    }

    private fun objectScale(lm: FloatArray, W: Float, H: Float, stride: Int): Float {
        var mnx = Float.MAX_VALUE; var mny = Float.MAX_VALUE; var mxx = -Float.MAX_VALUE; var mxy = -Float.MAX_VALUE
        for (j in 0 until 33) { val x = lm[j * stride] * W; val y = lm[j * stride + 1] * H; mnx = min(mnx, x); mxx = max(mxx, x); mny = min(mny, y); mxy = max(mxy, y) }
        return ((mxx - mnx) + (mxy - mny)) / 2f
    }

    private fun objectScaleWorld(w: FloatArray): Float {
        var mnx = Float.MAX_VALUE; var mny = Float.MAX_VALUE; var mxx = -Float.MAX_VALUE; var mxy = -Float.MAX_VALUE
        for (j in 0 until 33) { mnx = min(mnx, w[j * 3]); mxx = max(mxx, w[j * 3]); mny = min(mny, w[j * 3 + 1]); mxy = max(mxy, w[j * 3 + 1]) }
        return ((mxx - mnx) + (mxy - mny)) / 2f
    }

    override fun close() {
        for (n in nets) n.close()
        nets.clear()
        poseDet = null; poseLmk = null; poseChk = null; faceLmk = null; faceChk = null; blend = null
        handLmk.fill(null); handChk.fill(null)
    }
}

/** MediaPipe-style one-euro landmark filter (x, y, z), value scaled by 1/objectScale. */
class LandmarkFilter(private val n: Int, private val minCutoff: Float, private val beta: Float, private val dCutoff: Float) {
    private val x = FloatArray(n * 3); private val dx = FloatArray(n * 3)
    private var lastMs = -1L
    fun reset() { lastMs = -1L }

    private fun alpha(cutoff: Float, freq: Float): Float {
        val te = 1f / freq; val tau = 1f / (2f * PI.toFloat() * cutoff)
        return 1f / (1f + tau / te)
    }

    /** data: n entries of `stride` floats, first 3 are x,y,z (x,y normalized by W,H; filtered in pixels) */
    fun apply(data: FloatArray, stride: Int, tsMs: Long, objectScale: Float, W: Float, H: Float) {
        if (lastMs < 0 || tsMs <= lastMs || tsMs - lastMs > 500) {
            for (j in 0 until n) { x[j * 3] = data[j * stride] * W; x[j * 3 + 1] = data[j * stride + 1] * H; x[j * 3 + 2] = data[j * stride + 2] * W }
            dx.fill(0f); lastMs = tsMs; return
        }
        val freq = 1000f / (tsMs - lastMs); lastMs = tsMs
        val vs = if (objectScale > 0) 1f / objectScale else 1f
        val ad = alpha(dCutoff, freq)
        for (j in 0 until n) for (c in 0..2) {
            val mul = if (c == 1) H else W
            val v = data[j * stride + c] * mul
            val k = j * 3 + c
            val d = (v - x[k]) * vs * freq
            dx[k] = ad * d + (1 - ad) * dx[k]
            val cutoff = minCutoff + beta * abs(dx[k])
            val a = alpha(cutoff, freq)
            x[k] = a * v + (1 - a) * x[k]
            data[j * stride + c] = x[k] / mul
        }
    }
}
