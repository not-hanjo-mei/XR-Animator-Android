package org.dpdns.hanjomei.xraa

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.Future
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Native inference.
 *  - The camera frame is kept in its raw (sensor) orientation; nothing full-size is rotated.
 *  - Pose runs on one rotated+downscaled copy (long side POSE_INPUT).
 *  - Hands / face are cropped from the full-resolution raw frame (rotate+scale only the crop),
 *    and run in parallel worker threads.
 * Output JSON coordinates are normalized to the full upright frame.
 */
class Tracker(private val ctx: Context) : BaseTracker() {
    companion object {
        const val TAG = "XRATracker"
        const val POSE_INPUT = 512
        val MODES = arrayOf("Hybrid", "All GPU", "All CPU")   // pose GPU + hands/face CPU | all GPU | all CPU
    }

    private var pose: PoseLandmarker? = null
    private var handA: HandLandmarker? = null
    private var handB: HandLandmarker? = null
    private var face: FaceLandmarker? = null
    private val pool = Executors.newFixedThreadPool(3)

    @Volatile var mode = 0
        private set
    private var lastTs = 0L
    private var bsNamesJson: String? = null
    @Volatile private var lastPose: List<NormalizedLandmark>? = null
    private var lastFaceCrop: FloatArray? = null
    @Volatile private var handLastMs = 0f
    @Volatile private var faceLastMs = 0f

    fun init(m: Int) {
        close()
        lastPose = null
        lastFaceCrop = null
        mode = m
        val poseD = if (m == 2) Delegate.CPU else Delegate.GPU
        val restD = if (m == 1) Delegate.GPU else Delegate.CPU
        try {
            build(poseD, restD)
            modeLabel = ctx.getString(intArrayOf(R.string.mode0, R.string.mode1, R.string.mode2)[m])
        } catch (e: Throwable) {
            Log.w(TAG, "delegate init failed for mode $m, falling back to CPU", e)
            close()
            build(Delegate.CPU, Delegate.CPU)
            modeLabel = ctx.getString(R.string.mode2) + ctx.getString(R.string.fallback)
        }
    }

    private fun base(path: String, d: Delegate) =
        BaseOptions.builder().setModelAssetPath(path).setDelegate(d).build()

    private fun build(poseD: Delegate, restD: Delegate) {
        pose = PoseLandmarker.createFromOptions(ctx, PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(base("models/pose_landmarker_full.task", poseD))
            .setRunningMode(RunningMode.VIDEO)
            .setNumPoses(1)
            .build())
        val hb = {
            HandLandmarker.createFromOptions(ctx, HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(base("models/hand_landmarker.task", restD))
                .setRunningMode(RunningMode.VIDEO)
                .setNumHands(1)
                .setMinHandDetectionConfidence(0.3f)
                .setMinHandPresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.3f)
                .build())
        }
        handA = hb(); handB = hb()
        face = FaceLandmarker.createFromOptions(ctx, FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(base("models/face_landmarker.task", restD))
            .setRunningMode(RunningMode.VIDEO)
            .setNumFaces(1)
            .setOutputFaceBlendshapes(true)
            .build())
    }

    // ---------------------------------------------------------------- geometry
    /** raw (sensor) frame + clockwise rotation that makes it upright */
    private class Frame(val raw: Bitmap, val rot: Int) {
        val rw = raw.width
        val rh = raw.height
        val w = if (rot == 90 || rot == 270) rh else rw   // upright width
        val h = if (rot == 90 || rot == 270) rw else rh   // upright height

        /** upright square (ux,uy,s) -> raw rect left/top (same size s) */
        fun rawRect(ux: Int, uy: Int, s: Int): IntArray = when (rot) {
            90 -> intArrayOf(uy, rh - ux - s)
            180 -> intArrayOf(rw - ux - s, rh - uy - s)
            270 -> intArrayOf(rw - uy - s, ux)
            else -> intArrayOf(ux, uy)
        }

        /** crop an upright square, returned upright and scaled to target px */
        fun crop(ux: Int, uy: Int, s: Int, target: Int): Bitmap {
            val r = rawRect(ux, uy, s)
            val m = Matrix()
            m.postRotate(rot.toFloat())
            val sc = target.toFloat() / s
            m.postScale(sc, sc)
            return Bitmap.createBitmap(raw, r[0].coerceIn(0, rw - s), r[1].coerceIn(0, rh - s), s, s, m, true)
        }

        fun uprightScaled(longSide: Int): Bitmap {
            val m = Matrix()
            m.postRotate(rot.toFloat())
            val sc = min(1f, longSide.toFloat() / max(rw, rh))
            m.postScale(sc, sc)
            return Bitmap.createBitmap(raw, 0, 0, rw, rh, m, true)
        }
    }

    private data class Crop(val x: Int, val y: Int, val s: Int)

    private fun squareCrop(cx: Float, cy: Float, size: Float, w: Int, h: Int): Crop? {
        var s = size.roundToInt().coerceAtLeast(48)
        s = min(s, min(w, h))
        if (s < 16) return null
        val x = (cx - s / 2f).roundToInt().coerceIn(0, w - s)
        val y = (cy - s / 2f).roundToInt().coerceIn(0, h - s)
        return Crop(x, y, s)
    }

    // ---------------------------------------------------------------- main
    override fun process(raw: Bitmap, rot: Int, convStartNs: Long) {
        val p = pose ?: return
        val tStart = convStartNs
        var ts = SystemClock.uptimeMillis()
        if (ts <= lastTs) ts = lastTs + 1
        lastTs = ts

        val fr = Frame(raw, rot)
        val w = fr.w; val h = fr.h
        frameW = w; frameH = h

        // ---- hands/face crops come from the PREVIOUS pose, so they run concurrently with pose
        val prev = lastPose
        val handJobs = ArrayList<Future<String?>>(2)
        var faceJob: Future<String?>? = null
        val jobsT0 = SystemClock.elapsedRealtimeNanos()
        if (prev != null) {
            val sw = hypot((prev[11].x() - prev[12].x()) * w, (prev[11].y() - prev[12].y()) * h)
            val radius = sw * (1 + max(sw / max(w, h) * 5 - 0.5f, 0.1f)) / 2f * 1.4f
            for ((k, ids) in listOf(intArrayOf(15, 17, 19), intArrayOf(16, 18, 20)).withIndex()) {
                val wr = prev[ids[0]]
                if (vis(wr) < 0.3f) continue
                val cx = (wr.x() + prev[ids[1]].x() + prev[ids[2]].x()) / 3f * w
                val cy = (wr.y() + prev[ids[1]].y() + prev[ids[2]].y()) / 3f * h
                val c = squareCrop(cx, cy, max(radius * 2.2f, 64f), w, h) ?: continue
                val lmk = (if (k == 0) handA else handB) ?: continue
                handJobs.add(pool.submit(Callable { runHand(fr, c, lmk, ts, w, h) }))
            }
            if (SystemClock.uptimeMillis() < faceWantedUntil && vis(prev[0]) > 0.3f) {
                val earD = hypot((prev[7].x() - prev[8].x()) * w, (prev[7].y() - prev[8].y()) * h)
                val eyeD = hypot((prev[2].x() - prev[5].x()) * w, (prev[2].y() - prev[5].y()) * h)
                val size = max(earD * 2.2f, eyeD * 5.5f)
                // smooth the face crop (EMA) so the face model sees a stable window; snap on large jumps
                var fx = prev[0].x() * w; var fy = prev[0].y() * h; var fs = size
                val lf = lastFaceCrop
                if (lf != null && hypot(fx - lf[0], fy - lf[1]) < lf[2] * 0.25f && kotlin.math.abs(fs - lf[2]) < lf[2] * 0.3f) {
                    fx = lf[0] + (fx - lf[0]) * 0.35f; fy = lf[1] + (fy - lf[1]) * 0.35f; fs = lf[2] + (fs - lf[2]) * 0.2f
                }
                lastFaceCrop = floatArrayOf(fx, fy, fs)
                val c = squareCrop(fx, fy, fs, w, h)
                val fl = face
                if (c != null && fl != null) faceJob = pool.submit(Callable { runFace(fr, c, fl, ts, w, h) })
            }
        }

        var t0 = SystemClock.elapsedRealtimeNanos()
        val small = fr.uprightScaled(POSE_INPUT)
        convMs = (SystemClock.elapsedRealtimeNanos() - tStart) / 1e6f

        t0 = SystemClock.elapsedRealtimeNanos()
        val pr = p.detectForVideo(BitmapImageBuilder(small).build(), ts)
        poseMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6f
        small.recycle()

        val sb = StringBuilder(26000)
        sb.append("{\"ok\":1,\"ts\":").append(ts).append(",\"w\":").append(w).append(",\"h\":").append(h)

        val plm = pr.landmarks().firstOrNull()
        val pwl = pr.worldLandmarks().firstOrNull()
        lastPose = plm

        // collect jobs (always, so the raw bitmap is no longer in use when we return)
        val hands = ArrayList<String>(2); val labels = ArrayList<String>(2); val scores = ArrayList<Float>(2)
        for (j in handJobs) {
            val s = try { j.get() } catch (e: Exception) { Log.w(TAG, "hand job", e); null } ?: continue
            val label = s.substringAfter("\"label\":\"").substringBefore('"')
            val score = s.substringAfter("\"score\":").substringBefore(',').toFloatOrNull() ?: 0f
            val dup = labels.indexOf(label)
            if (dup >= 0) {
                if (score > scores[dup]) { hands[dup] = s; scores[dup] = score }
            } else { hands.add(s); labels.add(label); scores.add(score) }
        }
        handsMs = if (handJobs.isEmpty()) 0f else handLastMs
        val faceStr = if (faceJob != null) (try { faceJob.get() } catch (e: Exception) { Log.w(TAG, "face job", e); null }) else null
        faceMs = if (faceJob != null) faceLastMs else 0f
        waitMs = (SystemClock.elapsedRealtimeNanos() - jobsT0) / 1e6f

        if (plm != null && pwl != null) {
            sb.append(",\"pose\":{\"lm\":[")
            plm.forEachIndexed { i, q ->
                if (i > 0) sb.append(',')
                sb.append('['); num(sb, q.x()); sb.append(','); num(sb, q.y()); sb.append(','); num(sb, q.z()); sb.append(','); num(sb, vis(q)); sb.append(']')
            }
            sb.append("],\"world\":[")
            pwl.forEachIndexed { i, q ->
                if (i > 0) sb.append(',')
                sb.append('['); num(sb, q.x()); sb.append(','); num(sb, q.y()); sb.append(','); num(sb, q.z()); sb.append(','); num(sb, q.visibility().orElse(1f)); sb.append(']')
            }
            sb.append("]}")
            sb.append(",\"hands\":[").append(hands.joinToString(",")).append(']')
            if (faceStr != null) sb.append(faceStr)
        }
        sb.append('}')
        latestJson = sb.toString()
        totalMs = (SystemClock.elapsedRealtimeNanos() - tStart) / 1e6f

        tickFps()
    }

    private fun runHand(fr: Frame, c: Crop, lmk: HandLandmarker, ts: Long, w: Int, h: Int): String? {
        val t = SystemClock.elapsedRealtimeNanos()
        try { return runHand0(fr, c, lmk, ts, w, h) } finally { handLastMs = (SystemClock.elapsedRealtimeNanos() - t) / 1e6f }
    }

    private fun runHand0(fr: Frame, c: Crop, lmk: HandLandmarker, ts: Long, w: Int, h: Int): String? {
        val bmp = fr.crop(c.x, c.y, c.s, 256)
        val hr = lmk.detectForVideo(BitmapImageBuilder(bmp).build(), ts)
        bmp.recycle()
        val hl = hr.landmarks().firstOrNull() ?: return null
        val hw = hr.worldLandmarks().firstOrNull() ?: return null
        val cat = hr.handedness().firstOrNull()?.firstOrNull()
        val label = cat?.categoryName() ?: "Left"
        val score = cat?.score() ?: 0.5f
        val s = StringBuilder(2600)
        s.append("{\"label\":\"").append(label).append("\",\"score\":"); num(s, score); s.append(",\"lm\":[")
        hl.forEachIndexed { i, p ->
            if (i > 0) s.append(',')
            s.append('['); num(s, (p.x() * c.s + c.x) / w); s.append(','); num(s, (p.y() * c.s + c.y) / h); s.append(','); num(s, p.z() * c.s / w); s.append(']')
        }
        s.append("],\"world\":[")
        hw.forEachIndexed { i, p ->
            if (i > 0) s.append(',')
            s.append('['); num(s, p.x()); s.append(','); num(s, p.y()); s.append(','); num(s, p.z()); s.append(']')
        }
        s.append("]}")
        return s.toString()
    }

    private fun runFace(fr: Frame, c: Crop, fl: FaceLandmarker, ts: Long, w: Int, h: Int): String? {
        val t = SystemClock.elapsedRealtimeNanos()
        try { return runFace0(fr, c, fl, ts, w, h) } finally { faceLastMs = (SystemClock.elapsedRealtimeNanos() - t) / 1e6f }
    }

    private fun runFace0(fr: Frame, c: Crop, fl: FaceLandmarker, ts: Long, w: Int, h: Int): String? {
        val bmp = fr.crop(c.x, c.y, c.s, 256)
        val res = fl.detectForVideo(BitmapImageBuilder(bmp).build(), ts)
        bmp.recycle()
        val lm = res.faceLandmarks().firstOrNull() ?: return null
        val s = StringBuilder(16000)
        s.append(",\"face\":{\"lm\":[")
        lm.forEachIndexed { i, p ->
            if (i > 0) s.append(',')
            s.append('['); num(s, (p.x() * c.s + c.x) / w); s.append(','); num(s, (p.y() * c.s + c.y) / h); s.append(','); num(s, p.z() * c.s / w); s.append(']')
        }
        s.append(']')
        val bs = res.faceBlendshapes().orElse(null)?.firstOrNull()
        if (bs != null) {
            s.append(",\"bs\":[")
            bs.forEachIndexed { i, cat -> if (i > 0) s.append(','); num(s, cat.score()) }
            s.append(']')
            if (bsNamesJson == null) bsNamesJson = bs.joinToString(",", "[", "]") { "\"" + it.categoryName() + "\"" }
        }
        s.append('}')
        bsNamesJson?.let { s.append(",\"bs_names\":").append(it) }
        return s.toString()
    }

    private fun vis(p: NormalizedLandmark) = p.visibility().orElse(1f)

    override fun close() {
        runCatching { pose?.close() }; runCatching { handA?.close() }
        runCatching { handB?.close() }; runCatching { face?.close() }
        pose = null; handA = null; handB = null; face = null
    }
}
