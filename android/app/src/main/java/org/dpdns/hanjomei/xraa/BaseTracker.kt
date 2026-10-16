package org.dpdns.hanjomei.xraa

import android.graphics.Bitmap

/** Common surface used by MainActivity for both the MediaPipe Tasks tracker and the raw TFLite/QNN tracker. */
abstract class BaseTracker {
    @Volatile var modeLabel = "-"; protected set
    @Volatile var latestJson: String = "{\"ok\":0}"; protected set
    @Volatile var faceWantedUntil = 0L
    @Volatile var frameW = 0; protected set
    @Volatile var frameH = 0; protected set
    @Volatile var fps = 0f; protected set
    @Volatile var convMs = 0f; protected set
    @Volatile var poseMs = 0f; protected set
    @Volatile var handsMs = 0f; protected set
    @Volatile var faceMs = 0f; protected set
    @Volatile var waitMs = 0f; protected set
    @Volatile var totalMs = 0f; protected set
    @Volatile var diag = ""; protected set

    private var fpsCount = 0
    private var fpsT0 = 0L
    protected fun tickFps() {
        val now = android.os.SystemClock.uptimeMillis()
        if (fpsT0 == 0L) fpsT0 = now
        if (++fpsCount >= 15) {
            fps = fpsCount * 1000f / maxOf(1L, now - fpsT0)
            fpsCount = 0; fpsT0 = now
        }
    }

    abstract fun process(raw: Bitmap, rot: Int, convStartNs: Long)
    abstract fun close()

    companion object {
        /** fast fixed-point formatting, 5 decimals */
        fun num(sb: StringBuilder, v: Float) {
            if (v.isNaN() || v.isInfinite()) { sb.append('0'); return }
            var n = Math.round(v.toDouble() * 100000.0)
            if (n < 0) { sb.append('-'); n = -n }
            sb.append(n / 100000)
            val frac = (n % 100000).toInt()
            if (frac != 0) {
                sb.append('.')
                val f = frac.toString()
                for (i in f.length until 5) sb.append('0')
                sb.append(f)
            }
        }
    }
}
