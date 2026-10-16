package org.dpdns.hanjomei.xraa.bench

import android.app.Activity
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Color
import android.os.BatteryManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.util.Log
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import com.qualcomm.qti.QnnDelegate
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.concurrent.thread
import kotlin.math.sqrt

class MainActivity : Activity() {

    // name, landmark stride, landmark count, input size
    data class M(val name: String, val stride: Int, val pts: Int, val size: Int, val usePts: Int)
    private val models = listOf(
        M("pose", 5, 39, 256, 33),
        M("hand", 3, 21, 224, 21),
        M("face", 3, 478, 256, 478),
    )
    // backend id -> (label, model file suffix)
    private val backends = listOf(
        "cpu" to "CPU 4 threads (fp32)",
        "gpu" to "GPU (fp16)",
        "npu16" to "NPU fp16",
        "npu8" to "NPU int8",
        "npua16" to "NPU a16w8",
    )
    private fun fileFor(m: String, b: String) = when (b) {
        "npu8" -> "m/${m}_i8.tflite"
        "npua16" -> "m/${m}_a16.tflite"
        else -> "m/${m}_f.tflite"
    }

    private lateinit var out: TextView
    private lateinit var scroll: ScrollView
    private val ui = Handler(Looper.getMainLooper())
    private val log = StringBuilder()
    @Volatile private var busy = false
    @Volatile private var stopSustain = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val dp = resources.displayMetrics.density
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setBackgroundColor(Color.BLACK) }
        out = TextView(this).apply {
            setTextColor(Color.WHITE); textSize = 12f; typeface = android.graphics.Typeface.MONOSPACE
            setPadding((10 * dp).toInt(), (10 * dp).toInt(), (10 * dp).toInt(), (10 * dp).toInt())
            setTextIsSelectable(true)
        }
        scroll = ScrollView(this).apply { addView(out) }
        root.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
        fun row(vararg items: Pair<String, () -> Unit>) {
            val r = LinearLayout(this)
            for ((t, f) in items) r.addView(Button(this).apply {
                text = t; isAllCaps = false; textSize = 12f; minWidth = 0; minimumWidth = 0
                setPadding(0, 0, 0, 0); setOnClickListener { f() }
            }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            root.addView(r, LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT))
        }
        row("1 Bench all" to { runBench() }, "Copy results" to { copy() }, "Stop" to { stopSustain = true })
        row("Sustain GPU" to { sustain("gpu") }, "Sustain NPU16" to { sustain("npu16") }, "Sustain NPU8" to { sustain("npu8") }, "Sustain a16" to { sustain("npua16") })
        setContentView(root)
        line("XRA NPU Bench ${packageManager.getPackageInfo(packageName, 0).versionName}")
        line("SoC: ${if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE}  Android ${Build.VERSION.RELEASE}")
        try { line("QNN ${QnnDelegate.getVersion().joinToString(".")}  HTP=${QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_FP16)}/${QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_QUANTIZED)}") }
        catch (t: Throwable) { line("QNN query failed: $t") }
        line(status())
        line("Tap \"1 Bench all\" first (about 1-2 min), then pick a backend and tap \"Sustain\" to run 5 min and watch the temperature.")
    }

    private fun line(s: String) {
        Log.i("XRANpu", s)
        ui.post {
            log.append(s).append('\n'); out.text = log
            scroll.post { scroll.fullScroll(ScrollView.FOCUS_DOWN) }
        }
    }

    private fun copy() {
        (getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager).setPrimaryClip(ClipData.newPlainText("xra-npu", log.toString()))
        android.widget.Toast.makeText(this, "Copied", android.widget.Toast.LENGTH_SHORT).show()
    }

    private fun status(): String {
        val i = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val t = (i?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0) / 10f
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        val th = arrayOf("none", "light", "moderate", "severe", "critical", "emergency", "shutdown").getOrElse(pm.currentThermalStatus) { "?" }
        return "battery %.1f°C · thermal %s".format(t, th)
    }

    private fun loadAsset(path: String): MappedByteBuffer {
        val fd = assets.openFd(path)
        FileInputStream(fd.fileDescriptor).channel.use { ch ->
            return ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
    }

    private fun loadInput(m: M): FloatArray {
        val b = assets.open("m/${m.name}.bin").readBytes()
        val fb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(fb.remaining()).also { fb.get(it) }
    }

    inner class Runner(val m: M, val backend: String, sustained: Boolean = false) : AutoCloseable {
        val delegate: Delegate?
        val interp: Interpreter
        val inBuf: ByteBuffer
        val outs = HashMap<Int, Any>()
        val initMs: Double
        init {
            val t0 = System.nanoTime()
            val opts = Interpreter.Options()
            delegate = when (backend) {
                "cpu" -> { opts.setNumThreads(4); null }
                "gpu" -> GpuDelegate(GpuDelegate.Options().setPrecisionLossAllowed(true)
                    .setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED))
                else -> {
                    val o = QnnDelegate.Options()
                    o.setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
                    o.setSkelLibraryDir(applicationInfo.nativeLibraryDir)
                    o.setHtpPerformanceMode(if (sustained) QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_SUSTAINED_HIGH_PERFORMANCE
                        else QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_BURST)
                    o.setHtpPrecision(if (backend == "npu16") QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16
                        else QnnDelegate.Options.HtpPrecision.HTP_PRECISION_QUANTIZED)
                    o.setCacheDir(cacheDir.absolutePath)
                    o.setModelToken("${m.name}_$backend")
                    QnnDelegate(o)
                }
            }
            delegate?.let { opts.addDelegate(it) }
            if (backend != "cpu") opts.setNumThreads(2) // CPU fallback ops, if any
            interp = Interpreter(loadAsset(fileFor(m.name, backend)), opts)
            val it0 = interp.getInputTensor(0)
            inBuf = ByteBuffer.allocateDirect(it0.numBytes()).order(ByteOrder.nativeOrder())
            for (i in 0 until interp.outputTensorCount) {
                outs[i] = ByteBuffer.allocateDirect(interp.getOutputTensor(i).numBytes()).order(ByteOrder.nativeOrder())
            }
            setInput(loadInput(m))
            interp.runForMultipleInputsOutputs(arrayOf(inBuf), outs)
            initMs = (System.nanoTime() - t0) / 1e6
        }
        fun setInput(x: FloatArray) {
            val t = interp.getInputTensor(0); inBuf.rewind()
            when (t.dataType()) {
                DataType.FLOAT32 -> for (v in x) inBuf.putFloat(v)
                DataType.INT8 -> { val q = t.quantizationParams(); for (v in x) inBuf.put(Math.round(v / q.scale + q.zeroPoint).coerceIn(-128, 127).toByte()) }
                DataType.INT16 -> { val q = t.quantizationParams(); for (v in x) inBuf.putShort(Math.round(v / q.scale + q.zeroPoint).coerceIn(-32768, 32767).toShort()) }
                else -> error("input type ${t.dataType()}")
            }
            inBuf.rewind()
        }
        fun run() {
            for (b in outs.values) (b as ByteBuffer).rewind()
            inBuf.rewind()
            interp.runForMultipleInputsOutputs(arrayOf(inBuf), outs)
        }
        fun landmarks(): FloatArray {
            val want = m.stride * m.pts
            for (i in 0 until interp.outputTensorCount) {
                val t = interp.getOutputTensor(i)
                if (t.numElements() != want) continue
                val b = (outs[i] as ByteBuffer).also { it.rewind() }
                val q = t.quantizationParams()
                val r = FloatArray(want)
                for (k in 0 until want) r[k] = when (t.dataType()) {
                    DataType.FLOAT32 -> b.float
                    DataType.INT8 -> (b.get() - q.zeroPoint) * q.scale
                    DataType.INT16 -> (b.short - q.zeroPoint) * q.scale
                    else -> 0f
                }
                return r
            }
            return FloatArray(0)
        }
        override fun close() { interp.close(); (delegate as? AutoCloseable)?.close() }
    }

    private fun err(m: M, a: FloatArray, b: FloatArray): Double {
        if (a.isEmpty() || b.size != a.size) return Double.NaN
        var s = 0.0
        for (p in 0 until m.usePts) {
            val dx = a[p * m.stride] - b[p * m.stride]; val dy = a[p * m.stride + 1] - b[p * m.stride + 1]
            s += sqrt((dx * dx + dy * dy).toDouble())
        }
        return s / m.usePts / m.size * 100
    }

    private fun runBench() {
        if (busy) return; busy = true
        thread(name = "bench") {
            try {
                line("\n== Per-model latency (10 warm-up, 100 timed runs, ms) ==")
                line(status())
                for (m in models) {
                    var ref: FloatArray? = null
                    line("-- ${m.name} (${m.size}x${m.size})")
                    for ((b, label) in backends) {
                        try {
                            Runner(m, b).use { r ->
                                repeat(10) { r.run() }
                                val ts = DoubleArray(100)
                                for (i in ts.indices) { val t0 = System.nanoTime(); r.run(); ts[i] = (System.nanoTime() - t0) / 1e6 }
                                ts.sort()
                                val lm = r.landmarks()
                                if (b == "cpu") ref = lm
                                val e = ref?.let { err(m, it, lm) } ?: Double.NaN
                                line("%-14s avg %5.2f  p50 %5.2f  p90 %5.2f  init %5.0f  err %s".format(
                                    label, ts.average(), ts[50], ts[90], r.initMs,
                                    if (b == "cpu") "ref" else "%.2f%%".format(e)))
                            }
                        } catch (t: Throwable) {
                            Log.e("XRANpu", "fail ${m.name} $b", t)
                            line("%-14s failed: %s".format(label, (t.message ?: t.toString()).take(120)))
                        }
                    }
                }
                line(status())
                line("err = mean landmark offset vs CPU fp32, as a percentage of the crop side length.")
            } finally { busy = false }
        }
    }

    private fun sustain(b: String) {
        if (busy) return; busy = true; stopSustain = false
        thread(name = "sustain") {
            val rs = ArrayList<Runner>()
            try {
                val label = backends.first { it.first == b }.second
                line("\n== Sustained load: $label, pose + 2 hands + face per frame, 30 fps, 5 min (tap \"Stop\" to end early) ==")
                line("start " + status())
                for (m in models) rs += Runner(m, b, sustained = true)
                val pose = rs[0]; val hand = rs[1]; val face = rs[2]
                val start = System.nanoTime(); var lastReport = start
                var frames = 0; var sum = 0.0; var worst = 0.0
                while (!stopSustain && System.nanoTime() - start < 300e9) {
                    val f0 = System.nanoTime()
                    pose.run(); hand.run(); hand.run(); face.run()
                    val ms = (System.nanoTime() - f0) / 1e6
                    frames++; sum += ms; if (ms > worst) worst = ms
                    val now = System.nanoTime()
                    if (now - lastReport > 30e9) {
                        line("%3.0fs  per frame %.1f ms (worst %.1f) · %s".format((now - start) / 1e9, sum / frames, worst, status()))
                        lastReport = now; frames = 0; sum = 0.0; worst = 0.0
                    }
                    val sleep = 33.333 - (System.nanoTime() - f0) / 1e6
                    if (sleep > 0) Thread.sleep(sleep.toLong())
                }
                line("end " + status())
            } catch (t: Throwable) {
                Log.e("XRANpu", "sustain $b", t)
                line("Sustained test failed: ${t.message ?: t}")
            } finally {
                rs.forEach { runCatching { it.close() } }
                busy = false
            }
        }
    }
}
