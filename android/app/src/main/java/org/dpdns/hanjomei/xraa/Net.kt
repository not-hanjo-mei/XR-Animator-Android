package org.dpdns.hanjomei.xraa

import android.content.Context
import android.util.Log
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

/**
 * One TFLite model with a fixed backend. Handles float and int8 I/O.
 * backend: 0 = QNN HTP fp16, 1 = QNN HTP quantized (int8 model), 2 = GPU, 3 = CPU,
 *          4 = NNAPI fp16-relaxed (vendor NPU, e.g. MediaTek APU via Neuron), 5 = NNAPI int8
 */
class Net(ctx: Context, asset: String, val backend: Int, threads: Int = 2) : AutoCloseable {
    private val delegate: Delegate?
    private val interp: Interpreter
    val inBuf: ByteBuffer
    val inQuant: Boolean
    val inScale: Float
    val inZp: Int
    val inH: Int
    val inW: Int
    private val outBufs: Array<ByteBuffer>
    private val outQuant: BooleanArray
    private val outScale: FloatArray
    private val outZp: IntArray
    private val outMap = HashMap<Int, Any>()
    val outSize: IntArray

    init {
        val opts = Interpreter.Options()
        delegate = when (backend) {
            0, 1 -> {
                val o = QnnDelegate.Options()
                o.setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
                o.setSkelLibraryDir(ctx.applicationInfo.nativeLibraryDir)
                o.setHtpPerformanceMode(QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_SUSTAINED_HIGH_PERFORMANCE)
                o.setHtpPrecision(if (backend == 0) QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16
                    else QnnDelegate.Options.HtpPrecision.HTP_PRECISION_QUANTIZED)
                QnnDelegate(o)
            }
            2 -> GpuDelegate(GpuDelegate.Options().setPrecisionLossAllowed(true)
                .setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED))
            4, 5 -> org.tensorflow.lite.nnapi.NnApiDelegate(org.tensorflow.lite.nnapi.NnApiDelegate.Options()
                .setAllowFp16(backend == 4)
                .setUseNnapiCpu(false)
                .setExecutionPreference(org.tensorflow.lite.nnapi.NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED)
                .setCacheDir(ctx.codeCacheDir.absolutePath)
                .setModelToken(("xra_" + asset.replace('/', '_') + "_" + backend + "_" + ctx.packageManager.getPackageInfo(ctx.packageName, 0).lastUpdateTime)))
            else -> null
        }
        delegate?.let { opts.addDelegate(it) }
        if (delegate == null) opts.setNumThreads(threads)
        interp = Interpreter(load(ctx, asset), opts)
        val it = interp.getInputTensor(0)
        val sh = it.shape()
        inH = sh[1]; inW = sh[2]
        inQuant = it.dataType() == DataType.INT8
        inScale = it.quantizationParams().scale
        inZp = it.quantizationParams().zeroPoint
        inBuf = ByteBuffer.allocateDirect(it.numBytes()).order(ByteOrder.nativeOrder())
        val n = interp.outputTensorCount
        outBufs = Array(n) { i -> ByteBuffer.allocateDirect(interp.getOutputTensor(i).numBytes()).order(ByteOrder.nativeOrder()) }
        outQuant = BooleanArray(n) { i -> interp.getOutputTensor(i).dataType() == DataType.INT8 }
        outScale = FloatArray(n) { i -> interp.getOutputTensor(i).quantizationParams().scale }
        outZp = IntArray(n) { i -> interp.getOutputTensor(i).quantizationParams().zeroPoint }
        outSize = IntArray(n) { i -> interp.getOutputTensor(i).numElements() }
        for (i in 0 until n) outMap[i] = outBufs[i]
        Log.i("XRANet", "$asset backend=$backend in=${sh.joinToString("x")} q=$inQuant outs=${outSize.joinToString()}")
    }

    private var tmpF: FloatArray? = null
    private var tmpB: ByteArray? = null

    /** ARGB pixels -> input tensor, value range lo..hi */
    fun setPixels(px: IntArray, lo: Float, hi: Float) {
        inBuf.rewind()
        val k = (hi - lo) / 255f
        if (inQuant) {
            val inv = 1f / inScale
            var tb = tmpB
            if (tb == null || tb.size != px.size * 3) { tb = ByteArray(px.size * 3); tmpB = tb }
            // 256-entry LUT: channel value -> quantized byte
            val lut = ByteArray(256) { v -> q(v * k + lo, inv) }
            var j = 0
            for (p in px) { tb[j++] = lut[(p shr 16) and 255]; tb[j++] = lut[(p shr 8) and 255]; tb[j++] = lut[p and 255] }
            inBuf.put(tb)
        } else {
            val fb = inBuf.asFloatBuffer()
            var tmp = tmpF
            if (tmp == null || tmp.size != px.size * 3) { tmp = FloatArray(px.size * 3); tmpF = tmp }
            var j = 0
            for (p in px) {
                tmp[j++] = ((p shr 16) and 255) * k + lo
                tmp[j++] = ((p shr 8) and 255) * k + lo
                tmp[j++] = (p and 255) * k + lo
            }
            fb.put(tmp)
        }
        inBuf.rewind()
    }

    /** raw float input (e.g. blendshape points) */
    fun setFloats(v: FloatArray) {
        inBuf.rewind(); inBuf.asFloatBuffer().put(v); inBuf.rewind()
    }

    private fun q(v: Float, inv: Float): Byte {
        val r = Math.round(v * inv) + inZp
        return (if (r < -128) -128 else if (r > 127) 127 else r).toByte()
    }

    fun run() {
        for (b in outBufs) b.rewind()
        interp.runForMultipleInputsOutputs(arrayOf<Any>(inBuf), outMap)
    }

    fun out(i: Int, dst: FloatArray = FloatArray(outSize[i])): FloatArray {
        val b = outBufs[i]; b.rewind()
        if (outQuant[i]) {
            val s = outScale[i]; val z = outZp[i]
            for (k in 0 until outSize[i]) dst[k] = (b.get().toInt() - z) * s
        } else b.asFloatBuffer().get(dst, 0, outSize[i])
        return dst
    }

    fun out0(i: Int): Float {
        val b = outBufs[i]
        return if (outQuant[i]) (b.get(0).toInt() - outZp[i]) * outScale[i] else b.getFloat(0)
    }

    override fun close() {
        runCatching { interp.close() }
        runCatching { (delegate as? AutoCloseable)?.close() }
    }

    companion object {
        fun load(ctx: Context, asset: String): MappedByteBuffer {
            val fd = ctx.assets.openFd(asset)
            FileInputStream(fd.fileDescriptor).use { s ->
                return s.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }
}
