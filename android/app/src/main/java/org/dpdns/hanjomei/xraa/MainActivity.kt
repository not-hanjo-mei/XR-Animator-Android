package org.dpdns.hanjomei.xraa

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Base64
import android.util.Log
import android.util.Size
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.webkit.*
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.webkit.WebViewCompat
import androidx.webkit.WebViewFeature
import java.io.ByteArrayInputStream
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.HttpURLConnection
import java.net.InetAddress
import java.net.URL
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong

class MainActivity : AppCompatActivity() {
    companion object {
        const val TAG = "XRAA"
        const val START_URL = "https://sao.animetheme.com/XR_Animator.html"
        const val NATIVE_PATH = "/__xra_native__/frame"
    }

    private lateinit var web: WebView
    private lateinit var preview: PreviewView
    private lateinit var stats: TextView
    private val ENGINES by lazy { NpuTracker.ENGINES + getString(R.string.engine_legacy) }
    @Volatile private var tracker: BaseTracker = NpuTracker(this)
    private var engineIdx = 0
    @Volatile private var vmcHost = ""
    @Volatile private var vmcPort = 39539
    private val analysisExec = Executors.newSingleThreadExecutor()
    private val udpExec = Executors.newSingleThreadExecutor()
    private var udp: DatagramSocket? = null
    private val addrCache = ConcurrentHashMap<String, InetAddress>()
    private val vmcPackets = AtomicLong(0)
    private var vmcTarget = "-"
    private val patchCache = ConcurrentHashMap<String, ByteArray>()
    private lateinit var webPack: WebPack
    private val localServer = LocalServer { q -> handleFrameQuery(q) }
    private var lensFacing = CameraSelector.LENS_FACING_FRONT
    private lateinit var btnLens: Button
    @Volatile private var camErr = ""
    private var noFpsLock = false
    private var bindGen = 0
    private var cameraStarted = false
    private var trackerReady = false
    private val ui = Handler(Looper.getMainLooper())
    private val prefs by lazy { getSharedPreferences("xra", MODE_PRIVATE) }
    /** a selectable camera: a camera ID from CameraManager, optionally one physical lens behind a logical multi-camera */
    private data class CamOpt(val id: String, val phys: String?, val facing: Int, val focal: Float, val label: String, val zoom: Float = 1f) {
        val key get() = (if (phys == null) id else "$id/$phys") + (if (zoom < 1f) "@%.2f".format(java.util.Locale.US, zoom) else "")
    }
    private var cams: List<CamOpt> = emptyList()
    private var cam: CamOpt? = null
    private var camSize: Size? = null       // from the HAL's YUV_420_888 output sizes
    @Volatile private var camFpsLimit = 30   // from the HAL's AE ranges, or lower (software-dropped)
    private var lastKeptTs = 0L
    private lateinit var btnCamFps: Button
    @Volatile private var camFrames = 0L
    @Volatile private var capFps = 30
    private val CAPS = intArrayOf(30, 24, 20, 15)
    private var boxW = 0; private var boxH = 0
    @Volatile private var camW = 0
    @Volatile private var camH = 0
    private lateinit var btnRes: Button
    private lateinit var btnMode: Button
    private lateinit var btnVmc: Button
    private var vmcOn = false

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WebView.setWebContentsDebuggingEnabled(true)

        val outer = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setBackgroundColor(Color.BLACK) }
        val root = FrameLayout(this)
        outer.addView(root, LinearLayout.LayoutParams(-1, 0, 1f))
        web = WebView(this)
        root.addView(web, FrameLayout.LayoutParams(-1, -1))

        preview = PreviewView(this).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FIT_CENTER  // show the whole analysed field of view, no crop
            visibility = View.GONE
        }
        val dp = resources.displayMetrics.density
        root.addView(preview, FrameLayout.LayoutParams((120 * dp).toInt(), (160 * dp).toInt(), Gravity.TOP or Gravity.END).apply {
            topMargin = (8 * dp).toInt(); rightMargin = (8 * dp).toInt()
        })

        val panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.argb(140, 0, 0, 0))
        }
        val bar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding((6 * dp).toInt(), (2 * dp).toInt(), (6 * dp).toInt(), (2 * dp).toInt())
        }
        stats = TextView(this).apply { setTextColor(Color.WHITE); textSize = 10f; text = "native: idle" }
        loadCams()
        if (prefs.getInt("engine_v", 1) < 2) {   // pre-0.12.1 layouts: fp16,int8,GPU,CPU,legacy(4) [+NNAPI 4/5, legacy 6 in 0.12.0]
            val m = intArrayOf(0, 2, 3, 4, 5, 2, 5)
            prefs.edit().putInt("engine", m[prefs.getInt("engine", 0).coerceIn(0, 6)]).putInt("engine_v", 2).apply()
        }
        engineIdx = prefs.getInt("engine", 0).coerceIn(0, ENGINES.size - 1)
        btnRes = Button(this).apply { text = resLabel(); textSize = 10f; setOnClickListener { pickRes() } }
        btnMode = Button(this).apply { text = ENGINES[engineIdx] + " ▾"; textSize = 10f; setOnClickListener { pickEngine() } }
        btnLens = Button(this).apply { text = lensLabel(); textSize = 10f; setOnClickListener { pickLens() } }
        btnCamFps = Button(this).apply { text = getString(R.string.camfps_btn, camFpsLimit); textSize = 10f; setOnClickListener { pickCamFps() } }
        val btnPrev = Button(this).apply {
            text = getString(R.string.preview); textSize = 10f
            setOnClickListener { preview.visibility = if (preview.visibility == View.VISIBLE) View.GONE else View.VISIBLE }
        }
        stats.setPadding((8 * dp).toInt(), (4 * dp).toInt(), (8 * dp).toInt(), 0)
        val lp = { w: Float -> LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, w).apply { marginStart = (2 * dp).toInt(); marginEnd = (2 * dp).toInt() } }
        btnVmc = Button(this).apply { text = "VMC"; textSize = 10f; setOnClickListener { vmcDialog() } }
        val btnModel = Button(this).apply { text = getString(R.string.avatar); textSize = 10f }
        btnModel.setOnClickListener {
            web.evaluateJavascript("(function(){try{MMD_SA.hide_3D_avatar=!MMD_SA.hide_3D_avatar;System._browser.update_tray&&System._browser.update_tray();return MMD_SA.hide_3D_avatar?'off':'on'}catch(e){return 'err'}})()") { r ->
                val a = getString(R.string.avatar); btnModel.text = when { r?.contains("off") == true -> "$a✕"; r?.contains("on") == true -> a; else -> "$a?" }
            }
        }
        for (b in listOf(btnVmc, btnRes, btnMode, btnLens, btnCamFps, btnPrev, btnModel)) { b.minWidth = 0; b.minimumWidth = 0; b.setPadding(0, 0, 0, 0); b.isAllCaps = false; b.maxLines = 1 }
        bar.addView(btnVmc, lp(1f)); bar.addView(btnRes, lp(1f)); bar.addView(btnMode, lp(1.4f))
        panel.addView(stats, LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT))
        panel.addView(bar, LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT))
        val bar2 = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding((6 * dp).toInt(), 0, (6 * dp).toInt(), (2 * dp).toInt())
        }
        capFps = prefs.getInt("cap", 30).let { c -> if (CAPS.contains(c)) c else 30 }
        val btnCap = Button(this).apply { text = getString(R.string.cap_btn, capFps); textSize = 10f }
        btnCap.setOnClickListener {
            pick(getString(R.string.cap_title), CAPS.map { "$it fps" }.toTypedArray(), CAPS.indexOf(capFps)) { i ->
                capFps = CAPS[i]; prefs.edit().putInt("cap", capFps).apply()
                btnCap.text = getString(R.string.cap_btn, capFps)
            }
        }
        val btnRender = Button(this).apply { text = getString(R.string.render_on); textSize = 10f }
        btnRender.setOnClickListener {
            web.evaluateJavascript("(function(){try{var b=System._browser;b.skip_rendering=!b.skip_rendering;return b.skip_rendering?'off':'on'}catch(e){return 'err'}})()") { r ->
                btnRender.text = getString(when { r?.contains("off") == true -> R.string.render_off; r?.contains("on") == true -> R.string.render_on; else -> R.string.render_unk })
            }
        }
        for (b in listOf(btnCap, btnRender)) { b.minWidth = 0; b.minimumWidth = 0; b.setPadding(0, 0, 0, 0); b.isAllCaps = false; b.maxLines = 1 }
        bar.addView(btnCap, lp(1.2f))
        bar2.addView(btnLens, lp(1.3f)); bar2.addView(btnCamFps, lp(1f)); bar2.addView(btnPrev, lp(1f)); bar2.addView(btnModel, lp(1f)); bar2.addView(btnRender, lp(1.2f))
        panel.addView(bar2, LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT))
        outer.addView(panel, LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT))
        setContentView(outer)

        webPack = WebPack(applicationContext)
        setupWebView()
        localServer.start()
        webPack.scheduleUpdate()

        analysisExec.execute {
            initEngine(engineIdx)
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
            web.loadUrl(START_URL)
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
        }
        ui.post(statsLoop)
        ui.postDelayed(autoVmc, 5000)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        startCamera()
        web.loadUrl(START_URL)
    }

    private val statsLoop = object : Runnable {
        var lastPk = 0L
        var lastCam = 0L
        override fun run() {
            if (camW > 0 && camH > 0 && (camW != boxW || camH != boxH)) {
                boxW = camW; boxH = camH
                Log.i(TAG, "analysis frame ${camW}x${camH} (upright)")
                val dp = resources.displayMetrics.density
                val lp = preview.layoutParams
                val longPx = (160 * dp).toInt()
                if (camH >= camW) { lp.height = longPx; lp.width = (longPx.toLong() * camW / camH).toInt() }
                else { lp.width = longPx; lp.height = (longPx.toLong() * camH / camW).toInt() }
                preview.layoutParams = lp
            }
            val pk = vmcPackets.get()
            val pps = (pk - lastPk) / 0.5f; lastPk = pk
            val cf = camFrames
            val cfps = (cf - lastCam) / 0.5f; lastCam = cf
            stats.text = getString(R.string.stats,
                cfps, tracker.fps, tracker.modeLabel, tracker.frameW, tracker.frameH,
                tracker.convMs, tracker.poseMs, tracker.handsMs, tracker.faceMs, tracker.waitMs, tracker.totalMs, vmcTarget, pps,
                batteryTemp(), thermalLabel(), (if (tracker.diag.isEmpty()) "" else "\n" + tracker.diag) + (if (camErr.isEmpty()) "" else "\n" + camErr) + (if (webPack.miss.get() + webPack.hitNet.get() == 0) "" else "\n" + webPack.stats()))
            ui.postDelayed(this, 500)
        }
    }

    private fun batteryTemp(): Float {
        val i = registerReceiver(null, android.content.IntentFilter(android.content.Intent.ACTION_BATTERY_CHANGED))
        return (i?.getIntExtra(android.os.BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0) / 10f
    }

    private fun thermalLabel(): String {
        if (android.os.Build.VERSION.SDK_INT < 29) return "-"
        val pm = getSystemService(android.os.PowerManager::class.java) ?: return "-"
        return when (pm.currentThermalStatus) {
            in 0..6 -> getString(intArrayOf(R.string.th0, R.string.th1, R.string.th2, R.string.th3, R.string.th4, R.string.th5, R.string.th6)[pm.currentThermalStatus]); else -> "?"
        }
    }

    // ------------------------------------------------------------------ camera
    private fun startCamera() {
        if (cameraStarted) return
        cameraStarted = true
        bindCamera()
    }

    @androidx.annotation.OptIn(androidx.camera.camera2.interop.ExperimentalCamera2Interop::class)
    private fun lockFps(b: ImageAnalysis.Builder, r: android.util.Range<Int>) {
        try {
            androidx.camera.camera2.interop.Camera2Interop.Extender(b).setCaptureRequestOption(
                android.hardware.camera2.CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, r)
        } catch (e: Throwable) { Log.w(TAG, "fps lock failed", e) }
    }

    private val VMC_APPS = arrayOf("Others", "Warudo", "VNyan", "VNyan(+Z)", "VSeeFace")

    private fun vmcDialog() {
        val dp = resources.displayMetrics.density
        val pad = (16 * dp).toInt()
        val box = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(pad, pad / 2, pad, 0) }
        val host = android.widget.EditText(this).apply {
            hint = getString(R.string.vmc_host_hint)
            setText(prefs.getString("vmc_host", ""))
            inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_URI
        }
        val port = android.widget.EditText(this).apply {
            hint = getString(R.string.vmc_port_hint)
            setText(prefs.getInt("vmc_port", 39539).toString())
            inputType = android.text.InputType.TYPE_CLASS_NUMBER
        }
        val app = android.widget.Spinner(this).apply {
            adapter = android.widget.ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, VMC_APPS)
            setSelection(prefs.getInt("vmc_app", 4).coerceIn(0, VMC_APPS.size - 1))
        }
        box.addView(TextView(this).apply { text = getString(R.string.vmc_host_label) }); box.addView(host)
        box.addView(TextView(this).apply { text = getString(R.string.vmc_port_label) }); box.addView(port)
        box.addView(TextView(this).apply { text = getString(R.string.vmc_app_label) }); box.addView(app)
        android.app.AlertDialog.Builder(this)
            .setTitle(getString(R.string.vmc_title))
            .setView(box)
            .setPositiveButton(getString(R.string.vmc_start)) { _, _ ->
                val h = host.text.toString().trim()
                val p = port.text.toString().trim().toIntOrNull() ?: 39539
                if (h.isEmpty()) { android.widget.Toast.makeText(this, getString(R.string.vmc_need_ip), android.widget.Toast.LENGTH_SHORT).show(); return@setPositiveButton }
                prefs.edit().putString("vmc_host", h).putInt("vmc_port", p).putInt("vmc_app", app.selectedItemPosition).apply()
                setVmc(true, h, p, VMC_APPS[app.selectedItemPosition])
            }
            .setNegativeButton(getString(if (vmcOn) R.string.vmc_stop else R.string.cancel)) { _, _ -> if (vmcOn) setVmc(false, "", 0, "") }
            .show()
    }

    /** apply the saved receiver automatically once the page is ready; nothing is sent when none is saved */
    private val autoVmc = object : Runnable {
        var tries = 0
        override fun run() {
            val h = prefs.getString("vmc_host", "") ?: ""
            if (vmcOn || h.isEmpty() || tries++ > 100) return
            setVmc(true, h, prefs.getInt("vmc_port", 39539), VMC_APPS[prefs.getInt("vmc_app", 4).coerceIn(0, VMC_APPS.size - 1)], quiet = true)
            ui.postDelayed(this, 3000)
        }
    }

    private fun setVmc(on: Boolean, host: String, port: Int, app: String, quiet: Boolean = false) {
        val js = if (on) """
            (function(){try{
              var V=MMD_SA.OSC.VMC; var H=${org.json.JSONObject.quote(host)}, P=$port;
              V.options.plugin.send.host=H; V.options.plugin.send.port=P;
              if (V.plugin && V.plugin.options) { V.plugin.options.host=H; V.plugin.options.port=P; if (V.plugin.options.send) { V.plugin.options.send.host=H; V.plugin.options.send.port=P; } }
              MMD_SA.OSC.app_mode=${org.json.JSONObject.quote(app)};
              V.send_avatar_data=true;
              if (V.sender_enabled) V.sender_enabled=false;
              V.sender_enabled=true;
              return 'ok';
            }catch(e){return 'err:'+e.message}})()
        """ else """
            (function(){try{MMD_SA.OSC.VMC.sender_enabled=false;return 'off'}catch(e){return 'err:'+e.message}})()
        """
        web.evaluateJavascript(js) { r ->
            Log.i(TAG, "setVmc -> $r")
            val ok = r?.contains("ok") == true || r?.contains("off") == true
            if (ok) {
                vmcOn = on
                if (on) { vmcHost = host; vmcPort = port; vmcTarget = "$host:$port" }
                btnVmc.text = if (on) "VMC●" else "VMC"
                if (!on) vmcTarget = "-"
            } else if (!quiet) {
                android.widget.Toast.makeText(this, getString(R.string.vmc_fail, r), android.widget.Toast.LENGTH_LONG).show()
            }
        }
    }

    /** popup single-choice picker (used for any setting with more than three options) */
    private fun pick(title: String, items: Array<String>, cur: Int, onPick: (Int) -> Unit) {
        android.app.AlertDialog.Builder(this)
            .setTitle(title)
            .setSingleChoiceItems(items, cur) { d, i -> d.dismiss(); if (i != cur) onPick(i) }
            .setNegativeButton(getString(R.string.cancel), null)
            .show()
    }

    private fun pickEngine() {
        pick(getString(R.string.engine_title), ENGINES, engineIdx) { i ->
            engineIdx = i
            prefs.edit().putInt("engine", i).apply()
            btnMode.text = ENGINES[i] + "…"
            analysisExec.execute { initEngine(i) }
        }
    }

    /** runs on analysisExec */
    private fun initEngine(i: Int) {
        trackerReady = false
        val old = tracker
        runCatching { old.close() }
        try {
            if (i < NpuTracker.ENGINES.size) {
                val t = (old as? NpuTracker) ?: NpuTracker(this)
                t.init(i); tracker = t
            } else {
                val t = (old as? Tracker) ?: Tracker(this)
                t.init(0); tracker = t
            }
            trackerReady = true
        } catch (e: Throwable) {
            Log.e(TAG, "engine $i init failed, falling back to MediaPipe", e)
            try { val t = Tracker(this); t.init(0); tracker = t; trackerReady = true } catch (e2: Throwable) { Log.e(TAG, "fallback failed", e2) }
        }
        ui.post { btnMode.text = tracker.modeLabel + " ▾" }
    }

    private val cm by lazy { getSystemService(CAMERA_SERVICE) as android.hardware.camera2.CameraManager }
    private fun chars(id: String) = cm.getCameraCharacteristics(id)

    /** every camera ID the framework lists, plus the physical lenses behind each logical multi-camera */
    private fun enumCams(): List<CamOpt> {
        val out = ArrayList<CamOpt>()
        for (id in runCatching { cm.cameraIdList.toList() }.getOrDefault(emptyList())) {
            val ch = runCatching { chars(id) }.getOrNull() ?: continue
            val facing = ch.get(android.hardware.camera2.CameraCharacteristics.LENS_FACING) ?: -1
            fun focal(c: android.hardware.camera2.CameraCharacteristics) =
                c.get(android.hardware.camera2.CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.minOrNull() ?: 0f
            val fl = focal(ch)
            out += CamOpt(id, null, facing, fl, "${facingName(facing)} #$id · %.2fmm".format(fl))
            // no-root ultra-wide: a logical camera whose zoom range goes below 1x switches lenses inside the HAL (API 30+)
            if (Build.VERSION.SDK_INT >= 30) {
                val zr = ch.get(android.hardware.camera2.CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE)
                Log.i(TAG, "cam $id zoomRatioRange=$zr")
                if (zr != null && zr.lower < 0.99f) {
                    val zs = listOf(zr.lower, 0.8f).filter { it >= zr.lower && it < 0.99f }.distinct()
                    for (z in zs) out += CamOpt(id, null, facing, fl * z, "${facingName(facing)} #$id · %.1fx zoom".format(z), z)
                }
            }
            val caps = ch.get(android.hardware.camera2.CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: IntArray(0)
            if (Build.VERSION.SDK_INT >= 28 && caps.contains(android.hardware.camera2.CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA)) {
                for (p in ch.physicalCameraIds) {
                    val pf = runCatching { focal(chars(p)) }.getOrDefault(0f)
                    out += CamOpt(id, p, facing, pf, "${facingName(facing)} #$id/$p · %.2fmm".format(pf))
                }
            }
        }
        Log.i(TAG, "cameras: " + out.joinToString { it.label })
        return out
    }

    private fun facingName(f: Int) = getString(when (f) {
        android.hardware.camera2.CameraCharacteristics.LENS_FACING_FRONT -> R.string.lens_front
        android.hardware.camera2.CameraCharacteristics.LENS_FACING_BACK -> R.string.lens_back
        else -> R.string.lens_ext })

    private fun loadCams() {
        cams = enumCams()
        val k = prefs.getString("cam", null)
        cam = cams.firstOrNull { it.key == k }
            ?: cams.firstOrNull { it.phys == null && it.facing == android.hardware.camera2.CameraCharacteristics.LENS_FACING_FRONT }
            ?: cams.firstOrNull()
        loadCamSettings()
    }

    /** resolution and fps limit are stored per camera; defaults = 1920x1080 (or closest) and the HAL's max fps up to 30 */
    private fun loadCamSettings() {
        val c = cam ?: return
        val sizes = halSizes(c)
        val ss = prefs.getString("size_${c.key}", null)
        camSize = sizes.firstOrNull { "${it.width}x${it.height}" == ss }
            ?: sizes.minByOrNull { Math.abs(it.width * it.height - 1920 * 1080) }
        val fo = fpsOptions(c)
        val fv = prefs.getInt("fps_${c.key}", -1)
        camFpsLimit = if (fv in fo) fv else (fo.filter { it <= 30 }.maxOrNull() ?: fo.firstOrNull() ?: 30)
    }

    private fun halSizes(c: CamOpt): List<Size> = runCatching {
        val m = chars(c.phys ?: c.id).get(android.hardware.camera2.CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        m!!.getOutputSizes(android.graphics.ImageFormat.YUV_420_888).filter { it.width >= 480 }
            .distinct().sortedByDescending { it.width * it.height }
    }.getOrDefault(emptyList())

    private fun aeRanges(c: CamOpt): List<android.util.Range<Int>> = runCatching {
        chars(c.id).get(android.hardware.camera2.CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)!!.toList()
    }.getOrDefault(emptyList())

    /** fps the HAL can deliver (range uppers), plus lower limits that are reached by dropping frames */
    private fun fpsOptions(c: CamOpt): List<Int> {
        val rs = aeRanges(c)
        val max = rs.maxOfOrNull { it.upper } ?: 30
        return (rs.map { it.upper } + listOf(5, 10, 15, 20, 24, 25, 30)).filter { it in 1..max }.distinct().sortedDescending()
    }

    /** AE range for a limit f: [f,f]; else upper==f with the highest floor; else the lowest upper above f (rest dropped in software) */
    private fun pickAe(c: CamOpt, f: Int): android.util.Range<Int>? {
        val rs = aeRanges(c)
        return rs.firstOrNull { it.lower == f && it.upper == f }
            ?: rs.filter { it.upper == f }.maxByOrNull { it.lower }
            ?: rs.filter { it.upper > f }.sortedWith(compareBy({ it.upper }, { -it.lower })).firstOrNull()
    }

    private fun lensLabel() = cam?.let {
        (if (it.phys == null) "#${it.id}" else "#${it.id}/${it.phys}") + (if (it.zoom < 1f) " %.1fx ▾".format(it.zoom) else " %.1fmm ▾".format(it.focal))
    } ?: "cam?"
    private fun resLabel() = camSize?.let { "${it.width}x${it.height} ▾" } ?: "res?"

    private fun pickLens() {
        if (cams.isEmpty()) cams = enumCams()
        pick(getString(R.string.lens_title), cams.map { it.label }.toTypedArray(), cams.indexOf(cam)) { i ->
            cam = cams[i]; prefs.edit().putString("cam", cams[i].key).apply()
            loadCamSettings(); refreshCamButtons(); noFpsLock = false; bindCamera()
        }
    }

    private fun pickRes() {
        val c = cam ?: return
        val sizes = halSizes(c)
        pick(getString(R.string.res_title), sizes.map { "${it.width}x${it.height}" }.toTypedArray(), sizes.indexOf(camSize)) { i ->
            camSize = sizes[i]; prefs.edit().putString("size_${c.key}", "${sizes[i].width}x${sizes[i].height}").apply()
            refreshCamButtons(); bindCamera()
        }
    }

    private fun pickCamFps() {
        val c = cam ?: return
        val fo = fpsOptions(c)
        val hal = aeRanges(c).map { it.upper }.toSet()
        pick(getString(R.string.camfps_title), fo.map { if (it in hal) "$it fps" else "$it fps *" }.toTypedArray(), fo.indexOf(camFpsLimit)) { i ->
            camFpsLimit = fo[i]; prefs.edit().putInt("fps_${c.key}", fo[i]).apply()
            refreshCamButtons(); noFpsLock = false; bindCamera()
        }
    }

    private fun refreshCamButtons() {
        btnLens.text = lensLabel(); btnRes.text = resLabel(); btnCamFps.text = getString(R.string.camfps_btn, camFpsLimit)
    }

    private fun bindCamera() {
        val f = ProcessCameraProvider.getInstance(this)
        f.addListener({
            val provider = f.get()
            // CameraX's default AspectRatioStrategy prefers 4:3, which silently turned every 16:9 pick
            // (1920x1080, 1280x720...) into a 4:3 size. Match the picked size's aspect and put the exact
            // size first in the filter so the picked HAL size is what analysis gets.
            val want = camSize ?: Size(1920, 1080)
            val sameAspect = { z: Size -> Math.abs(z.width.toLong() * want.height - z.height.toLong() * want.width) <= want.width.toLong() * want.height / 100 }
            val ar = if (Math.abs(want.width * 3 - want.height * 4) < want.width / 50) androidx.camera.core.AspectRatio.RATIO_4_3 else androidx.camera.core.AspectRatio.RATIO_16_9
            val rs = ResolutionSelector.Builder()
                .setAspectRatioStrategy(androidx.camera.core.resolutionselector.AspectRatioStrategy(ar, androidx.camera.core.resolutionselector.AspectRatioStrategy.FALLBACK_RULE_AUTO))
                .setResolutionStrategy(ResolutionStrategy(want, ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER))
                .setResolutionFilter { list, _ -> list.sortedWith(compareBy({ if (it == want) 0 else if (sameAspect(it)) 1 else 2 }, { Math.abs(it.width * it.height - want.width * want.height) })) }
                .build()
            val ab = ImageAnalysis.Builder()
                .setResolutionSelector(rs)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            val c = cam
            val fr = if (noFpsLock || c == null) null else pickAe(c, camFpsLimit)
            Log.i(TAG, "bind ${c?.key} size=$camSize limit=$camFpsLimit ae=$fr")
            if (fr != null) lockFps(ab, fr)
            if (c?.phys != null && Build.VERSION.SDK_INT >= 28)
                androidx.camera.camera2.interop.Camera2Interop.Extender(ab).setPhysicalCameraId(c.phys)
            val analysis = ab.build()
            analysis.setAnalyzer(analysisExec) { img -> analyze(img) }
            // small preview stream with the same aspect (= same field of view) as the analysis stream
            val pw = 640; val ph = (640L * want.height / want.width).toInt()
            val prs = ResolutionSelector.Builder()
                .setAspectRatioStrategy(androidx.camera.core.resolutionselector.AspectRatioStrategy(ar, androidx.camera.core.resolutionselector.AspectRatioStrategy.FALLBACK_RULE_AUTO))
                .setResolutionStrategy(ResolutionStrategy(Size(pw, ph), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                .setResolutionFilter { list, _ -> list.sortedWith(compareBy({ if (sameAspect(it)) 0 else 1 }, { Math.abs(it.width - pw) })) }
                .build()
            val pb = Preview.Builder().setResolutionSelector(prs)
            if (c?.phys != null && Build.VERSION.SDK_INT >= 28)
                androidx.camera.camera2.interop.Camera2Interop.Extender(pb).setPhysicalCameraId(c.phys)
            val pv = pb.build().also { it.setSurfaceProvider(preview.surfaceProvider) }
            val sel = if (c == null) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.Builder().addCameraFilter { infos ->
                infos.filter { androidx.camera.camera2.interop.Camera2CameraInfo.from(it).cameraId == c.id }
            }.build()
            try {
                provider.unbindAll()
                val camera = provider.bindToLifecycle(this, sel, analysis, pv)
                if (c != null && c.zoom < 1f) camera.cameraControl.setZoomRatio(c.zoom)
                camErr = ""
                // watchdog: no frames within 2.5 s -> rebind once without the fps lock
                val gen = ++bindGen; val f0 = camFrames
                ui.postDelayed({
                    if (gen == bindGen && camFrames == f0 && !noFpsLock) {
                        Log.w(TAG, "no frames from ${c?.key}, rebinding without fps lock")
                        camErr = getString(R.string.cam_nofps); noFpsLock = true; bindCamera()
                    }
                }, 2500)
            } catch (e: Exception) {
                Log.e(TAG, "camera bind failed", e)
                camErr = getString(R.string.cam_fail, e.javaClass.simpleName + " " + (e.message ?: "").take(60))
                if (!noFpsLock) { noFpsLock = true; ui.postDelayed({ bindCamera() }, 500) }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private var frameBmp: Bitmap? = null
    private var rowBuf: java.nio.ByteBuffer? = null

    /** RGBA_8888 plane -> reused Bitmap (no per-frame 6 MB allocation like ImageProxy.toBitmap()) */
    private fun frameBitmap(img: ImageProxy): Bitmap {
        val w = img.width; val h = img.height
        var b = frameBmp
        if (b == null || b.width != w || b.height != h) { b?.recycle(); b = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888); frameBmp = b }
        val pl = img.planes[0]
        val buf = pl.buffer; buf.rewind()
        if (pl.rowStride == w * 4 && pl.pixelStride == 4) {
            b!!.copyPixelsFromBuffer(buf)
        } else {
            var rb = rowBuf
            if (rb == null || rb.capacity() != w * h * 4) { rb = java.nio.ByteBuffer.allocateDirect(w * h * 4); rowBuf = rb }
            rb!!.clear()
            val row = ByteArray(w * 4)
            for (y in 0 until h) { buf.position(y * pl.rowStride); buf.get(row, 0, w * 4); rb.put(row) }
            rb.rewind(); b!!.copyPixelsFromBuffer(rb)
        }
        return b
    }

    private fun analyze(img: ImageProxy) {
        camFrames++
        val r = img.imageInfo.rotationDegrees
        if (r == 90 || r == 270) { camW = img.height; camH = img.width } else { camW = img.width; camH = img.height }
        try {
            if (!trackerReady) return
            // software limit: min(camera fps limit, tracking cap), gated on sensor timestamps (25% tolerance)
            val lim = minOf(capFps, camFpsLimit)
            val ts = img.imageInfo.timestamp
            val iv = 1_000_000_000L / lim
            if (lastKeptTs != 0L && ts - lastKeptTs < iv * 3 / 4) return
            lastKeptTs = if (lastKeptTs != 0L && ts - lastKeptTs < iv * 2) lastKeptTs + iv else ts
            val t0 = android.os.SystemClock.elapsedRealtimeNanos()
            val raw = frameBitmap(img)
            tracker.process(raw, img.imageInfo.rotationDegrees, t0)
        } catch (e: Throwable) {
            Log.e(TAG, "analyze failed", e)
        } finally {
            img.close()
        }
    }

    private fun handleFrameQuery(q: String): String {
        if (q.contains("face=1")) tracker.faceWantedUntil = android.os.SystemClock.uptimeMillis() + 1500
        return tracker.latestJson
    }

    // ------------------------------------------------------------------ webview
    inner class Bridge {
        @JavascriptInterface
        fun renderFps(): Int = capFps

        @JavascriptInterface
        fun cameraInfo(): String {
            // wait briefly for the first frame so the placeholder stream has the real aspect ratio
            var n = 0
            while (camW == 0 && n++ < 100) Thread.sleep(50)
            val w = if (camW > 0) camW else 720
            val h = if (camH > 0) camH else 1280
            val k = 480.0 / maxOf(w, h)
            return "{\"w\":${Math.round(w * k)},\"h\":${Math.round(h * k)}}"
        }

        @JavascriptInterface
        fun cameraStart() { ui.post { startCamera() } }

        @JavascriptInterface
        fun vmcOpen(host: String, port: Int) { }

        @JavascriptInterface
        fun vmcSend(b64: String, host0: String, port0: Int) {
            // only send once the user has set a receiver (saved in prefs); the page's default 127.0.0.1 is ignored
            if (!vmcOn) return
            val host = vmcHost; val port = vmcPort
            if (host.isEmpty()) return
            val data = Base64.decode(b64, Base64.NO_WRAP)
            vmcTarget = "$host:$port"
            udpExec.execute {
                try {
                    val s = udp ?: DatagramSocket().also { udp = it }
                    val addr = addrCache.getOrPut(host) { InetAddress.getByName(host) }
                    s.send(DatagramPacket(data, data.size, addr, port))
                    vmcPackets.incrementAndGet()
                } catch (e: Exception) {
                    Log.w(TAG, "VMC send failed", e)
                    addrCache.remove(host)
                }
            }
        }

        @JavascriptInterface
        fun stats(): String = "{\"fps\":${tracker.fps},\"pose_ms\":${tracker.poseMs},\"hands_ms\":${tracker.handsMs},\"face_ms\":${tracker.faceMs},\"total_ms\":${tracker.totalMs}}"
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun setupWebView() {
        web.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            databaseEnabled = true
            mediaPlaybackRequiresUserGesture = false
            allowFileAccess = true
            mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
            cacheMode = WebSettings.LOAD_DEFAULT
        }
        web.addJavascriptInterface(Bridge(), "XRANative")

        val startJs = assets.open("inject/document_start.js").bufferedReader().readText()
        if (WebViewFeature.isFeatureSupported(WebViewFeature.DOCUMENT_START_SCRIPT)) {
            WebViewCompat.addDocumentStartJavaScript(web, startJs, setOf("*"))
        }

        web.webChromeClient = object : WebChromeClient() {
            override fun onPermissionRequest(request: PermissionRequest) {
                ui.post { request.grant(request.resources) }
            }
            override fun onConsoleMessage(m: ConsoleMessage): Boolean {
                Log.d("XRAWeb", "${m.message()} @${m.sourceId()}:${m.lineNumber()}")
                return true
            }
        }
        web.webViewClient = object : WebViewClient() {
            override fun onPageStarted(view: WebView, url: String?, favicon: Bitmap?) {
                if (!WebViewFeature.isFeatureSupported(WebViewFeature.DOCUMENT_START_SCRIPT))
                    view.evaluateJavascript(startJs, null)
            }

            override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? {
                return intercept(request.url, request.method, request.requestHeaders)
            }
        }
    }

    private fun jsResponse(bytes: ByteArray, mime: String = "application/javascript"): WebResourceResponse =
        WebResourceResponse(mime, "utf-8", 200, "OK",
            mapOf("Access-Control-Allow-Origin" to "*", "Cache-Control" to "no-store"),
            ByteArrayInputStream(bytes))

    private fun intercept(uri: Uri, method: String? = "GET", headers: Map<String, String>? = null): WebResourceResponse? {
        val path = uri.path ?: return null
        if (uri.host != "sao.animetheme.com") return null
        return try {
            when {
                path == NATIVE_PATH ->
                    jsResponse(handleFrameQuery(uri.query ?: "").toByteArray(), "application/json")
                path.endsWith("/js/@mediapipe/tasks/tasks-vision/XRA_module_loader.js") ->
                    jsResponse(assets.open("inject/XRA_module_loader.js").readBytes())
                path.endsWith("/js/mocap_lib_module.js") -> patched(uri, path) { s ->
                    replaceAll(s, "canvas_hands = (data.options.use_canvas_hands && !data.options.use_holistic) ? _canvas_hands : null;",
                        "canvas_hands = null; /* XRA native: hands come pre-mapped to the full frame */")
                }
                path.endsWith("/js/facemesh_lib.js") -> patched(uri, path) { s ->
                    replaceAll(s, "faces = model.detect(rgba, vt);",
                        "self.__xra_face_bb = {w:w, h:h, bb:bb}; faces = model.detect(rgba, vt);")
                }
                path.endsWith("/animate.js") && path.contains("XR Animator") -> patched(uri, path) { s ->
                    replaceAll(s, "var panorama_f = (SA_topmost_window.is_cordova) ? 'cdungeon' : '_dungeon';",
                        "var panorama_f = '_dungeon'; /* XRA native: cordova flag set, keep web texture path */")
                }
                path.endsWith("/js/SA_system_emulation.min.js") -> patched(uri, path) { s0 ->
                    // streamer mode: take the desktop VMC branch (ends with "hide 3D avatar"), default to
                    // full-body mocap instead of the mobile face-only default, and keep the camera/mesh view
                    // minified names change between upstream releases, so match identifiers with \w+
                    var s = replaceRe(s0, Regex("""if\(webkit_electron_mode&&MMD_SA\.THREEX\.enabled\)\{(\w+)=!!\(MMD_SA_options\.user_camera\.streamer_mode\.VMC_sender_enabled"""),
                        "if((webkit_electron_mode||self.__xra_native)&&MMD_SA.THREEX.enabled){$1=!!(MMD_SA_options.user_camera.streamer_mode.VMC_sender_enabled")
                    s = replaceAll(s, "is_mobile?System._browser.translation.get(\"XR_Animator.UI.streamer_mode.mocap_initializing.face\"",
                        "(is_mobile&&!self.__xra_native)?System._browser.translation.get(\"XR_Animator.UI.streamer_mode.mocap_initializing.face\"")
                    s = replaceRe(s, Regex("""if\((\w+)\.initialized&&\1\.visible\)\1\.hide\(\);(\w+)\(\)\}\}"""),
                        "if($1.initialized&&$1.visible&&!self.__xra_native)$1.hide();$2()}}")
                    // the mesh canvas copies objectFit from the (hidden) video canvas, which is "" -> "fill":
                    // its window box keeps the screen aspect, so any non-matching camera aspect got stretched
                    replaceRe(s, Regex("""(\w+)\.objectFit=(\w+)\.objectFit\}"""),
                        "$1.objectFit=$2.objectFit||(self.__xra_native?\"contain\":\"\")}")
                }
                path.endsWith("/js/osc.min.js") -> patched(uri, path) { s ->
                    s + assets.open("inject/osc_append.js").bufferedReader().readText()
                }
                webPack.handles(uri.host, method, headers) ->
                    webPack.respond(uri.encodedPath ?: path, uri.toString(), method, headers)
                else -> null
            }
        } catch (e: Exception) {
            Log.e(TAG, "intercept failed for $uri", e)
            null
        }
    }

    private fun replaceAll(s: String, a: String, b: String): String {
        val n = s.split(a).size - 1
        Log.i(TAG, "patch '${a.take(40)}…' matched $n time(s)")
        return s.replace(a, b)
    }

    private fun replaceRe(s: String, re: Regex, b: String): String {
        val n = re.findAll(s).count()
        if (n == 0) Log.w(TAG, "patch /${re.pattern.take(40)}…/ matched 0 times") else Log.i(TAG, "patch /${re.pattern.take(40)}…/ matched $n time(s)")
        return re.replace(s, b)
    }

    private fun patched(uri: Uri, key: String, fn: (String) -> String): WebResourceResponse {
        val bytes = patchCache.getOrPut(key) {
            val src = webPack.bytes(uri.encodedPath ?: key, uri.toString()).toString(Charsets.UTF_8)
            fn(src).toByteArray()
        }
        return jsResponse(bytes)
    }

    override fun onDestroy() {
        super.onDestroy()
        localServer.stop()
        analysisExec.execute { tracker.close() }
        runCatching { udp?.close() }
        web.destroy()
    }
}
