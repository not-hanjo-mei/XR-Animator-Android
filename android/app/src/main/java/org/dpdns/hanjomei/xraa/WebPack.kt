package org.dpdns.hanjomei.xraa

import android.content.Context
import android.util.Log
import android.webkit.WebResourceResponse
import org.json.JSONObject
import java.io.ByteArrayInputStream
import java.io.File
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

/**
 * Offline pack for the XR Animator web app (sao.animetheme.com, served by GitHub Pages, which some
 * networks can't reach). Requests to that origin are answered, in order, from:
 *   1. the disk cache (files fetched at runtime, plus background updates applied at launch),
 *   2. the pack bundled in the APK (assets/web, built by tools/build_webpack.py),
 *   3. the network; a successful download is written to the disk cache.
 * The origin stays https://sao.animetheme.com, so same-origin logic and the existing patches keep
 * working. Updates: a background pass re-validates every known file with If-None-Match and stages
 * changed files; staged files take effect on the next launch, so one session never mixes versions.
 */
class WebPack(private val ctx: Context) {
    companion object {
        const val TAG = "XRAWebPack"
        const val ORIGIN = "https://sao.animetheme.com"
        private const val UPDATE_INTERVAL_MS = 12 * 3600 * 1000L
        private const val MAX_CACHE_FILE = 100L * 1024 * 1024

        private val MIME = mapOf(
            "html" to "text/html", "htm" to "text/html", "js" to "application/javascript",
            "mjs" to "application/javascript", "json" to "application/json", "css" to "text/css",
            "wasm" to "application/wasm", "png" to "image/png", "jpg" to "image/jpeg",
            "jpeg" to "image/jpeg", "gif" to "image/gif", "webp" to "image/webp",
            "svg" to "image/svg+xml", "ico" to "image/x-icon", "ttf" to "font/ttf",
            "otf" to "font/otf", "woff" to "font/woff", "woff2" to "font/woff2",
            "zip" to "application/zip", "txt" to "text/plain", "xml" to "text/xml",
            "mp3" to "audio/mpeg", "wav" to "audio/wav", "ogg" to "audio/ogg",
            "mp4" to "video/mp4", "webm" to "video/webm"
        )

        fun mimeOf(path: String): String {
            val ext = path.substringAfterLast('/').substringAfterLast('.', "").lowercase()
            return MIME[ext] ?: "application/octet-stream"
        }

        private fun isText(m: String) = m.startsWith("text/") || m.endsWith("javascript") ||
            m.endsWith("json") || m.endsWith("xml") || m.endsWith("svg+xml")

        private fun sha1(s: String): String =
            MessageDigest.getInstance("SHA-1").digest(s.toByteArray()).joinToString("") { "%02x".format(it) }
    }

    private class Entry(val file: String, val mime: String, val etag: String)

    private val dir = File(ctx.filesDir, "webpack").apply { mkdirs() }
    private val filesDir = File(dir, "f").apply { mkdirs() }
    private val indexFile = File(dir, "index.json")
    private val pendingFile = File(dir, "pending.json")
    private val bundled = HashMap<String, Entry>()
    private val disk = ConcurrentHashMap<String, Entry>()
    private val pending = ConcurrentHashMap<String, Entry>()
    private val io = Executors.newFixedThreadPool(4)

    val hitBundled = AtomicInteger(); val hitDisk = AtomicInteger()
    val hitNet = AtomicInteger(); val miss = AtomicInteger(); val hitHead = AtomicInteger()
    @Volatile var updateStatus = ""

    init {
        try {
            val j = JSONObject(ctx.assets.open("web/index.json").bufferedReader().readText())
            for (k in j.keys()) { val e = j.getJSONObject(k); bundled[k] = Entry(e.getString("f"), e.getString("m"), e.optString("e")) }
        } catch (e: Exception) { Log.w(TAG, "no bundled pack: $e") }
        load(indexFile, disk)
        // apply updates staged by the last session
        val p = HashMap<String, Entry>(); load(pendingFile, p)
        if (p.isNotEmpty()) { disk.putAll(p); save(indexFile, disk); pendingFile.delete(); Log.i(TAG, "applied ${p.size} staged update(s)") }
        // drop orphans (replaced versions)
        try {
            val live = disk.values.map { it.file }.toHashSet()
            filesDir.listFiles()?.forEach { if (it.name !in live) it.delete() }
        } catch (_: Exception) {}
        Log.i(TAG, "bundled=${bundled.size} disk=${disk.size}")
    }

    private fun load(f: File, into: MutableMap<String, Entry>) {
        try {
            if (!f.exists()) return
            val j = JSONObject(f.readText())
            for (k in j.keys()) { val e = j.getJSONObject(k)
                if (File(filesDir, e.getString("f")).exists()) into[k] = Entry(e.getString("f"), e.getString("m"), e.optString("e")) }
        } catch (e: Exception) { Log.w(TAG, "bad index ${f.name}: $e") }
    }

    @Synchronized private fun save(f: File, m: Map<String, Entry>) {
        val j = JSONObject()
        for ((k, e) in m) j.put(k, JSONObject().put("f", e.file).put("m", e.mime).put("e", e.etag))
        val tmp = File(f.path + ".tmp"); tmp.writeText(j.toString()); tmp.renameTo(f)
    }

    /** True for a request this pack should answer. HEAD matters: XRA's zip loader keeps zips in the
     *  Cache API and re-validates them with HEAD on later launches; an unanswered HEAD goes to the
     *  (unreachable) network and the whole zip load fails. Range is served as 206 from the same bytes. */
    fun handles(host: String?, method: String?, headers: Map<String, String>?): Boolean =
        host == "sao.animetheme.com" && (method == null || method == "GET" || method == "HEAD")

    private fun etagOf(key: String) = disk[key]?.etag?.takeIf { it.isNotEmpty() } ?: bundled[key]?.etag ?: ""

    private fun sizeOf(key: String): Long {
        disk[key]?.let { e -> val f = File(filesDir, e.file); if (f.exists()) return f.length() }
        bundled[key]?.let { e ->
            try { ctx.assets.openFd(e.file).use { return it.length } } catch (_: Exception) {}
            try { ctx.assets.open(e.file).use { s -> var n = 0L; val b = ByteArray(65536); while (true) { val r = s.read(b); if (r < 0) break; n += r }; return n } } catch (_: Exception) {}
        }
        return -1
    }

    /** Entry point used by the WebView interceptor. */
    fun respond(key: String, url: String, method: String?, headers: Map<String, String>?): WebResourceResponse {
        val range = headers?.entries?.firstOrNull { it.key.equals("Range", true) }?.value
        if (method == "HEAD") {
            var size = sizeOf(key)
            var mime = (disk[key] ?: bundled[key])?.mime ?: mimeOf(key)
            if (size < 0) {  // not local: fetch (and cache) it so HEAD and the later GET agree
                try { size = bytes(key, url).size.toLong(); mime = disk[key]?.mime ?: mimeOf(key) }
                catch (e: Exception) { miss.incrementAndGet(); Log.w(TAG, "HEAD offline, not in pack: $key ($e)")
                    return response(ByteArrayInputStream(ByteArray(0)), mimeOf(key), 504) }
            }
            hitHead.incrementAndGet()
            val h = HashMap<String, String>(); h["Content-Length"] = size.toString(); h["Accept-Ranges"] = "bytes"
            etagOf(key).takeIf { it.isNotEmpty() }?.let { h["ETag"] = it }
            return response(ByteArrayInputStream(ByteArray(0)), mime, 200, h)
        }
        if (range != null) {
            val m = Regex("""bytes=(\d*)-(\d*)""").find(range)
            if (m != null) {
                val all = try { bytes(key, url) } catch (e: Exception) { null }
                if (all != null) {
                    val n = all.size.toLong()
                    val (a, b) = m.destructured
                    var start: Long; var end: Long
                    if (a.isEmpty()) { start = maxOf(0, n - (b.toLongOrNull() ?: 0)); end = n - 1 }
                    else { start = a.toLong(); end = b.toLongOrNull()?.coerceAtMost(n - 1) ?: (n - 1) }
                    if (start >= n || start > end) return response(ByteArrayInputStream(ByteArray(0)), mimeOf(key), 416, mapOf("Content-Range" to "bytes */$n"))
                    val h = mapOf("Content-Range" to "bytes $start-$end/$n", "Accept-Ranges" to "bytes", "Content-Length" to (end - start + 1).toString())
                    return response(ByteArrayInputStream(all, start.toInt(), (end - start + 1).toInt()), (disk[key] ?: bundled[key])?.mime ?: mimeOf(key), 206, h)
                }
            }
        }
        return respond(key, url)
    }

    private fun open(key: String): Pair<InputStream, String>? {
        disk[key]?.let { e -> try { hitDisk.incrementAndGet(); return File(filesDir, e.file).inputStream() to e.mime } catch (_: Exception) { disk.remove(key) } }
        bundled[key]?.let { e -> try { hitBundled.incrementAndGet(); return ctx.assets.open(e.file) to e.mime } catch (_: Exception) {} }
        return null
    }

    private class Fetched(val code: Int, val bytes: ByteArray?, val etag: String, val mime: String?)

    private fun fetch(url: String, etag: String? = null, connectMs: Int = 8000): Fetched {
        val c = URL(url).openConnection() as HttpURLConnection
        try {
            c.connectTimeout = connectMs; c.readTimeout = 30000
            c.setRequestProperty("User-Agent", "XRA-Native")
            if (!etag.isNullOrEmpty()) c.setRequestProperty("If-None-Match", etag)
            val code = c.responseCode
            if (code != 200) return Fetched(code, null, "", null)
            val len = c.contentLengthLong
            if (len > MAX_CACHE_FILE) return Fetched(413, null, "", null)
            val b = c.inputStream.use { it.readBytes() }
            return Fetched(200, b, c.getHeaderField("ETag") ?: "", c.contentType?.substringBefore(';')?.trim())
        } finally { c.disconnect() }
    }

    private fun store(key: String, f: Fetched, into: MutableMap<String, Entry>, persist: File): Entry {
        val name = sha1(key + "|" + f.etag + "|" + f.bytes!!.size)
        val out = File(filesDir, name)
        val tmp = File(filesDir, "$name.tmp"); tmp.writeBytes(f.bytes); tmp.renameTo(out)
        val mime = if (f.mime.isNullOrEmpty() || f.mime == "application/octet-stream") mimeOf(key) else f.mime
        val e = Entry(name, mime, f.etag); into[key] = e; save(persist, into); return e
    }

    /** Raw bytes for a path (used by the patchers). */
    fun bytes(key: String, url: String): ByteArray {
        open(key)?.let { (s, _) -> return s.use { it.readBytes() } }
        val f = fetch(url)
        if (f.code != 200) throw java.io.IOException("HTTP ${f.code} for $url")
        hitNet.incrementAndGet()
        try { store(key, f, disk, indexFile) } catch (e: Exception) { Log.w(TAG, "cache write failed: $e") }
        return f.bytes!!
    }

    fun respond(key: String, url: String): WebResourceResponse {
        val hit = open(key)
        if (hit != null) return response(hit.first, hit.second, 200)
        return try {
            val f = fetch(url)
            if (f.code == 200) {
                hitNet.incrementAndGet()
                val e = try { store(key, f, disk, indexFile) } catch (ex: Exception) { null }
                response(ByteArrayInputStream(f.bytes), e?.mime ?: mimeOf(key), 200)
            } else {
                miss.incrementAndGet(); Log.w(TAG, "HTTP ${f.code}: $key")
                response(ByteArrayInputStream(ByteArray(0)), mimeOf(key), if (f.code in 400..599) f.code else 502)
            }
        } catch (e: Exception) {
            miss.incrementAndGet(); Log.w(TAG, "offline, not in pack: $key ($e)")
            response(ByteArrayInputStream(ByteArray(0)), mimeOf(key), 504)
        }
    }

    private fun response(s: InputStream, mime: String, code: Int, extra: Map<String, String> = emptyMap()): WebResourceResponse {
        val reason = when (code) { 200 -> "OK"; 206 -> "Partial Content"; 404 -> "Not Found"; 416 -> "Range Not Satisfiable"; 504 -> "Gateway Timeout"; else -> "Error" }
        val h = HashMap<String, String>(); h["Access-Control-Allow-Origin"] = "*"; h["Cache-Control"] = "no-cache"
        h["Access-Control-Expose-Headers"] = "ETag, Content-Length, Content-Range"
        h.putAll(extra)
        return WebResourceResponse(mime, if (isText(mime)) "utf-8" else null, code, reason, h, s)
    }

    /** Background re-validation; changed files are staged for the next launch. Never blocks startup. */
    fun scheduleUpdate(delayMs: Long = 20000) {
        val prefs = ctx.getSharedPreferences("webpack", Context.MODE_PRIVATE)
        if (System.currentTimeMillis() - prefs.getLong("last_update", 0) < UPDATE_INTERVAL_MS) return
        Thread {
            try {
                Thread.sleep(delayMs)
                val keys = (bundled.keys + disk.keys).toSortedSet()
                // probe once; if the origin is unreachable, skip quietly
                try { fetch(ORIGIN + "/XR_Animator.html", null, 6000) } catch (e: Exception) { updateStatus = "offline"; return@Thread }
                val changed = AtomicInteger(); val failed = AtomicInteger()
                val tasks = keys.map { k -> io.submit {
                    try {
                        val etag = disk[k]?.etag ?: bundled[k]?.etag
                        val f = fetch(ORIGIN + k, etag)
                        if (f.code == 200 && f.bytes != null && (etag.isNullOrEmpty() || f.etag != etag)) {
                            store(k, f, pending, pendingFile); changed.incrementAndGet()
                        } else if (f.code != 200 && f.code != 304) failed.incrementAndGet()
                    } catch (e: Exception) { failed.incrementAndGet() }
                } }
                tasks.forEach { try { it.get() } catch (_: Exception) {} }
                updateStatus = "checked ${keys.size}, ${changed.get()} new, ${failed.get()} failed"
                Log.i(TAG, "update: $updateStatus")
                if (failed.get() < keys.size / 4) prefs.edit().putLong("last_update", System.currentTimeMillis()).apply()
            } catch (e: Exception) { Log.w(TAG, "update failed: $e") }
        }.apply { isDaemon = true; priority = Thread.MIN_PRIORITY }.start()
    }

    fun stats() = "web B${hitBundled.get()} D${hitDisk.get()} N${hitNet.get()} H${hitHead.get()} X${miss.get()}"
}
