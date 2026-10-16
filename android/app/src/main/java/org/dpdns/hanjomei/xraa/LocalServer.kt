package org.dpdns.hanjomei.xraa

import android.util.Log
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import kotlin.concurrent.thread

/** Fallback transport for the worker shim (http://127.0.0.1:47321/frame) in case WebView
 *  does not route worker requests through shouldInterceptRequest. */
class LocalServer(private val handler: (String) -> String) {
    @Volatile private var running = false
    private var server: ServerSocket? = null

    fun start(port: Int = 47321) {
        if (running) return
        running = true
        thread(name = "xra-local-server", isDaemon = true) {
            try {
                server = ServerSocket(port, 16, InetAddress.getByName("127.0.0.1"))
                while (running) {
                    val s = server!!.accept()
                    thread(isDaemon = true) { serve(s) }
                }
            } catch (e: Exception) {
                if (running) Log.w("XRALocalServer", "server stopped", e)
            }
        }
    }

    private fun serve(s: Socket) {
        s.use { sock ->
            sock.tcpNoDelay = true
            val r = BufferedReader(InputStreamReader(sock.getInputStream()))
            while (true) {
                val line = r.readLine() ?: return
                val parts = line.split(" ")
                if (parts.size < 2) return
                while (true) { val h = r.readLine() ?: return; if (h.isEmpty()) break }
                val method = parts[0]; val path = parts[1]
                val cors = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Private-Network: true\r\n" +
                        "Access-Control-Allow-Methods: GET, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\n"
                val out = sock.getOutputStream()
                if (method == "OPTIONS") {
                    out.write(("HTTP/1.1 204 No Content\r\n$cors" + "Content-Length: 0\r\n\r\n").toByteArray())
                } else {
                    val q = path.substringAfter('?', "")
                    val body = handler(q).toByteArray()
                    out.write(("HTTP/1.1 200 OK\r\n$cors" + "Content-Type: application/json\r\nCache-Control: no-store\r\n" +
                            "Content-Length: ${body.size}\r\n\r\n").toByteArray())
                    out.write(body)
                }
                out.flush()
            }
        }
    }

    fun stop() {
        running = false
        runCatching { server?.close() }
    }
}
