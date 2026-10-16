# XRAA - XR Animator for Android

Full-body motion capture on an Android phone. Camera capture and landmark inference
(pose, hands, face) run natively on the device, XR Animator's solver runs in a WebView,
and the motion data goes out over VMC (OSC over UDP) to a receiver on your LAN
(VSeeFace, Warudo, VNyan, VirtualMotionCapture, etc.).

- Package: `org.dpdns.hanjomei.xraa`
- ABI: arm64-v8a, Android 10 (API 29) or newer
- UI languages: English, Simplified Chinese, Traditional Chinese, Japanese, Korean

## How it works

| Part | Where |
| --- | --- |
| Camera (CameraX, YUV analysis) | `MainActivity.kt` |
| Inference engines: QNN (Snapdragon HTP), NNAPI (other NPUs), LiteRT GPU, CPU (XNNPACK), legacy MediaPipe Tasks | `NpuTracker.kt`, `Net.kt`, `BaseTracker.kt`, `Tracker.kt` |
| XR Animator page served from the bundled pack, disk cache, then network | `WebPack.kt`, `LocalServer.kt` |
| JS shims (fake camera stream, render FPS cap, native VMC transport) | `app/src/main/assets/inject/` |
| Models (BlazePose Full, hand landmark, face mesh, blendshapes) | `app/src/main/assets/m/`, `app/src/main/assets/models/` |

On Snapdragon devices the NPU engines run on the Hexagon HTP through the QNN LiteRT delegate.
On other SoCs they go through NNAPI, with per-model fallback (NNAPI, then GPU, then CPU) and a
crash blacklist for models whose driver compile crashes.

## Build

Requirements: JDK 17, Android SDK (compileSdk 34, build-tools 34.0.0).

```sh
cd android
./gradlew assembleRelease
```

Release signing: create `android/keystore.properties` with `storeFile`, `storePassword`,
`keyAlias` and `keyPassword`. Without it, release builds are signed with your local debug key
(`~/.android/debug.keystore`).

### Offline web pack (optional)

The app can bundle a copy of the XR Animator web app so it works when the origin is unreachable.
The pack is not committed. To build it:

```sh
pip install playwright && playwright install chromium
python3 tools/crawl.py              # writes the list of requested URLs
python3 tools/build_webpack.py reqs.json
```

Without the pack the app loads XR Animator from the network and caches it on disk.

## Tools

- `tools/`: headless crawl and offline pack builder.
- `npu-tools/`: model conversion and quantization scripts, a Python reference of the
  native pipeline (`pipe/pipe.py`, compared against MediaPipe by `pipe/cmp.py`), and a
  stand-alone NPU benchmark app (`bench/`).

## License

XR Animator is licensed under CC BY-NC-SA 4.0, and so is this port: non-commercial use only.
Models are from Google MediaPipe (Apache 2.0).
