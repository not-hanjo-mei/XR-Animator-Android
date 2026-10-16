// XRAA (XR Animator for Android) - injected at document start.
// The native side owns the camera, so getUserMedia returns a tiny placeholder stream
// with the camera's aspect ratio. Pixels are never used: inference results come from native.
// Present as the official XRA Mobile (cordova) build: unlocks the desktop inventory
// (VMC protocol, streamer mode, ...) that the plain mobile web page hides.
if (typeof window.is_cordova === 'undefined') window.is_cordova = true;
window.__xra_native = true;

(function () {
  if (window.__xra_native_cam) return;
  window.__xra_native_cam = true;
  if (!window.XRANative || !navigator.mediaDevices) return;

  function info() {
    try { return JSON.parse(XRANative.cameraInfo()); } catch (e) { return { w: 240, h: 320 }; }
  }

  let stream = null;
  function make_stream() {
    if (stream && stream.getVideoTracks()[0].readyState == 'live') return stream;
    const i = info();
    const canvas = document.createElement('canvas');
    canvas.width = i.w; canvas.height = i.h;
    const ctx = canvas.getContext('2d');
    let n = 0;
    const draw = () => {
      ctx.fillStyle = '#1e1e1e';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = (n++ & 1) ? '#1e1e1f' : '#1f1e1e';
      ctx.fillRect(0, 0, 2, 2);
      // keep the placeholder's aspect in sync with the native camera (resolution/lens switches,
      // or a first frame that arrived before the final resolution was bound)
      if ((n % 30) == 0) {
        try {
          const j = JSON.parse(XRANative.cameraInfo());
          if (j.w > 0 && j.h > 0 && (j.w != canvas.width || j.h != canvas.height)) {
            canvas.width = j.w; canvas.height = j.h;
            settings.width = j.w; settings.height = j.h; settings.aspectRatio = j.w / j.h;
            console.log('(XRA native: placeholder stream resized to ' + j.w + 'x' + j.h + ')');
          }
        } catch (e) {}
      }
      requestAnimationFrame(draw);
    };
    stream = canvas.captureStream(30);
    const track = stream.getVideoTracks()[0];
    const settings = { width: i.w, height: i.h, frameRate: 30, deviceId: 'xra-native', groupId: 'xra', facingMode: 'user', aspectRatio: i.w / i.h };
    track.getSettings = () => Object.assign({}, settings);
    draw();
    track.getCapabilities = () => ({ width: { min: settings.width, max: settings.width }, height: { min: settings.height, max: settings.height }, frameRate: { min: 1, max: 30 }, deviceId: 'xra-native', facingMode: ['user', 'environment'] });
    track.getConstraints = () => ({});
    track.applyConstraints = async () => {};
    Object.defineProperty(track, 'label', { get: () => 'XRA Native Camera' });
    return stream;
  }

  const md = navigator.mediaDevices;
  md.getUserMedia = async function (c) {
    if (c && c.video) {
      XRANative.cameraStart();
      return make_stream();
    }
    throw new DOMException('Audio is not available', 'NotAllowedError');
  };
  md.enumerateDevices = async function () {
    return [{ deviceId: 'xra-native', kind: 'videoinput', label: 'XRA Native Camera', groupId: 'xra', toJSON() { return this; } }];
  };
})();

// The web page normally center-crops the camera to the window's aspect ratio (target_width x
// target_height). Native landmarks are normalized to the FULL camera frame, so disable that crop:
// with pixel_limit.disabled the page's video_canvas uses the placeholder stream's exact size,
// which has the native camera's aspect ratio.
(function () {
  if (window.__xra_pixel_limit_hook) return;
  window.__xra_pixel_limit_hook = true;
  const t = setInterval(() => {
    try {
      const pl = window.MMD_SA_options && MMD_SA_options.user_camera && MMD_SA_options.user_camera.pixel_limit;
      if (pl) {
        if (!pl.disabled) { pl.disabled = true; console.log('(XRA native: camera pixel_limit crop disabled)'); }
      }
    } catch (e) {}
  }, 500);
})();

// Streamer mode / mesh fixes:
// - streamer mode only offers "hide 3D avatar" at the end when VMC_sender_enabled is set; default it on.
// - the face/pose wireframe is drawn at 0.25x of the camera display in the top-right corner on mobile,
//   which the native preview covers. Draw it at ~45% of the window width on the left edge instead.
(function () {
  setInterval(() => {
    try {
      const uc = window.MMD_SA_options && MMD_SA_options.user_camera;
      if (!uc) return;
      if (uc.streamer_mode && uc.streamer_mode.VMC_sender_enabled == null) uc.streamer_mode.VMC_sender_enabled = true;
      const wf = uc.display && uc.display.wireframe;
      const cam = window.System && System._browser && System._browser.camera;
      const vs = cam && cam.video_canvas && cam.video_canvas.style;
      if (wf && vs && vs.pixelWidth > 0) {
        const sc = Math.min(window.innerWidth * 0.45 / vs.pixelWidth, 4);
        if (!wf.__xra || Math.abs((wf.scale || 0) - sc) > 0.01) { wf.scale = sc; wf.left = -1; wf.top = -0.4; wf.align_with_video = false; wf.__xra = true; }
      }
    } catch (e) {}
  }, 500);
})();

// Render throttle: camera/inference run at <=30 fps, so drawing the page at the display's 60/120 Hz
// only burns GPU and heats the phone. Gate requestAnimationFrame per display tick to the native cap.
(function () {
  if (window.__xra_raf_cap) return;
  window.__xra_raf_cap = true;
  const raf = window.requestAnimationFrame.bind(window);
  let fps = 30, polls = 0, last = -1e9, tickT = -1, tickOk = true;
  function cap() {
    if ((polls++ % 120) == 0) { try { const f = +XRANative.renderFps(); if (f > 0) fps = f; } catch (e) {} }
    return fps;
  }
  window.requestAnimationFrame = function (cb) {
    const gate = function (t) {
      if (t !== tickT) {
        tickT = t;
        const iv = 1000 / cap();
        tickOk = (t - last) >= iv - 4;            // 4 ms slack so 60 Hz -> exactly every 2nd tick
        if (tickOk) last = (t - last > iv * 3) ? t : last + iv;
      }
      if (tickOk) cb(t); else raf(gate);
    };
    return raf(gate);
  };
})();
