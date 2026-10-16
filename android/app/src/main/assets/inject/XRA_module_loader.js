// XRAA (XR Animator for Android) - replacement for js/@mediapipe/tasks/tasks-vision/XRA_module_loader.js
// Instead of loading MediaPipe WASM, every *Landmarker.detectForVideo() returns the latest
// result produced by native (Android MediaPipe Tasks, GPU) inference.
(()=>{
  const ENDPOINTS = [
    self.location.origin + '/__xra_native__/frame',
    'http://127.0.0.1:47321/frame',
  ];
  let ep_ok = null;
  let cache = null, cache_t = -1, cache_wall = 0;

  function fetch_native(query) {
    const list = ep_ok ? [ep_ok] : ENDPOINTS;
    for (const u of list) {
      try {
        const x = new XMLHttpRequest();
        x.open('GET', u + '?' + query, false);
        x.send();
        if (x.status == 200 && x.responseText && x.responseText.charAt(0) == '{') {
          if (!ep_ok) { ep_ok = u; console.log('(XRA native endpoint: ' + u + ')'); }
          return JSON.parse(x.responseText);
        }
      } catch (e) {}
    }
    if (ep_ok) { ep_ok = null; }
    return null;
  }

  // one fetch per frame per worker (pose + hands share it)
  function get(t, need_face) {
    const now = performance.now();
    if (cache && (t === cache_t || now - cache_wall < 4) && (!need_face || cache._face_req)) return cache;
    const d = fetch_native('face=' + (need_face ? 1 : 0) + '&t=' + Math.round(now));
    cache_t = t; cache_wall = now;
    if (d) { d._face_req = need_face; cache = d; }
    return cache;
  }

  const lm = (a)=>a.map(p=>({ x:p[0], y:p[1], z:p[2], visibility:(p[3] == null) ? 1 : p[3], presence:(p[3] == null) ? 1 : p[3] }));

  class FilesetResolver {
    static async forVisionTasks() { return { native:true }; }
  }

  class Base {
    static async createFromOptions(v, o) { return new this(o); }
    static async createFromModelPath(v, p) { return new this({}); }
    constructor(o) { this.options = o || {}; }
    async setOptions(o) { Object.assign(this.options, o); }
    close() {}
  }

  class PoseLandmarker extends Base {
    detectForVideo(video, t) {
      const d = get(t, false);
      if (!d || !d.pose) return { landmarks:[], worldLandmarks:[] };
      return { landmarks:[lm(d.pose.lm)], worldLandmarks:[lm(d.pose.world)] };
    }
    detect(img) { return this.detectForVideo(img, performance.now()); }
  }

  class HandLandmarker extends Base {
    detectForVideo(video, t) {
      const d = get(t, false);
      const r = { landmarks:[], worldLandmarks:[], handedness:[], handednesses:[] };
      if (!d || !d.hands) return r;
      d.hands.forEach((h, i)=>{
        r.landmarks.push(lm(h.lm));
        r.worldLandmarks.push(lm(h.world));
        const c = [{ index:(h.label == 'Left') ? 1 : 0, score:h.score, categoryName:h.label, displayName:h.label }];
        r.handedness.push(c);
      });
      r.handednesses = r.handedness;
      return r;
    }
    detect(img) { return this.detectForVideo(img, performance.now()); }
  }

  class FaceLandmarker extends Base {
    detectForVideo(video, t) {
      const d = get(t, true);
      const r = { faceLandmarks:[], faceBlendshapes:[], facialTransformationMatrixes:[] };
      if (!d || !d.face) return r;
      // native gives full-frame normalized coords; facemesh_lib expects coords normalized to its crop
      const c = self.__xra_face_bb;
      let map = (p)=>({ x:p[0], y:p[1], z:p[2] });
      if (c && c.bb) {
        const u = c.bb.scale || 1;
        const a = c.w / u, rr = c.h / u;       // downscaled full-frame size
        const f = c.bb.w / u;                  // crop size in that space
        const l = c.bb.x, top = c.bb.y;
        map = (p)=>({ x:(p[0]*a - l)/f, y:(p[1]*rr - top)/f, z:p[2]*a/f });
      }
      r.faceLandmarks.push(d.face.lm.map(map));
      if (d.face.bs) {
        r.faceBlendshapes.push({ headIndex:0, headName:'', categories:d.face.bs.map((s, i)=>({ index:i, score:s, categoryName:d.bs_names[i], displayName:'' })) });
      }
      return r;
    }
    detect(img) { return this.detectForVideo(img, performance.now()); }
  }

  // emotion AI / object detection are not bridged in this build
  class ObjectDetector extends Base {
    detectForVideo() { return { detections:[] }; }
    detect() { return { detections:[] }; }
  }
  class ImageClassifier extends Base {
    classifyForVideo() { return { classifications:[] }; }
    classify() { return { classifications:[] }; }
  }

  Object.assign(self, { FilesetResolver, PoseLandmarker, HandLandmarker, FaceLandmarker, ObjectDetector, ImageClassifier });
  console.log('(XRA native shim loaded)');
})();
