#!/usr/bin/env python3
"""Build assets/web (offline pack for sao.animetheme.com) from crawled URL lists.
usage: build_webpack.py reqs.json [more.json ...]   (reqs = {url: [status, size, type]} from work/offline/crawl.py)
Extra paths (mocap worker chain, not reached by the headless crawl) are in EXTRA."""
import sys, json, os, hashlib, urllib.request, urllib.parse, urllib.error, shutil
ORIGIN = 'https://sao.animetheme.com'
EXTRA = ['/js/pose_worker.js', '/js/pose_lib.js', '/js/mocap_lib_module.js', '/js/one_euro_filter.js',
         '/js/facemesh_worker.js', '/js/facemesh_lib.js', '/js/facemesh_triangulation.json']
SKIP = ['/js/@mediapipe/tasks/tasks-vision/XRA_module_loader.js', '/__xra_native__/']
MIME = {'html':'text/html','htm':'text/html','js':'application/javascript','mjs':'application/javascript','json':'application/json',
 'css':'text/css','wasm':'application/wasm','png':'image/png','jpg':'image/jpeg','jpeg':'image/jpeg','gif':'image/gif','webp':'image/webp',
 'svg':'image/svg+xml','ttf':'font/ttf','otf':'font/otf','woff':'font/woff','woff2':'font/woff2','zip':'application/zip','txt':'text/plain','xml':'text/xml',
 'mp3':'audio/mpeg','wav':'audio/wav','ogg':'audio/ogg'}
out = os.path.join(os.path.dirname(__file__), '..', 'app/src/main/assets/web')
paths = set(EXTRA)
for fn in sys.argv[1:]:
    for u, v in json.load(open(fn)).items():
        p = urllib.parse.urlsplit(u)
        if p.netloc == 'sao.animetheme.com' and v[0] == 200: paths.add(p.path)
paths = sorted(p for p in paths if not any(s in p for s in SKIP))
shutil.rmtree(out, ignore_errors=True); os.makedirs(out + '/f')
idx = {}; total = 0
for p in paths:
    try:
        with urllib.request.urlopen(urllib.request.Request(ORIGIN + p, headers={'User-Agent': 'XRA-Native'}), timeout=60) as h:
            b = h.read(); et = h.headers.get('ETag', '')
    except urllib.error.HTTPError as e:
        print('SKIP', e.code, p); continue
    name = 'f/' + hashlib.sha1(p.encode()).hexdigest()
    open(out + '/' + name, 'wb').write(b)
    ext = p.rsplit('/', 1)[-1].rsplit('.', 1)[-1].lower() if '.' in p.rsplit('/', 1)[-1] else ''
    idx[p] = {'f': 'web/' + name, 'm': MIME.get(ext, 'application/octet-stream'), 'e': et}
    total += len(b); print(f'{len(b):>9} {p}')
json.dump(idx, open(out + '/index.json', 'w'), indent=0, sort_keys=True)
print(len(idx), 'files', round(total / 1e6, 2), 'MB')
