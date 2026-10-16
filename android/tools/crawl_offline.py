import urllib.parse, asyncio, json, sys, time, urllib.request, os, hashlib
from playwright.async_api import async_playwright
A=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../') + '/app/src/main/assets/inject/'
FAKE="""window.XRANative={cameraInfo:()=>JSON.stringify({w:1080,h:1920}),cameraStart:()=>{},cameraStop:()=>{},renderFps:()=>30};"""
reqs={}
WEB=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../') + '/app/src/main/assets/web/'
IDX=json.load(open(WEB+'index.json'))
async def main(wait):
    async with async_playwright() as p:
        b=await p.chromium.launch(executable_path='/usr/bin/chromium',args=['--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream','--enable-unsafe-swiftshader','--ignore-gpu-blocklist','--no-sandbox'])
        ctx=await b.new_context(user_agent='Mozilla/5.0 (Linux; Android 16; 24031PN0DC Build/BP2A; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/141.0.0.0 Mobile Safari/537.36',
            viewport={'width':915,'height':412},device_scale_factor=2.6,is_mobile=True,has_touch=True)
        await ctx.add_init_script(FAKE+open(A+'document_start.js').read())
        async def route(r):
            u=r.request.url
            if '/__xra_native__/' in u: return await r.fulfill(status=200,body='{}',content_type='application/json')
            if u.endswith('/XRA_module_loader.js'): return await r.fulfill(status=200,body=open(A+'XRA_module_loader.js').read(),content_type='application/javascript')
            if not u.startswith('http'): return await r.continue_()
            sp=urllib.parse.urlsplit(u)
            if sp.netloc=='sao.animetheme.com' and sp.path in IDX:
                e=IDX[sp.path]; reqs[u]=['PACK',0,r.request.resource_type]
                return await r.fulfill(status=200,headers={'content-type':e['m'],'access-control-allow-origin':'*'},body=open(WEB+e['f'][4:],'rb').read())
            reqs[u]=['MISS',0,r.request.resource_type]; return await r.abort()
            def get():
                rq=urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0 (Linux; Android 16) Chrome/141 Mobile'},method=r.request.method,data=r.request.post_data_buffer)
                try:
                    with urllib.request.urlopen(rq,timeout=60) as h: return h.status,dict(h.headers),h.read()
                except urllib.error.HTTPError as e: return e.code,dict(e.headers),e.read()
            try: st,hd,body=await asyncio.to_thread(get)
            except Exception as e:
                reqs[u]=['FAIL',0,r.request.resource_type]; return await r.abort()
            reqs[u]=[st,len(body),r.request.resource_type]
            if st==200 and r.request.method=='GET':
                fn='cache/'+hashlib.md5(u.encode()).hexdigest(); os.makedirs('cache',exist_ok=True); open(fn,'wb').write(body)
            hd={k:v for k,v in hd.items() if k.lower() not in ('content-encoding','transfer-encoding','content-length','connection')}
            hd['access-control-allow-origin']='*'
            await r.fulfill(status=st,headers=hd,body=body)
        await ctx.route('**/*',route)
        def onresp(resp):
            async def f():
                try:
                    body=await resp.body(); reqs[resp.url]=[resp.status,len(body),resp.request.resource_type]
                except Exception as e: reqs.setdefault(resp.url,[resp.status,-1,resp.request.resource_type])
            asyncio.ensure_future(f())
        ctx.on('requestfailed',lambda rq: reqs.setdefault(rq.url,['FAIL',0,rq.resource_type]))
        pg=await ctx.new_page()
        pg.on('console',lambda m: open('console.log','a').write(m.text[:300]+'\n'))
        await pg.goto('https://sao.animetheme.com/XR_Animator.html',wait_until='load',timeout=120000)
        await asyncio.sleep(25)
        await pg.mouse.click(457,206)
        await asyncio.sleep(70)
        for xy in CLICKS:
            await pg.mouse.dblclick(*xy); await asyncio.sleep(6); await pg.screenshot(path=f'c_{xy[0]}_{xy[1]}.png')
        for i in range(wait//20):
            await asyncio.sleep(20)
            await pg.screenshot(path=f'shot{i}.png')
            json.dump(reqs,open('reqs_offline.json','w'),indent=0)
        json.dump(reqs,open('reqs_offline.json','w'),indent=0)
        await b.close()
CLICKS=json.loads(sys.argv[2]) if len(sys.argv)>2 else []
asyncio.run(main(int(sys.argv[1])))
