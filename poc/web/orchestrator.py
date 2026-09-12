#!/usr/bin/env python3
"""orchestrator — photo(s) in, tracked VRM avatar in livebody out, one page.

    python3 poc/web/orchestrator.py                # http://localhost:8901/
    python3 poc/web/orchestrator.py --port 8902 --blender /Applications/Blender.app/Contents/MacOS/Blender

Stdlib only (Python 3.12). Serves the repo root like `serve.sh` did, so
livebody.html, the WASM and the models load from the same origin, plus:

    GET  /                        upload page
    POST /jobs                    start a job (multipart: photos or a mesh)
    GET  /jobs/<id>               progress page (polls /jobs/<id>.json)
    GET  /jobs/<id>.json          job state
    POST /jobs/<id>/upload        hand-deliver a stage's output (manual rig)
    POST /jobs/<id>/cancel

Stages of a job:   generate  →  rig  →  convert  →  done
    generate   photo(s) → textured mesh.        provider: meshy | tripo | upload
    rig        mesh → rigged humanoid FBX.       rigger:   meshy | tripo | mixamo (manual upload)
    convert    FBX → VRM 1.0 via poc/tools/fbx_to_vrm.py (Blender headless)
    done       poc/models3d/generated/<id>/avatar.vrm, opened in livebody.html?model=

Every artifact lives in poc/models3d/generated/<id>/ (gitignored: uploaded
photos are real people's likenesses). Job state is mirrored to job.json so a
restarted server still lists and serves finished jobs.

Environment:  MESHY_API_KEY   for provider/rigger "meshy"   (meshy.ai/settings/api)
              TRIPO_API_KEY   for provider/rigger "tripo"   (platform.tripo3d.ai → API Keys)
              BLENDER         path to the Blender binary (or --blender)
              VRM_ADDON_ZIP   passed through to fbx_to_vrm.py
"""
from __future__ import annotations

import argparse, base64, email, email.policy, html, io, json, mimetypes, os
import shutil, subprocess, sys, threading, time, traceback, urllib.error
import urllib.parse, urllib.request, uuid, zipfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
GEN_DIR = os.path.join(REPO, 'poc', 'models3d', 'generated')
FBX_TO_VRM = os.path.join(REPO, 'poc', 'tools', 'fbx_to_vrm.py')
LIVEBODY = '/poc/web/livebody.html'

STAGES = ('generate', 'rig', 'convert', 'done')
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.webp'}
MESH_EXT = {'.glb', '.gltf', '.fbx', '.obj', '.zip'}

# ----------------------------------------------------------------------------- helpers

def log(job, msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    job['log'].append(line)
    with open(os.path.join(job['dir'], 'log.txt'), 'a') as f:
        f.write(line + '\n')
    print(f'{job["id"]}: {msg}', flush=True)


def save_job(job):
    public = {k: v for k, v in job.items() if k not in ('thread', 'event', 'dir')}
    with open(os.path.join(job['dir'], 'job.json'), 'w') as f:
        json.dump(public, f, indent=1)


def http_json(method, url, headers=None, body=None, timeout=120):
    data = None
    hdrs = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode()
        hdrs['Content-Type'] = 'application/json'
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode() or '{}')
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'{method} {url} -> HTTP {e.code}: {e.read().decode(errors="replace")[:500]}')


def http_multipart(url, headers, field, filename, content, extra=None, timeout=300):
    boundary = '----orch' + uuid.uuid4().hex
    ctype = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
    buf = io.BytesIO()
    for k, v in (extra or {}).items():
        buf.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    buf.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; '
              f'filename="{filename}"\r\nContent-Type: {ctype}\r\n\r\n'.encode())
    buf.write(content)
    buf.write(f'\r\n--{boundary}--\r\n'.encode())
    hdrs = dict(headers)
    hdrs['Content-Type'] = f'multipart/form-data; boundary={boundary}'
    req = urllib.request.Request(url, data=buf.getvalue(), method='POST', headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode() or '{}')
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'POST {url} -> HTTP {e.code}: {e.read().decode(errors="replace")[:500]}')


def download(url, dest, timeout=600):
    req = urllib.request.Request(url, headers={'User-Agent': 'hackcmu-orchestrator/1'})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, 'wb') as f:
        shutil.copyfileobj(r, f)
    return dest


def unpack_if_zip(path, job):
    """Rig/model downloads sometimes arrive zipped with textures beside the mesh."""
    if not zipfile.is_zipfile(path):
        return path
    out = os.path.join(job['dir'], os.path.splitext(os.path.basename(path))[0] + '_unzipped')
    with zipfile.ZipFile(path) as z:
        z.extractall(out)
    meshes = [os.path.join(r, f) for r, _, fs in os.walk(out) for f in fs
              if os.path.splitext(f)[1].lower() in ('.fbx', '.glb', '.gltf', '.obj')]
    if not meshes:
        raise RuntimeError(f'{path} is a zip with no mesh inside')
    log(job, f'unzipped {os.path.basename(path)} -> {os.path.relpath(meshes[0], job["dir"])}')
    return meshes[0]


def find_texture(near_path):
    d = os.path.dirname(near_path)
    for r, _, fs in os.walk(d):
        for f in fs:
            fl = f.lower()
            if fl.endswith(('.png', '.jpg', '.jpeg')) and any(
                    k in fl for k in ('base', 'albedo', 'diffuse', 'color', 'colour', 'texture')):
                return os.path.join(r, f)
    return None


def wait_for_upload(job, stage, want_ext):
    """Manual step: block until the browser posts a file to /jobs/<id>/upload."""
    job['awaiting'] = {'stage': stage, 'accept': sorted(want_ext)}
    save_job(job)
    while True:
        job['event'].wait(timeout=1.0)
        job['event'].clear()
        if job['status'] == 'cancelled':
            raise RuntimeError('cancelled')
        p = job.pop('delivered', None)
        if p:
            job['awaiting'] = None
            save_job(job)
            return p


# ----------------------------------------------------------------------------- Blender

def blender_bin(explicit=None):
    cands = [explicit, os.environ.get('BLENDER'), shutil.which('blender'),
             '/Applications/Blender.app/Contents/MacOS/Blender']
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def run_blender(job, args, timeout=900):
    b = job['blender']
    if not b:
        raise RuntimeError('Blender not found: set BLENDER=/path/to/blender or pass --blender')
    cmd = [b, '--background', '--python', *args]
    log(job, 'blender: ' + ' '.join(os.path.relpath(a, REPO) if os.path.isabs(a) else a for a in cmd[3:]))
    env = dict(os.environ)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=REPO)
    tail = (p.stdout + '\n' + p.stderr).strip().splitlines()[-25:]
    for line in tail:
        log(job, '  | ' + line)
    if p.returncode != 0:
        raise RuntimeError(f'blender exited {p.returncode}')


CONVERT_PY = r"""
import bpy, sys
argv = sys.argv[sys.argv.index('--') + 1:]
src, dst = argv[0], argv[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
ext = src.lower().rsplit('.', 1)[-1]
if ext in ('glb', 'gltf'):  bpy.ops.import_scene.gltf(filepath=src)
elif ext == 'obj':          bpy.ops.wm.obj_import(filepath=src)
elif ext == 'fbx':          bpy.ops.import_scene.fbx(filepath=src)
else: raise SystemExit('unsupported input: ' + src)
bpy.ops.object.select_all(action='SELECT')
if dst.lower().endswith('.fbx'):
    # Mixamo wants one mesh, real-world scale, textures embedded.
    bpy.ops.export_scene.fbx(filepath=dst, use_selection=True, path_mode='COPY',
                             embed_textures=True, apply_scale_options='FBX_SCALE_ALL')
else:
    bpy.ops.export_scene.gltf(filepath=dst, export_format='GLB', use_selection=True)
print('wrote', dst)
"""


def mesh_to(job, mesh_path, fmt):
    """Convert a mesh to .fbx (Mixamo rejects .glb) or .glb (the API riggers want GLB)."""
    if mesh_path.lower().endswith('.' + fmt):
        return mesh_path
    script = os.path.join(job['dir'], '_convert.py')
    with open(script, 'w') as f:
        f.write(CONVERT_PY)
    out = os.path.join(job['dir'], 'model.' + fmt)
    run_blender(job, [script, '--', mesh_path, out])
    return out


# ----------------------------------------------------------------------------- providers
#
# Two hosted generate+rig APIs and one manual path. Endpoints and bodies come from the
# vendors' docs as read on 2026-09-12 (docs.meshy.ai; developers.tripo3d.ai for Tripo v3 —
# Tripo's v2 task API is being retired 2026-11-01). Items marked UNVERIFIED were not
# confirmable from the docs and are the first place to look if a call 4xx's.

def data_uri(path):
    ctype = mimetypes.guess_type(path)[0] or 'application/octet-stream'
    with open(path, 'rb') as f:
        return f'data:{ctype};base64,' + base64.b64encode(f.read()).decode()


def poll(job, label, fetch, done, failed, timeout=1800, every=4):
    """Generic task poller: fetch() -> dict; done/failed(dict) -> bool."""
    t0, last = time.time(), None
    while time.time() - t0 < timeout:
        d = fetch()
        st, prog = d.get('status'), d.get('progress')
        if (st, prog) != last:
            log(job, f'{label}: {st} {"" if prog is None else str(prog) + "%"}'.rstrip())
            job['progress'] = {'stage': label, 'status': st, 'percent': prog}
            save_job(job)
            last = (st, prog)
        if done(d):
            return d
        if failed(d):
            err = d.get('task_error') or d.get('output') or d.get('message') or d
            raise RuntimeError(f'{label}: ended as {st}: {err}')
        if job['status'] == 'cancelled':
            raise RuntimeError('cancelled')
        time.sleep(every)
    raise RuntimeError(f'{label}: timed out after {timeout}s')


class MeshyProvider:
    """Meshy — api.meshy.ai/openapi/v1. Fully documented; images and meshes go up as data URIs.
    image-to-3d has pose_mode a-pose/t-pose (what auto-riggers want); rigging takes any textured
    humanoid GLB facing +Z (Meshy task id or a data URI), 5 credits, FBX + GLB back."""
    BASE = 'https://api.meshy.ai/openapi/v1'
    name = 'meshy'

    def __init__(self, key):
        if not key:
            raise RuntimeError('MESHY_API_KEY is not set')
        self.h = {'Authorization': f'Bearer {key}'}

    def _wait(self, job, path, tid, label):
        return poll(job, label, lambda: http_json('GET', f'{self.BASE}/{path}/{tid}', self.h),
                    lambda d: d.get('status') == 'SUCCEEDED',
                    lambda d: d.get('status') in ('FAILED', 'CANCELED'))

    def generate(self, job, images):
        o = job['options']
        body = {'should_texture': True, 'enable_pbr': bool(o.get('pbr', False)),
                'target_formats': ['glb', 'fbx'], 'origin_at': 'bottom'}
        if o.get('face_limit'):
            body['should_remesh'] = True
            body['target_polycount'] = max(100, min(300000, int(o['face_limit'])))
        if len(images) == 1:
            path = 'image-to-3d'
            body['image_url'] = data_uri(images[0])
            body['pose_mode'] = o.get('pose_mode', 'a-pose')   # only on the single-image endpoint
            body['image_enhancement'] = True
            body['remove_lighting'] = True
        else:
            path = 'multi-image-to-3d'                          # first image = front; rest any order
            body['image_urls'] = [data_uri(p) for p in images[:4]]
        r = http_json('POST', f'{self.BASE}/{path}', self.h, body, timeout=300)
        tid = r.get('result')
        if not tid:
            raise RuntimeError(f'meshy {path}: no task id in {r}')
        log(job, f'meshy {path} task {tid}')
        job['meshy'] = {'generate_task': tid}
        d = self._wait(job, path, tid, 'generate')
        urls = d.get('model_urls') or {}
        if not urls.get('glb'):
            raise RuntimeError(f'generate: no glb in {urls}')
        dest = download(urls['glb'], os.path.join(job['dir'], 'model.glb'))
        log(job, f'model.glb {os.path.getsize(dest)//1024} KB ({d.get("consumed_credits")} credits)')
        if urls.get('fbx'):
            download(urls['fbx'], os.path.join(job['dir'], 'model.fbx'))
        if d.get('thumbnail_url'):
            try: download(d['thumbnail_url'], os.path.join(job['dir'], 'model_preview.png'))
            except Exception as e: log(job, f'preview skipped: {e}')
        tex = ((d.get('texture_urls') or [{}])[0]).get('base_color')
        if tex:
            try:
                download(tex, os.path.join(job['dir'], 'texture_base_color.png'))
                job['texture'] = 'texture_base_color.png'
            except Exception as e:
                log(job, f'texture download skipped: {e}')
        return dest

    def rig(self, job, mesh_path):
        body = {'height_meters': float(job['options'].get('height_m', 1.7))}
        tid = (job.get('meshy') or {}).get('generate_task')
        if tid:
            body['input_task_id'] = tid
        else:
            glb = mesh_to(job, mesh_path, 'glb')            # rigging wants a textured GLB, +Z facing
            body['model_url'] = data_uri(glb)               # UNVERIFIED: request-size cap for data URIs
        r = http_json('POST', f'{self.BASE}/rigging', self.h, body, timeout=300)
        rid = r.get('result')
        if not rid:
            raise RuntimeError(f'meshy rigging: no task id in {r}')
        log(job, f'meshy rigging task {rid}')
        job.setdefault('meshy', {})['rig_task'] = rid
        d = self._wait(job, 'rigging', rid, 'rig')
        res = d.get('result') or {}
        url = res.get('rigged_character_fbx_url')
        if not url:
            raise RuntimeError(f'rig: no fbx url in {res}')
        dest = download(url, os.path.join(job['dir'], 'rigged.fbx'))
        log(job, f'rigged.fbx {os.path.getsize(dest)//1024} KB ({d.get("consumed_credits")} credits). '
                 'Bone naming is undocumented; fbx_to_vrm.py will say if a required bone is unmapped.')
        return unpack_if_zip(dest, job)


class TripoProvider:
    """Tripo v3 — openapi.tripo3d.ai/v3. Rig endpoints are confirmed (rig-check free, rig ~25-30
    credits, `spec: mixamo` = Mixamo bone names, `out_format: fbx`). Generation body and the
    file-upload response are UNVERIFIED for v3 (only v2's are documented); output URLs expire
    5 minutes after issue, so every result is downloaded the moment it appears."""
    BASE = 'https://openapi.tripo3d.ai/v3'
    name = 'tripo'

    def __init__(self, key):
        if not key:
            raise RuntimeError('TRIPO_API_KEY is not set')
        self.h = {'Authorization': f'Bearer {key}'}

    @staticmethod
    def _data(r):
        if isinstance(r, dict) and r.get('code', 0) not in (0, None):
            raise RuntimeError(f'tripo error {r.get("code")}: {r.get("message") or r.get("suggestion") or r}')
        return r.get('data', r) if isinstance(r, dict) else r

    def upload(self, path):
        with open(path, 'rb') as f:
            r = http_multipart(f'{self.BASE}/file-upload', self.h, 'file', os.path.basename(path), f.read())
        d = self._data(r)
        tok = d.get('file_token') or d.get('image_token')     # UNVERIFIED which key v3 uses
        if not tok:
            raise RuntimeError(f'tripo file-upload: no token in {r}')
        return tok

    def _task(self, path, body):
        d = self._data(http_json('POST', f'{self.BASE}/{path}', self.h, body, timeout=300))
        tid = d.get('task_id')
        if not tid:
            raise RuntimeError(f'tripo {path}: no task_id in {d}')
        return tid

    def _wait(self, job, tid, label):
        return poll(job, label, lambda: self._data(http_json('GET', f'{self.BASE}/tasks/{tid}', self.h)),
                    lambda d: d.get('status') == 'success',
                    lambda d: d.get('status') in ('failed', 'banned', 'expired', 'cancelled', 'unknown'),
                    every=3)

    def generate(self, job, images):
        o = job['options']
        toks = [self.upload(p) for p in images[:4]]
        log(job, f'uploaded {len(toks)} image(s)')
        common = {'texture': True, 'pbr': bool(o.get('pbr', False))}
        if o.get('face_limit'):
            common['face_limit'] = int(o['face_limit'])
        if o.get('model_version'):
            common['model_version'] = o['model_version']
        if len(toks) == 1:
            path, body = 'generation/image-to-model', {'input': toks[0], **common}   # UNVERIFIED v3 body
        else:
            # v2 order is fixed [front, left, back, right]; empty slots allowed, front mandatory
            files = [{'type': 'image', 'file_token': t} for t in toks] + [{}] * (4 - len(toks))
            path, body = 'generation/multiview-to-model', {'files': files, **common}  # UNVERIFIED v3 path/body
        tid = self._task(path, body)
        log(job, f'tripo {path} task {tid}')
        job['tripo'] = {'generate_task': tid}
        d = self._wait(job, tid, 'generate')
        out = d.get('output') or {}
        url = out.get('pbr_model') or out.get('model') or out.get('base_model') or out.get('model_url')
        if not url:
            raise RuntimeError(f'generate: no model url in {out}')
        dest = download(url, os.path.join(job['dir'], 'model.glb'))
        log(job, f'model.glb {os.path.getsize(dest)//1024} KB ({d.get("consumed_credit") or d.get("consumed_credits")} credits)')
        if out.get('rendered_image'):
            try: download(out['rendered_image'], os.path.join(job['dir'], 'model_preview.webp'))
            except Exception as e: log(job, f'preview skipped: {e}')
        return dest

    def rig(self, job, mesh_path):
        inp = (job.get('tripo') or {}).get('generate_task')
        if not inp:
            inp = self.upload(mesh_to(job, mesh_path, 'glb'))    # rig accepts glb/gltf/fbx/obj ≤150MB
        d = self._wait(job, self._task('animations/rig-check', {'input': inp}), 'rig-check')
        out = d.get('output') or {}
        if out.get('riggable') is False:
            raise RuntimeError('tripo rig-check: not riggable (needs a clear humanoid, limbs off the '
                               'torso). Try another photo, or rigger "mixamo".')
        rig_type = out.get('rig_type') or 'biped'
        if rig_type != 'biped':
            log(job, f'rig-check says {rig_type}; livebody needs a biped — forcing biped')
        body = {'input': inp, 'model': 'v2.5-20260210', 'rig_type': 'biped',
                'spec': 'mixamo', 'out_format': 'fbx'}
        rid = self._task('animations/rig', body)
        job.setdefault('tripo', {})['rig_task'] = rid
        d = self._wait(job, rid, 'rig')
        out = d.get('output') or {}
        url = out.get('model_url') or out.get('model')
        if not url:
            raise RuntimeError(f'rig: no model url in {out}')
        ext = '.zip' if '.zip' in urllib.parse.urlparse(url).path.lower() else '.fbx'
        dest = download(url, os.path.join(job['dir'], 'rigged' + ext))
        log(job, f'{os.path.basename(dest)} {os.path.getsize(dest)//1024} KB '
                 f'({d.get("consumed_credit") or d.get("credits_consumed")} credits), mixamo bone names')
        return unpack_if_zip(dest, job)


class UploadProvider:
    """No generation API: the user hands in a mesh (or, for rigging, the Mixamo FBX)."""
    name = 'upload'

    def generate(self, job, images):
        if job.get('mesh'):
            return unpack_if_zip(job['mesh'], job)   # a mesh was uploaded instead of photos (maybe zipped)
        log(job, 'no generation provider — waiting for a mesh upload (.glb/.fbx/.obj/.zip)')
        return unpack_if_zip(wait_for_upload(job, 'generate', MESH_EXT), job)

    def rig(self, job, mesh_path):
        fbx = mesh_to(job, mesh_path, 'fbx')  # Mixamo takes fbx/obj, not glb
        job['mixamo_input'] = os.path.relpath(fbx, job['dir'])
        log(job, f'manual rig: upload {os.path.basename(fbx)} to mixamo.com, auto-rig, download '
                 f'FBX (T-pose, "with skin"), then hand it in below')
        return unpack_if_zip(wait_for_upload(job, 'rig', {'.fbx', '.zip'}), job)


PROVIDERS = {'meshy': MeshyProvider, 'tripo': TripoProvider, 'upload': UploadProvider, 'mixamo': UploadProvider}


def make_provider(name, keys):
    cls = PROVIDERS.get(name)
    if not cls:
        raise RuntimeError(f'unknown provider {name!r}')
    return cls(keys.get(name)) if name in ('meshy', 'tripo') else cls()


# ----------------------------------------------------------------------------- job runner

class Jobs:
    def __init__(self, blender, keys):
        self.blender = blender
        self.keys = keys            # {'meshy': ..., 'tripo': ...}
        self.jobs = {}
        self.lock = threading.Lock()
        os.makedirs(GEN_DIR, exist_ok=True)
        gi = os.path.join(GEN_DIR, '.gitignore')
        if not os.path.exists(gi):
            with open(gi, 'w') as f:
                f.write('# generated avatars + uploaded photos: real people, keep local\n*\n!.gitignore\n')
        self._load_finished()

    def _load_finished(self):
        for d in sorted(os.listdir(GEN_DIR)):
            p = os.path.join(GEN_DIR, d, 'job.json')
            if os.path.exists(p):
                try:
                    j = json.load(open(p))
                    j['dir'] = os.path.join(GEN_DIR, d)
                    if j.get('status') == 'running':
                        j['status'] = 'failed'
                        j['error'] = 'server restarted mid-job'
                    self.jobs[j['id']] = j
                except Exception:
                    pass

    def create(self, name, provider, rigger, images, mesh, options):
        jid = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
        d = os.path.join(GEN_DIR, jid)
        os.makedirs(d)
        job = {'id': jid, 'name': name or jid, 'status': 'running', 'stage': 'generate',
               'provider': provider, 'rigger': rigger, 'options': options,
               'created': time.time(), 'log': [], 'error': None, 'awaiting': None,
               'artifacts': {}, 'progress': None, 'dir': d, 'blender': self.blender,
               'event': threading.Event()}
        job['images'] = []
        for fn, data in images:
            p = os.path.join(d, 'input_' + safe_name(fn))
            open(p, 'wb').write(data)
            job['images'].append(p)
        if mesh:
            p = os.path.join(d, 'input_' + safe_name(mesh[0]))
            open(p, 'wb').write(mesh[1])
            job['mesh'] = p
        with self.lock:
            self.jobs[jid] = job
        save_job(job)
        job['thread'] = threading.Thread(target=self._run, args=(job,), daemon=True)
        job['thread'].start()
        return job

    def deliver(self, job, filename, data):
        if not job.get('awaiting'):
            raise RuntimeError('job is not waiting for an upload')
        ext = os.path.splitext(filename)[1].lower()
        if ext not in job['awaiting']['accept']:
            raise RuntimeError(f'{ext} not accepted here, want one of {job["awaiting"]["accept"]}')
        p = os.path.join(job['dir'], f'delivered_{job["awaiting"]["stage"]}_' + safe_name(filename))
        open(p, 'wb').write(data)
        log(job, f'received {os.path.basename(p)} ({len(data)//1024} KB)')
        job['delivered'] = p
        job['event'].set()

    def cancel(self, job):
        if job['status'] == 'running':
            job['status'] = 'cancelled'
            job['event'].set()
            save_job(job)

    def _run(self, job):
        try:
            gen = make_provider(job['provider'], self.keys)
            rig = make_provider(job['rigger'], self.keys)

            job['stage'] = 'generate'; save_job(job)
            mesh = gen.generate(job, job['images'])
            job['artifacts']['mesh'] = os.path.relpath(mesh, job['dir'])
            for prev in ('model_preview.webp', 'model_preview.png'):
                if os.path.exists(os.path.join(job['dir'], prev)):
                    job['artifacts']['preview'] = prev

            job['stage'] = 'rig'; save_job(job)
            rigged = rig.rig(job, mesh)
            job['artifacts']['rigged'] = os.path.relpath(rigged, job['dir'])

            job['stage'] = 'convert'; save_job(job)
            vrm = os.path.join(job['dir'], 'avatar.vrm')
            args = [FBX_TO_VRM, '--', rigged, vrm]
            tex = find_texture(rigged) or (os.path.join(job['dir'], job['texture']) if job.get('texture') else None)
            if tex:
                log(job, f'texture: {os.path.relpath(tex, job["dir"])}')
                args.append(tex)
            run_blender(job, args)
            if not os.path.exists(vrm):
                raise RuntimeError('fbx_to_vrm.py finished but avatar.vrm is missing')
            job['artifacts']['vrm'] = 'avatar.vrm'
            job['livebody'] = f'{LIVEBODY}?model=' + urllib.parse.quote(
                f'/poc/models3d/generated/{job["id"]}/avatar.vrm', safe='/')
            job['stage'] = 'done'; job['status'] = 'done'
            log(job, f'done: {job["livebody"]}')
        except Exception as e:
            if job['status'] != 'cancelled':
                job['status'] = 'failed'
            job['error'] = str(e)
            log(job, 'FAILED: ' + str(e))
            for line in traceback.format_exc().splitlines()[-6:]:
                log(job, '  ' + line)
        finally:
            job['awaiting'] = None
            save_job(job)


def safe_name(fn):
    fn = os.path.basename(fn or 'file')
    return ''.join(c if c.isalnum() or c in '._-' else '_' for c in fn)[:80] or 'file'


# ----------------------------------------------------------------------------- HTTP

PAGE_CSS = '''
body{font:15px/1.45 system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#eee;background:#111}
a{color:#7cf}h1{font-size:1.4rem}code,pre{background:#222;padding:.1em .3em;border-radius:4px}
pre{padding:.7em;overflow:auto;max-height:22em;font-size:12.5px}
fieldset{border:1px solid #333;border-radius:8px;margin:1rem 0;padding:.8rem 1rem}
label{display:block;margin:.4rem 0}input[type=text],select{padding:.3em .5em;background:#1b1b1b;color:#eee;border:1px solid #444;border-radius:4px}
button{padding:.5em 1.1em;border-radius:6px;border:0;background:#2d6cdf;color:#fff;font-size:1rem;cursor:pointer}
.stages{display:flex;gap:.5rem;margin:1rem 0}.stages span{padding:.3em .8em;border-radius:999px;background:#222;color:#888}
.stages .on{background:#2d6cdf;color:#fff}.stages .ok{background:#1f7a3a;color:#fff}.stages .bad{background:#a12;color:#fff}
.err{background:#3a1010;border:1px solid #a33;padding:.7em;border-radius:6px;white-space:pre-wrap}
.box{background:#1a2a1a;border:1px solid #2a6;padding:.8em;border-radius:6px}.muted{color:#888}
img.prev{max-width:240px;border-radius:6px;border:1px solid #333}
'''

INDEX_HTML = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>photo → avatar → livebody</title><style>%(css)s</style>
<h1>photo → 3D → rig → VRM → livebody</h1>
<p class="muted">One job = one avatar. Photos and outputs stay in <code>poc/models3d/generated/</code> (gitignored).</p>
<form method="post" action="/jobs" enctype="multipart/form-data">
<fieldset><legend>1 · input</legend>
<label>Photos (1 = single view; 2–4 = front, left, back, right) <input type="file" name="photo" accept="image/*" multiple></label>
<label>…or an existing mesh (.glb/.fbx/.obj/.zip), skips generation <input type="file" name="mesh" accept=".glb,.gltf,.fbx,.obj,.zip"></label>
<label>Name <input type="text" name="name" placeholder="friend"></label>
<p class="muted">Best photo: whole body visible, standing, arms slightly away from the torso (A-pose), plain background. Auto-riggers need limbs clear of the body.</p>
</fieldset>
<fieldset><legend>2 · generation</legend>
<label>Provider <select name="provider">%(provider_opts)s</select></label>
<label>Pose (Meshy only; auto-riggers want limbs off the torso) <select name="pose_mode"><option value="a-pose">A-pose</option><option value="t-pose">T-pose</option><option value="">as photographed</option></select></label>
<label>Texture <select name="pbr"><option value="">base color (default)</option><option value="1">PBR</option></select></label>
<label>Face limit <input type="text" name="face_limit" placeholder="e.g. 30000 (blank = provider default)" size="34"></label>
</fieldset>
<fieldset><legend>3 · rigging</legend>
<label>Rigger <select name="rigger">%(rigger_opts)s</select></label>
<label>Character height, metres (Meshy) <input type="text" name="height_m" value="1.7" size="5"></label>
<p class="muted">Meshy rigs any textured humanoid GLB (~5 credits); Tripo rigs with Mixamo bone names (~25–30 credits); Mixamo is free but by hand: the page pauses, gives you the FBX to upload, and takes the rigged one back.</p>
</fieldset>
<fieldset><legend>4 · convert</legend>
<p>Blender: <code>%(blender)s</code> → <code>poc/tools/fbx_to_vrm.py</code> → <code>avatar.vrm</code> → <a href="/poc/web/livebody.html">livebody.html</a></p>
</fieldset>
<button type="submit">Start</button>
</form>
<h2>Jobs</h2>%(jobs)s
'''

JOB_HTML = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>job %(id)s</title><style>%(css)s</style>
<p><a href="/">← all jobs</a></p>
<h1>%(name)s <span class="muted">%(id)s</span></h1>
<div class="stages" id="stages"></div>
<div id="progress" class="muted"></div>
<div id="await"></div>
<div id="result"></div>
<div id="error"></div>
<h3>artifacts</h3><div id="artifacts" class="muted">—</div>
<h3>log</h3><pre id="log">…</pre>
<p><form method="post" action="/jobs/%(id)s/cancel" onsubmit="return confirm('cancel this job?')"><button style="background:#444">cancel</button></form></p>
<script>
const id=%(id_json)s, stages=['generate','rig','convert','done'];
async function tick(){
  const r=await fetch(`/jobs/${id}.json`,{cache:'no-store'}); const j=await r.json();
  const cur=stages.indexOf(j.stage);
  document.getElementById('stages').innerHTML=stages.map((s,i)=>{
    let c=''; if(j.status==='done') c='ok'; else if(i<cur) c='ok'; else if(i===cur) c=(j.status==='running')?'on':'bad';
    return `<span class="${c}">${s}</span>`}).join('');
  document.getElementById('progress').textContent=j.progress?`${j.progress.stage}: ${j.progress.status} ${j.progress.percent??''}%%`:'';
  const aw=document.getElementById('await');
  if(j.awaiting){
    const extra = j.awaiting.stage==='rig' && j.mixamo_input ?
      `<p>1. <a href="/poc/models3d/generated/${id}/${j.mixamo_input}" download>download ${j.mixamo_input}</a><br>
       2. <a href="https://www.mixamo.com/#/?page=1&type=Character" target="_blank">mixamo.com</a> → Upload Character → place the markers → Auto-Rig<br>
       3. Download: Format <b>FBX Binary</b>, Pose <b>T-pose</b>, Skin <b>With Skin</b><br>4. hand it in:</p>` :
      `<p>This job has no generation provider — hand in the mesh (${j.awaiting.accept.join(' ')}):</p>`;
    aw.innerHTML=`<div class="box"><b>waiting for you · stage “${j.awaiting.stage}”</b>${extra}
      <form method="post" action="/jobs/${id}/upload" enctype="multipart/form-data">
      <input type="file" name="file" accept="${j.awaiting.accept.join(',')}" required> <button>hand in</button></form></div>`;
  } else aw.innerHTML='';
  const res=document.getElementById('result');
  if(j.status==='done') res.innerHTML=`<div class="box"><b>ready.</b> <a href="${j.livebody}"><b>open in livebody →</b></a><br>
     <span class="muted">or on any static server: livebody.html?model=/poc/models3d/generated/${id}/avatar.vrm</span></div>`;
  document.getElementById('error').innerHTML=j.error?`<div class="err">${j.status}: ${j.error.replace(/</g,'&lt;')}</div>`:'';
  const arts=Object.entries(j.artifacts||{});
  document.getElementById('artifacts').innerHTML=arts.length?arts.map(([k,v])=>
    k==='preview'?`<img class="prev" src="/poc/models3d/generated/${id}/${v}" alt="preview">`:
    `${k}: <a href="/poc/models3d/generated/${id}/${v}">${v}</a>`).join(' · '):'—';
  const pre=document.getElementById('log'); pre.textContent=(j.log||[]).join('\\n'); pre.scrollTop=pre.scrollHeight;
  if(j.status==='running') setTimeout(tick,2000);
}
tick();
</script>
'''


class Handler(SimpleHTTPRequestHandler):
    jobs: Jobs = None  # set in main
    server_version = 'orchestrator/1'

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=REPO, **kw)

    def log_message(self, fmt, *args):  # quieter than SimpleHTTPRequestHandler
        if '/jobs/' in (args[0] if args else '') and '.json' in args[0]:
            return
        super().log_message(fmt, *args)

    # -- routing
    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == '/':
            return self._html(self._index())
        if path.startswith('/jobs/'):
            jid = path[len('/jobs/'):]
            if jid.endswith('.json'):
                job = self.jobs.jobs.get(jid[:-5])
                if not job:
                    return self._text(404, 'no such job')
                pub = {k: v for k, v in job.items() if k not in ('thread', 'event', 'dir', 'blender')}
                return self._json(pub)
            job = self.jobs.jobs.get(jid)
            if not job:
                return self._text(404, 'no such job')
            return self._html(JOB_HTML % {'css': PAGE_CSS, 'id': html.escape(jid), 'id_json': json.dumps(jid),
                                          'name': html.escape(job['name'])})
        self.end_headers_extra = True
        return super().do_GET()

    def end_headers(self):
        # the WASM + model fetches from livebody must not be cached across regenerations
        if getattr(self, 'end_headers_extra', False):
            self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            form = self._form()
            if path == '/jobs':
                photos = [(f.get_filename(), f.get_payload(decode=True)) for f in form.get('photo', [])
                          if f.get_filename()]
                mesh = next(((f.get_filename(), f.get_payload(decode=True)) for f in form.get('mesh', [])
                             if f.get_filename()), None)
                photos = [p for p in photos if os.path.splitext(p[0])[1].lower() in IMAGE_EXT]
                provider = self._field(form, 'provider', 'meshy')
                rigger = self._field(form, 'rigger', 'meshy')
                if provider not in PROVIDERS or rigger not in PROVIDERS:
                    return self._text(400, 'unknown provider/rigger')
                if mesh:
                    provider = 'upload'                 # a mesh skips generation; any rigger still works
                if not photos and not mesh and provider != 'upload':
                    return self._text(400, 'need at least one photo (or a mesh)')
                for role in (provider, rigger):
                    if role in ('meshy', 'tripo') and not self.jobs.keys.get(role):
                        return self._text(400, f'{role.upper()}_API_KEY not set — pick another provider/rigger or set the key')
                options = {}
                if self._field(form, 'pose_mode', '') in ('a-pose', 't-pose'):
                    options['pose_mode'] = self._field(form, 'pose_mode')
                try:
                    options['height_m'] = float(self._field(form, 'height_m', '1.7') or 1.7)
                except ValueError:
                    pass
                if self._field(form, 'pbr', '') in ('0', '1'):
                    options['pbr'] = self._field(form, 'pbr') == '1'
                if self._field(form, 'face_limit', '').strip().isdigit():
                    options['face_limit'] = int(self._field(form, 'face_limit'))
                job = self.jobs.create(self._field(form, 'name', ''), provider, rigger, photos, mesh, options)
                return self._redirect(f'/jobs/{job["id"]}')
            if path.startswith('/jobs/') and path.endswith('/upload'):
                job = self.jobs.jobs.get(path.split('/')[2])
                if not job:
                    return self._text(404, 'no such job')
                f = next((f for f in form.get('file', []) if f.get_filename()), None)
                if not f:
                    return self._text(400, 'no file')
                self.jobs.deliver(job, f.get_filename(), f.get_payload(decode=True))
                return self._redirect(f'/jobs/{job["id"]}')
            if path.startswith('/jobs/') and path.endswith('/cancel'):
                job = self.jobs.jobs.get(path.split('/')[2])
                if job:
                    self.jobs.cancel(job)
                return self._redirect(f'/jobs/{job["id"]}' if job else '/')
            return self._text(404, 'not found')
        except Exception as e:
            traceback.print_exc()
            return self._text(500, f'error: {e}')

    # -- form parsing (stdlib: wrap the body as a MIME message)
    def _form(self):
        n = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(n)
        ctype = self.headers.get('Content-Type', '')
        out = {}
        if ctype.startswith('multipart/form-data'):
            msg = email.message_from_bytes(f'Content-Type: {ctype}\r\nMIME-Version: 1.0\r\n\r\n'.encode() + raw,
                                           policy=email.policy.HTTP)
            for part in msg.iter_parts():
                name = part.get_param('name', header='content-disposition')
                if name:
                    out.setdefault(name, []).append(part)
        elif ctype.startswith('application/x-www-form-urlencoded'):
            for k, v in urllib.parse.parse_qsl(raw.decode()):
                out.setdefault(k, []).append(v)
        return out

    @staticmethod
    def _field(form, name, default=''):
        parts = form.get(name)
        if not parts:
            return default
        p = parts[0]
        if isinstance(p, str):
            return p
        if p.get_filename():
            return default
        return (p.get_payload(decode=True) or b'').decode(errors='replace').strip() or default

    # -- pages
    def _index(self):
        rows = []
        for j in sorted(self.jobs.jobs.values(), key=lambda j: j['created'], reverse=True)[:40]:
            link = f' · <a href="{html.escape(j["livebody"])}">open in livebody</a>' if j.get('livebody') else ''
            rows.append(f'<li><a href="/jobs/{j["id"]}">{html.escape(j["name"])}</a> '
                        f'<span class="muted">{j["status"]} / {j["stage"]} · {j["provider"]}+{j["rigger"]}</span>{link}</li>')
        k = self.jobs.keys
        def opts(kind):
            api = [('meshy', 'Meshy API'), ('tripo', 'Tripo API')]
            manual = ('upload', "none — I'll upload a mesh") if kind == 'provider' else ('mixamo', 'Mixamo, by hand')
            first = next((n for n, _ in api if k.get(n)), manual[0])
            out = []
            for n, label in api + [manual]:
                state = '' if n in ('upload', 'mixamo') else (' (key set)' if k.get(n) else ' (no key)')
                dis = ' disabled' if state == ' (no key)' else ''
                out.append(f'<option value="{n}"{" selected" if n == first else ""}{dis}>{label}{state}</option>')
            return ''.join(out)
        return INDEX_HTML % {
            'css': PAGE_CSS, 'jobs': '<ul>' + ''.join(rows) + '</ul>' if rows else '<p class="muted">none yet</p>',
            'blender': html.escape(self.jobs.blender or 'NOT FOUND — set BLENDER=/path or --blender'),
            'provider_opts': opts('provider'), 'rigger_opts': opts('rigger')}

    # -- responses
    def _html(self, body, code=200):
        b = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(b)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(b)

    def _json(self, obj):
        b = json.dumps(obj, default=str).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(b)

    def _text(self, code, s):
        b = s.encode()
        self.send_response(code)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _redirect(self, to):
        self.send_response(303)
        self.send_header('Location', to)
        self.send_header('Content-Length', '0')
        self.end_headers()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--port', type=int, default=8901)
    ap.add_argument('--bind', default='127.0.0.1', help='camera access needs a secure context: localhost is one')
    ap.add_argument('--blender', default=None)
    a = ap.parse_args()

    keys = {n: os.environ.get(f'{n.upper()}_API_KEY', '').strip() for n in ('meshy', 'tripo')}
    Handler.jobs = Jobs(blender_bin(a.blender), keys)
    srv = ThreadingHTTPServer((a.bind, a.port), Handler)
    print(f'orchestrator on http://{a.bind}:{a.port}/   (repo root served from {REPO})')
    print(f'  blender: {Handler.jobs.blender or "NOT FOUND"}   keys: ' +
          ', '.join(f'{n}={"set" if v else "missing"}' for n, v in keys.items()))
    print(f'  livebody: http://{a.bind}:{a.port}{LIVEBODY}')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
