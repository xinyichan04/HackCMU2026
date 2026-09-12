# Tools — making and checking a `.glb` without any 3D software

Two files. Both are stdlib-only: no Blender, no glTF library, no GUI, no venv.

## `make_arkit51_glb.js` — writes a `.glb` from scratch

```bash
node poc/tools/make_arkit51_glb.js poc/models3d/arkit51-test-head.glb
```

Emits a plain glTF 2.0 binary carrying **all 51 drivable ARKit morph targets** (see
`AVATAR3D-HANDOFF.md` §3 — ARKit's 52 minus `tongueOut`, which MediaPipe never emits), spelled in
Apple camelCase, with `mesh.extras.targetNames` present so the names survive into three.js as
`morphTargetDictionary`.

A `.glb` is a 12-byte header + a JSON chunk + a binary chunk. That is the whole container — which
is why writing one needs no tooling at all.

**What it is for:** unblocking the renderer, the tracking hookup and the validator *before* any
character art exists. It is a **format donor**, not a character: the mesh is a sphere and the 51
"expressions" are deterministic bumps on the +Z hemisphere. Every target carries a real non-zero
displacement on purpose — a zero-filled morph passes a name check while proving nothing at runtime.

Deliberately plain: **no Draco, no meshopt, no KTX2.** For contrast, the `facecap.glb` test asset is
gltfpack-compressed and lists `KHR_mesh_quantization`, `EXT_meshopt_compression` and
`KHR_texture_basisu` in `extensionsRequired` — so it will not load in three.js until `MeshoptDecoder`
and `KTX2Loader` are wired up. Worth knowing before you blame your own code.

## `verify_glb_threejs.html` — the acceptance test

The bar for any model is: **loads in three.js, and all 51 names resolve by name.** This page checks
exactly that and nothing else.

```bash
# serve the repo (any static server); then open:
#   verify_glb_threejs.html?model=../models3d/your-model.glb
python3 -m http.server 8899        # from the repo root, for example
```

Reads `window.__RESULT` as JSON: `ok`, `names`, `missing[]`, `headNode`, `tris`,
`jawOpenMaxDelta`, `influences`. Drive it headlessly over CDP if you want it in CI.

Measured on the generated test head: `{"ok":true,"names":51,"missing":[],"headNode":true,
"tris":768,"jawOpenMaxDelta":0.0083,"influences":51}` on three.js r160.

## `glb_inspect.py` — the same bar, headless

```bash
pip install pygltflib
python3 poc/tools/glb_inspect.py poc/models3d/facecap.glb          # human-readable
python3 poc/tools/glb_inspect.py poc/models3d/your.glb --json      # for CI
```

`verify_glb_threejs.html` is the real acceptance test — it proves the file loads in the renderer
that actually ships. This runs the same checks with no browser and no GPU, so it works over SSH and
in CI, and it exits non-zero on failure. Use it to reject a model in seconds; use the HTML page to
bless one.

Checks: ARKit-51 parity after `_L`/`_R` normalisation · `mesh.extras.targetNames` survived export ·
every drivable target actually displaces geometry · head bone or clean root · triangle budget ·
textures embedded · `extensionsRequired` that three.js will not open without extra loaders.

Measured on the three test assets:

| model | result | why |
|---|---|---|
| `facecap.glb` | **PASS 51/51** | `_L`/`_R` spelling, normalised. A strict check scores this 15/51 |
| `arkit51-test-head.glb` | **PASS 51/51** | Apple camelCase, 768 tris |
| `RobotExpressive.glb` | FAIL 0/51 | only `Angry`/`Sad`/`Surprised`; correct rejection |
| `vrm-sample.vrm` | FAIL 0/51 | VRoid `Fcl_*` naming, 399 targets, 19 textures |

Two traps it is deliberately built not to fall into, both of which reject a *good* model:

- **Per-primitive zero morphs are legal.** glTF requires every primitive in a mesh to declare the
  same number of targets, so a body primitive that does not move for `jawOpen` carries an all-zero
  target by design. A target counts as dead only if it displaces nothing across the whole mesh.
- **`tongueOut` is outside the drivable 51.** `facecap.glb` carries it flat. Nothing drives it, so
  it is reported as a note, never a failure.

`.vrm` is parsed too — the file type is chosen by magic bytes, not by extension.

## Naming

Key morphs **by name, never by index**, and normalise before comparing: ARKit shapes appear both as
Apple's `mouthSmileLeft` and as the `_L`/`_R` suffix that Blender, Unreal and Live Link emit. A
strict equality check scores `facecap.glb` 15/51 and rejects a perfectly good model. See
`poc/models3d/README.md`.

## Seeing it work — `poc/web/live3d.html`

```bash
./poc/web/serve.sh              # then open http://localhost:8901/poc/web/live3d.html
```

Camera → MediaPipe → `.glb`, all in the browser, one process owning the camera so the frame you
track is the frame you draw on (`AVATAR3D-HANDOFF.md` §6.3). No UDP, no WebSocket bridge, no
camera sharing, no sync drift.

Blendshapes are keyed **by name through the same `_L`/`_R` normalisation** the validator uses, so
`facecap.glb` drives correctly despite spelling its targets `mouthSmile_L`. Head rotation comes from
MediaPipe's `facialTransformationMatrixes`; screen position and size come from landmarks, which
keeps the overlay locked to the video.

`facecap.glb` needs `MeshoptDecoder` and `KTX2Loader` wired — the page does that, and it is the
concrete cost of the `extensionsRequired` warning `glb_inspect.py` prints.

Measured headless (Chrome with a still face piped in as a fake camera): model loaded, tracker
running, `face=tracking`, **`mapped=51/51`**, head pose pitch 12.3° / yaw −5° / roll 14.4°, 60 fps.

Keys: `h` hide head · `v` hide video · `w` wireframe · `d` pose readout.
