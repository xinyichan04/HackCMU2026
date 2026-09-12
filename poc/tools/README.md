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

## `make_stylised_head_glb.py` — a stylised head with a rig that actually works

```bash
pip install pygltflib
python3 poc/tools/make_stylised_head_glb.py poc/models3d/stylised-head.glb
```

2337 verts, 4480 tris, all 51 drivable ARKit morphs, one clean root node `head`, an embedded
128×128 face map, Y-up, face on −Z. Passes `glb_inspect.py --rig-check` 51/51.

The difference from `make_arkit51_glb.js`: that one is a **format donor** — a sphere whose 51
"expressions" are deterministic bumps, perfect for unblocking a loader and proving nothing about
anatomy. This one deforms the **correct region in the correct direction**: `jawOpen` drops the jaw,
`eyeBlinkLeft` closes the subject's left eyelid, `mouthPucker` pushes the lips along −Z. So the
tracker hookup can be verified by eye, not just by name.

**What it is not:** finished character art, and not a likeness of any real person — deliberately.
The project rule is stylised only, never a photoreal face-matched likeness of a real individual.
Proportions are parametric, there is no hair geometry and no sculpted detail. It is a high-quality
placeholder that exercises the whole pipeline while real art is made in VRoid Studio or Ready
Player Me — then run the same validator against that.

`SUBJECT_LEFT` is declared once at the top of the file and every paired morph is keyed off it, so
the left/right convention is stated in exactly one place and cannot drift.

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

`--rig-check` answers the question names cannot: **where does each morph actually move the mesh?**
A model can carry all 51 correct names and still blink the wrong eye — the single most common defect
in an ARKit rig. For every `*Left`/`*Right` pair it reports the displacement centroid and fails the
run if the subject's left shape deforms −X. Verified against a deliberately mirrored build: 36
paired morphs flagged, exit 1. Compressed or sparse morphs are skipped, never failed.

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
