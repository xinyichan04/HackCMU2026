# Test avatars for the 3D route

Binaries are gitignored (third-party assets, and one is 10 MB). Get them with `./fetch.sh`.

Every number below was read out of the actual file, not taken from a product page.

| file | tris | bones | morph targets | body | use it for |
|---|---|---|---|---|---|
| `RobotExpressive.glb` | 3.2k | 74 | 3 (`Angry`/`Sad`/`Surprised`) | full, 14 animations | **first test.** Smallest and fastest to load, cartoon, fully rigged |
| `vrm-sample.vrm` | 36k | 171 | 57, VRoid `Fcl_*` naming | full, spring bones | closest to the target look, and it has **hair physics** |
| `facecap.glb` | 8k | 13 | **all 52 ARKit shapes** | head only | testing the **face/blendshape** pipeline end to end |

Sources: the first two ship with three.js examples (`RobotExpressive` by Tomás Laulhé, CC0, modified
by Don McCurdy); the VRM is a pixiv `three-vrm` sample. Check the upstream licence before any of
this appears in something public — they are here to prove the pipeline, not to ship.

## The naming trap — read this before writing a name check

`facecap.glb` has **all 52 ARKit shapes including `tongueOut`**, but spelled
`mouthSmile_L`, not `mouthSmileLeft`:

```
facecap:  browDown_L   eyeBlink_L   mouthSmile_R   noseSneer_L
ARKit:    browDownLeft eyeBlinkLeft mouthSmileRight noseSneerLeft
```

So there are **two ARKit naming conventions in the wild** — Apple's own `Left`/`Right`, and the
`_L`/`_R` suffix that Blender, Unreal and Live Link exports use. A strict equality check against the
51 names **rejects this model**, and it is a perfectly good model: a naive check scores it 15/51
purely because `browInnerUp`, `jawOpen`, `cheekPuff` and friends happen to have no side suffix.

Normalise before comparing — `_L`/`_R` → `Left`/`Right` — and the mapping is exact and mechanical.
Same idea for VRoid's `Fcl_*`, except that one is a genuinely different set and needs a real lookup
table, not a suffix rewrite.

## What none of these give you

No model here is Chaewon, and none is both cartoon-cute *and* ARKit-named. The two
routes that export ARKit naming directly are **Ready Player Me** (`?morphTargets=ARKit` on the model
URL — needs a real avatar id, the old public demo ids 404) and **VRoid Studio + HANA**. That choice
is open; see `AVATAR3D-HANDOFF.md` §5.
