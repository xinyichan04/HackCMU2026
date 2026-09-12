#!/usr/bin/env python3
"""glb_inspect.py - headless acceptance check for an avatar .glb.

Companion to verify_glb_threejs.html, not a replacement. That page is the real
bar ("loads in three.js, all 51 names resolve"); this runs the same checks with
no browser, so it works in CI, over SSH, and on a machine with no GPU.

Checks, in the order they matter:

  1. ARKit-51 name parity   - AVATAR3D-HANDOFF.md section 3, after normalising
                              the _L/_R vs camelCase conventions
  2. targetNames survival   - names actually reached mesh.extras.targetNames
  3. Non-zero morphs        - each target displaces geometry (a zero-filled
                              morph passes a name check and proves nothing)
  4. Head bone / clean root - head rotation can be driven apart from the face
  5. Triangle budget        - under ~100k
  6. Textures embedded      - what .glb buys over .gltf
  7. extensionsRequired     - meshopt/KTX2/Draco need extra loaders wired up
                              before three.js will open the file at all

Usage:
    python3 poc/tools/glb_inspect.py <model.glb> [--json] [--quiet]
    python3 poc/tools/glb_inspect.py <model.glb> --max-tris 60000

Exit status is 0 when every blocking check passes, 1 otherwise, so it drops
straight into CI. Warnings alone do not fail the run.

Requires: pygltflib  (pip install pygltflib)
"""

import argparse
import json
import re
import struct
import sys
from pathlib import Path

# AVATAR3D-HANDOFF.md section 3. ARKit's 52 minus tongueOut, which MediaPipe
# never emits. _neutral is MediaPipe's own and is not a morph target.
ARKIT_51 = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]

# Loaders three.js will not open without extra setup wired up first.
NEEDS_EXTRA_LOADER = {
    "EXT_meshopt_compression": "MeshoptDecoder",
    "KHR_texture_basisu": "KTX2Loader",
    "KHR_draco_mesh_compression": "DRACOLoader",
}

HEAD_BONE_RE = re.compile(r"(^|[^a-z])head([^a-z]|$)", re.IGNORECASE)


def canon(name):
    """Fold a morph name to a convention-independent key.

    ARKit shapes appear both as Apple's `mouthSmileLeft` and as the `_L`/`_R`
    suffix Blender, Unreal and Live Link emit. Comparing raw strings scores
    facecap.glb 15/51 and rejects a perfectly good model, so every comparison
    in this file goes through here first.
    """
    n = name.strip()
    n = re.sub(r"[_.\- ]?([LR])$", lambda m: "left" if m.group(1) == "L" else "right", n)
    return re.sub(r"[^a-z0-9]", "", n.lower())


CANON_TO_ARKIT = {canon(n): n for n in ARKIT_51}


def _accessor_count(gltf, idx):
    return gltf.accessors[idx].count if idx is not None and idx < len(gltf.accessors) else 0


def _displaces(gltf, blob, idx):
    """Does this morph POSITION accessor move any vertex?

    Returns True / False, or None when it cannot be judged (compressed or
    sparse data) - undecidable is never reported as a failure.

    Prefers the accessor's declared min/max, which the spec requires and which
    is both authoritative and far cheaper than walking every vertex.
    """
    acc = gltf.accessors[idx]
    if acc.sparse is not None:
        return None  # sparse stores only the changed vertices; not judged here
    if acc.min and acc.max:
        return any(abs(v) > 1e-7 for v in list(acc.min) + list(acc.max))
    if acc.bufferView is None or acc.componentType != 5126 or acc.type != "VEC3":
        return None
    bv = gltf.bufferViews[acc.bufferView]
    start = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    stride = bv.byteStride or 12
    for i in range(acc.count):
        off = start + i * stride
        if off + 12 > len(blob):
            return None
        if any(abs(c) > 1e-7 for c in struct.unpack_from("<3f", blob, off)):
            return True
    return False


def inspect(path, max_tris=100000):
    from pygltflib import GLTF2

    r = {
        "file": str(path), "ok": False,
        "errors": [], "warnings": [], "notes": [],
    }

    # Pick the parser by magic, not by extension: .vrm is glTF binary too.
    with open(path, "rb") as fh:
        magic = fh.read(4)
    gltf = GLTF2().load_binary(str(path)) if magic == b"glTF" else GLTF2().load(str(path))
    try:
        blob = gltf.binary_blob() or b""
    except Exception:
        blob = b""

    # --- 7. extensions that block loading outright -----------------------
    required = list(gltf.extensionsRequired or [])
    r["extensionsRequired"] = required
    for ext in required:
        if ext in NEEDS_EXTRA_LOADER:
            r["warnings"].append(
                f"requires {ext} - three.js needs {NEEDS_EXTRA_LOADER[ext]} wired up or it will not load"
            )

    # --- 1/2. morph target names ----------------------------------------
    found, anonymous_targets = [], 0
    for mesh in gltf.meshes or []:
        names = (mesh.extras or {}).get("targetNames") if isinstance(mesh.extras, dict) else None
        for prim in mesh.primitives or []:
            n_targets = len(prim.targets or [])
            if not n_targets:
                continue
            if names:
                found.extend(names[:n_targets])
            else:
                anonymous_targets += n_targets

    r["morphTargetNames"] = found
    r["morphTargetCount"] = len(found) + anonymous_targets

    if anonymous_targets:
        r["errors"].append(
            f"{anonymous_targets} morph target(s) have no name - mesh.extras.targetNames was "
            "dropped on export, so three.js sees anonymous indices and the rig cannot key by name"
        )

    canon_found = {canon(n): n for n in found}
    matched = {a: canon_found[c] for c, a in CANON_TO_ARKIT.items() if c in canon_found}
    missing = [n for n in ARKIT_51 if n not in matched]

    r["arkit"] = {
        "matched": len(matched), "required": len(ARKIT_51),
        "missing": missing,
        "extra": sorted(n for c, n in canon_found.items() if c not in CANON_TO_ARKIT),
    }

    # Which spelling did the file use? Useful when a hookup misbehaves.
    exact = sum(1 for a, actual in matched.items() if a == actual)
    if matched:
        r["arkit"]["convention"] = (
            "apple" if exact == len(matched)
            else "suffix_LR" if exact == 0 else "mixed"
        )
        if exact != len(matched):
            r["notes"].append(
                f"names use the _L/_R convention ({len(matched) - exact} of {len(matched)}); "
                "matched after normalising - key by name through the same normalisation at runtime"
            )

    if missing:
        r["errors"].append(f"missing {len(missing)} of 51 ARKit morph targets: {', '.join(missing[:8])}"
                           + (" ..." if len(missing) > 8 else ""))

    # --- 3. morphs that actually move geometry ---------------------------
    # Judged per target index across the WHOLE mesh, never per primitive.
    # glTF requires every primitive in a mesh to declare the same number of
    # targets, so a body primitive that does not deform for "jawOpen" carries
    # a legitimately all-zero target. Flagging those would reject good models.
    dead = []
    for mesh in gltf.meshes or []:
        names = (mesh.extras or {}).get("targetNames") if isinstance(mesh.extras, dict) else None
        n_targets = max((len(p.targets or []) for p in mesh.primitives or []), default=0)
        for i in range(n_targets):
            verdict = False
            for prim in mesh.primitives or []:
                tgt = (prim.targets or [None] * n_targets)[i] if i < len(prim.targets or []) else None
                pos = None
                if tgt is not None:
                    pos = tgt.get("POSITION") if isinstance(tgt, dict) else getattr(tgt, "POSITION", None)
                if pos is None:
                    continue
                moved = _displaces(gltf, blob, pos)
                if moved is None:
                    verdict = None  # undecidable anywhere -> do not judge
                    break
                if moved:
                    verdict = True
                    break
            if verdict is False:
                dead.append(names[i] if names and i < len(names) else f"#{i}")

    # Only a target we actually drive can block the run. facecap.glb carries a
    # flat tongueOut, which is correct: MediaPipe never emits it, so it is
    # outside the 51 and failing a good model for it would be a false positive.
    r["deadMorphs"] = dead
    dead_required = [n for n in dead if canon(n) in CANON_TO_ARKIT]
    dead_other = [n for n in dead if canon(n) not in CANON_TO_ARKIT]
    if dead_required:
        r["errors"].append(
            f"{len(dead_required)} drivable morph target(s) displace nothing anywhere in the mesh "
            f"({', '.join(dead_required[:6])}" + (" ..." if len(dead_required) > 6 else "")
            + ") - these pass a name check but do nothing at runtime"
        )
    if dead_other:
        r["notes"].append(
            f"{len(dead_other)} flat target(s) outside the drivable 51 ({', '.join(dead_other[:6])})"
            " - harmless, nothing drives them"
        )

    # --- 4. head bone -----------------------------------------------------
    node_names = [n.name for n in (gltf.nodes or []) if n.name]
    head = [n for n in node_names if HEAD_BONE_RE.search(n)]
    r["headNode"] = head[0] if head else None
    r["hasSkin"] = bool(gltf.skins)
    if not head:
        if len(gltf.nodes or []) == 1:
            r["notes"].append("no head bone, but a single clean root - head rotation can drive the root")
        else:
            r["warnings"].append(
                "no node matching /head/ - head pose cannot be applied separately from the face"
            )

    # --- 5. triangle budget ----------------------------------------------
    tris = 0
    for mesh in gltf.meshes or []:
        for prim in mesh.primitives or []:
            if (prim.mode or 4) != 4:
                continue
            if prim.indices is not None:
                tris += _accessor_count(gltf, prim.indices) // 3
            else:
                attrs = prim.attributes
                pos = attrs.POSITION if hasattr(attrs, "POSITION") else None
                tris += _accessor_count(gltf, pos) // 3
    r["tris"] = tris
    if tris > max_tris:
        r["errors"].append(f"{tris} triangles exceeds the {max_tris} budget")

    # --- 6. embedded textures --------------------------------------------
    external = [im.uri for im in (gltf.images or []) if im.uri and not im.uri.startswith("data:")]
    r["images"] = len(gltf.images or [])
    r["externalImages"] = external
    if external:
        r["errors"].append(
            f"{len(external)} texture(s) reference external files ({', '.join(external[:3])}) - "
            "a .glb must embed them or it breaks once moved"
        )
    if r["images"] > 2:
        r["warnings"].append(f"{r['images']} textures; the spec asks for 1-2")

    r["ok"] = not r["errors"]
    return r


def render(r, quiet=False):
    ok = r["ok"]
    print(f"{'PASS' if ok else 'FAIL'}  {Path(r['file']).name}")
    a = r.get("arkit", {})
    print(f"  ARKit morphs   {a.get('matched', 0)}/{a.get('required', 51)}"
          f"   convention={a.get('convention', 'n/a')}"
          f"   total targets={r.get('morphTargetCount', 0)}")
    print(f"  head node      {r.get('headNode') or '-'}"
          f"   skin={'yes' if r.get('hasSkin') else 'no'}")
    print(f"  triangles      {r.get('tris', 0)}")
    print(f"  textures       {r.get('images', 0)} embedded={not r.get('externalImages')}")
    if r.get("extensionsRequired"):
        print(f"  extensions     {', '.join(r['extensionsRequired'])}")
    for e in r["errors"]:
        print(f"  ERROR   {e}")
    for w in r["warnings"]:
        print(f"  WARN    {w}")
    if not quiet:
        for n in r["notes"]:
            print(f"  note    {n}")
        extra = r.get("arkit", {}).get("extra") or []
        if extra:
            shown = ", ".join(extra[:10]) + (" ..." if len(extra) > 10 else "")
            print(f"  note    {len(extra)} non-ARKit target(s): {shown}")


def main():
    ap = argparse.ArgumentParser(description="Headless ARKit-51 acceptance check for a .glb")
    ap.add_argument("model", type=Path)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--quiet", action="store_true", help="errors and warnings only")
    ap.add_argument("--max-tris", type=int, default=100000)
    args = ap.parse_args()

    if not args.model.exists():
        print(f"no such file: {args.model}", file=sys.stderr)
        return 2
    try:
        r = inspect(args.model, max_tris=args.max_tris)
    except ImportError:
        print("pygltflib is required:  pip install pygltflib", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"could not read {args.model.name}: {e}", file=sys.stderr)
        return 2

    print(json.dumps(r, indent=2)) if args.json else render(r, quiet=args.quiet)
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
