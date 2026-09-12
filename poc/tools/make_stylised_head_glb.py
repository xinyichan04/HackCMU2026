#!/usr/bin/env python3
"""make_stylised_head_glb.py - a stylised, riggable head with all 51 ARKit morphs.

WHAT THIS IS
    A stylised anime-proportioned head: large eyes, soft jaw, dark hair cap.
    Every one of the 51 drivable ARKit morph targets (AVATAR3D-HANDOFF.md s3)
    deforms the anatomically correct region, so when the tracker fires jawOpen
    the jaw drops and when it fires eyeBlinkLeft the subject's left eyelid
    closes. That is the difference between this and a sphere with bumps: the
    rig can be verified by eye, not just by name.

WHAT THIS IS NOT
    Not a likeness of any real person, and deliberately so. The project rule is
    stylised only - never a photoreal face-matched likeness of a real
    individual. This is also not finished character art: proportions are
    parametric, there is no hair geometry and no sculpted detail. Treat it as a
    high-quality placeholder that exercises the whole pipeline end to end while
    real art is produced in VRoid Studio / Ready Player Me, then run the same
    validator against that.

CONVENTIONS (AVATAR3D-HANDOFF.md s5)
    Y-up, face pointing -Z, textures embedded, one primitive, ~4.5k triangles.
    Left/Right are the SUBJECT's. With the face on -Z and +Y up, the subject's
    left is +X - get this backwards and blinks land on the wrong eye, which is
    the classic symptom. Every paired morph below is keyed off SUBJECT_LEFT so
    the convention is stated once and cannot drift.

Usage:
    python3 poc/tools/make_stylised_head_glb.py poc/models3d/stylised-head.glb
    python3 poc/tools/make_stylised_head_glb.py out.glb --rings 56 --segments 72

Requires: pygltflib   (same dependency as glb_inspect.py)
"""

import argparse
import json
import math
import struct
import sys
import zlib
from pathlib import Path

SUBJECT_LEFT = +1.0  # +X is the subject's left when the face looks down -Z

# ---------------------------------------------------------------- geometry ---


def head_vertex(u, v, rings, segs):
    """One vertex of a head-shaped surface, parameterised over a sphere.

    u: 0..1 around (0 = back, 0.5 = front), v: 0..1 top to bottom.
    Shaping is deliberately parametric and gentle - this is a stylised head,
    not a scan.
    """
    theta = u * 2.0 * math.pi
    phi = v * math.pi
    x = math.sin(phi) * math.sin(theta)
    y = math.cos(phi)
    z = -math.sin(phi) * math.cos(theta)  # z<0 at u=0.5 -> face looks down -Z

    frontness = max(0.0, -z)          # 1 at the face, 0 at the back
    downness = max(0.0, -y)           # 1 at the chin

    # Stylised proportions: narrow, tall cranium; tapered chin; flatter back.
    x *= 0.86 - 0.16 * downness * downness
    z *= 0.94 + 0.10 * frontness
    if y < 0:                          # chin taper
        x *= 1.0 - 0.30 * downness
        z *= 1.0 - 0.12 * downness
    y *= 1.12
    # Brow ridge and a slightly flattened face plane.
    if frontness > 0.4 and -0.05 < y < 0.45:
        z -= 0.035 * frontness
    return [x, y, z]


def build_mesh(rings, segs):
    verts, uvs, norms, idx = [], [], [], []
    for r in range(rings + 1):
        v = r / rings
        for s in range(segs + 1):
            u = s / segs
            p = head_vertex(u, v, rings, segs)
            verts.append(p)
            uvs.append([u, v])
            n = math.sqrt(sum(c * c for c in p)) or 1.0
            norms.append([c / n for c in p])
    for r in range(rings):
        for s in range(segs):
            a = r * (segs + 1) + s
            b = a + segs + 1
            idx += [a, b, a + 1, a + 1, b, b + 1]
    return verts, uvs, norms, idx


# ----------------------------------------------------------------- morphs ---
# Each entry: (centre, sigma, direction, magnitude). A gaussian falloff around
# the centre pushes vertices along the direction. Mirrored shapes are generated
# from one definition so left/right can never disagree.

EYE_L = (0.30 * SUBJECT_LEFT, 0.17, -0.80)
BROW_L = (0.30 * SUBJECT_LEFT, 0.36, -0.78)
CHEEK_L = (0.44 * SUBJECT_LEFT, -0.14, -0.70)
MOUTH_C = (0.0, -0.34, -0.92)
MOUTH_L = (0.17 * SUBJECT_LEFT, -0.34, -0.88)
NOSE_C = (0.0, -0.08, -1.00)
JAW_C = (0.0, -0.72, -0.70)

# name -> (centre, sigma, direction, magnitude)
PAIRED = {
    # eyes
    "eyeBlink":     (EYE_L, 0.15, (0, -1, 0), 0.055),
    "eyeSquint":    (EYE_L, 0.16, (0, -0.6, -0.4), 0.030),
    "eyeWide":      (EYE_L, 0.17, (0, 0.8, -0.2), 0.032),
    "eyeLookUp":    (EYE_L, 0.13, (0, 0.5, -0.5), 0.020),
    "eyeLookDown":  (EYE_L, 0.13, (0, -0.5, -0.5), 0.020),
    "eyeLookIn":    (EYE_L, 0.13, (-SUBJECT_LEFT, 0, -0.4), 0.020),
    "eyeLookOut":   (EYE_L, 0.13, (SUBJECT_LEFT, 0, -0.4), 0.020),
    # brows
    "browDown":     (BROW_L, 0.20, (0, -1, -0.15), 0.045),
    "browOuterUp":  ((0.44 * SUBJECT_LEFT, 0.38, -0.72), 0.19, (0, 1, 0), 0.048),
    # cheeks
    "cheekSquint":  (CHEEK_L, 0.18, (0, 0.7, -0.5), 0.028),
    # nose
    "noseSneer":    ((0.11 * SUBJECT_LEFT, -0.02, -0.97), 0.14, (0, 0.8, -0.3), 0.026),
    # mouth, paired
    "mouthSmile":   (MOUTH_L, 0.21, (0.45 * SUBJECT_LEFT, 0.85, -0.1), 0.060),
    "mouthFrown":   (MOUTH_L, 0.21, (0.25 * SUBJECT_LEFT, -0.9, 0), 0.055),
    "mouthDimple":  ((0.26 * SUBJECT_LEFT, -0.32, -0.84), 0.15, (0, 0.3, 0.6), 0.028),
    "mouthStretch": (MOUTH_L, 0.20, (SUBJECT_LEFT, -0.15, 0.2), 0.050),
    "mouthPress":   (MOUTH_L, 0.17, (0, 0.35, 0.35), 0.026),
    "mouthLowerDown": ((0.13 * SUBJECT_LEFT, -0.41, -0.90), 0.16, (0, -1, -0.2), 0.045),
    "mouthUpperUp": ((0.13 * SUBJECT_LEFT, -0.27, -0.90), 0.16, (0, 1, -0.2), 0.042),
}

SINGLE = {
    "browInnerUp":   ((0.0, 0.37, -0.90), 0.22, (0, 1, -0.1), 0.050),
    "cheekPuff":     ((0.0, -0.20, -0.80), 0.42, (0.0, 0, -1), 0.055),
    "jawOpen":       (JAW_C, 0.55, (0, -1, 0.1), 0.115),
    "jawForward":    (JAW_C, 0.45, (0, -0.1, -1), 0.055),
    "jawLeft":       (JAW_C, 0.45, (SUBJECT_LEFT, 0, 0), 0.055),
    "jawRight":      (JAW_C, 0.45, (-SUBJECT_LEFT, 0, 0), 0.055),
    "mouthClose":    (MOUTH_C, 0.20, (0, 0.5, 0.1), 0.038),
    "mouthFunnel":   (MOUTH_C, 0.19, (0, 0, -1), 0.062),
    "mouthPucker":   (MOUTH_C, 0.16, (0, 0, -1), 0.075),
    "mouthLeft":     (MOUTH_C, 0.24, (SUBJECT_LEFT, 0, 0), 0.048),
    "mouthRight":    (MOUTH_C, 0.24, (-SUBJECT_LEFT, 0, 0), 0.048),
    "mouthRollLower": ((0.0, -0.40, -0.92), 0.15, (0, 0.4, 0.8), 0.036),
    "mouthRollUpper": ((0.0, -0.28, -0.92), 0.15, (0, -0.4, 0.8), 0.036),
    "mouthShrugLower": ((0.0, -0.44, -0.90), 0.17, (0, 0.6, -0.6), 0.034),
    "mouthShrugUpper": ((0.0, -0.26, -0.90), 0.17, (0, -0.3, -0.7), 0.034),
}


def morph_table():
    """Expand into the exact 51 ARKit names, Left/Right from one definition."""
    out = {}
    for base, (centre, sigma, direction, mag) in PAIRED.items():
        cx, cy, cz = centre
        dx, dy, dz = direction
        out[base + "Left"] = ((cx, cy, cz), sigma, (dx, dy, dz), mag)
        # Mirror across X for the subject's right.
        out[base + "Right"] = ((-cx, cy, cz), sigma, (-dx, dy, dz), mag)
    out.update(SINGLE)
    return out


def morph_deltas(verts, spec):
    (cx, cy, cz), sigma, (dx, dy, dz), mag = spec
    n = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    dx, dy, dz = dx / n, dy / n, dz / n
    two_s2 = 2.0 * sigma * sigma
    out, moved = [], False
    for (x, y, z) in verts:
        d2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
        w = math.exp(-d2 / two_s2)
        if w < 1e-4:
            out.append((0.0, 0.0, 0.0))
            continue
        k = w * mag
        out.append((dx * k, dy * k, dz * k))
        moved = True
    return out, moved


# ---------------------------------------------------------------- texture ---


def png(width, height, pixel_fn):
    """Minimal RGB PNG writer (stdlib zlib only) so the .glb can embed a map."""
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0
        for x in range(width):
            raw += bytes(pixel_fn(x / (width - 1), y / (height - 1)))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
            + chunk(b"IEND", b""))


def face_texel(u, v):
    """Stylised flat-shaded face map. u wraps (0.5 = front), v is top->bottom."""
    skin = (247, 223, 205)
    hair = (38, 30, 34)
    if v < 0.20 or (v < 0.34 and not (0.30 < u < 0.70)):
        return hair                                    # hair cap + sides
    front = 0.30 < u < 0.70
    if front and 0.34 < v < 0.42:
        return (54, 42, 40)                            # brows / fringe edge
    if front and 0.44 < v < 0.52 and (0.36 < u < 0.46 or 0.54 < u < 0.64):
        return (40, 34, 46)                            # large stylised eyes
    if front and 0.62 < v < 0.68 and 0.45 < u < 0.55:
        return (206, 122, 124)                         # lips
    return skin


# ------------------------------------------------------------------ output ---


def build(path, rings, segs):
    from pygltflib import (GLTF2, Scene, Node, Mesh, Primitive, Attributes, Accessor,
                           BufferView, Buffer, Material, PbrMetallicRoughness, Image,
                           Texture, Sampler, TextureInfo)

    verts, uvs, norms, idx = build_mesh(rings, segs)
    table = morph_table()
    assert len(table) == 51, f"expected 51 morph names, built {len(table)}"

    blob = bytearray()
    views, accessors = [], []

    def add(data, fmt, count, acc_type, comp_type, mins=None, maxs=None, target=None):
        while len(blob) % 4:
            blob.append(0)
        off = len(blob)
        for item in data:
            blob.extend(struct.pack(fmt, *item) if isinstance(item, (list, tuple))
                        else struct.pack(fmt, item))
        views.append(BufferView(buffer=0, byteOffset=off, byteLength=len(blob) - off,
                                target=target))
        accessors.append(Accessor(bufferView=len(views) - 1, componentType=comp_type,
                                  count=count, type=acc_type, min=mins, max=maxs))
        return len(accessors) - 1

    def bounds(vs):
        return ([min(v[i] for v in vs) for i in range(3)],
                [max(v[i] for v in vs) for i in range(3)])

    lo, hi = bounds(verts)
    a_pos = add(verts, "<3f", len(verts), "VEC3", 5126, lo, hi)
    a_nrm = add(norms, "<3f", len(norms), "VEC3", 5126)
    a_uv = add(uvs, "<2f", len(uvs), "VEC2", 5126)
    a_idx = add(idx, "<I", len(idx), "SCALAR", 5125, target=34963)

    targets, names, flat = [], [], []
    for name in sorted(table):                     # deterministic order
        deltas, moved = morph_deltas(verts, table[name])
        if not moved:
            flat.append(name)
        dlo, dhi = bounds(deltas)
        targets.append({"POSITION": add(deltas, "<3f", len(deltas), "VEC3", 5126, dlo, dhi)})
        names.append(name)
    if flat:
        print(f"warning: {len(flat)} morph(s) displace nothing: {', '.join(flat)}",
              file=sys.stderr)

    tex_png = png(128, 128, face_texel)
    while len(blob) % 4:
        blob.append(0)
    tex_off = len(blob)
    blob.extend(tex_png)
    views.append(BufferView(buffer=0, byteOffset=tex_off, byteLength=len(tex_png)))
    img_view = len(views) - 1

    g = GLTF2(
        scene=0,
        scenes=[Scene(nodes=[0])],
        # One clean root named "head": head pose drives this node, the 51 morphs
        # drive the face. That separation is requirement 2 of the spec.
        nodes=[Node(mesh=0, name="head")],
        meshes=[Mesh(primitives=[Primitive(
            attributes=Attributes(POSITION=a_pos, NORMAL=a_nrm, TEXCOORD_0=a_uv),
            indices=a_idx, material=0, targets=targets)],
            extras={"targetNames": names}, name="head")],
        materials=[Material(pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorTexture=TextureInfo(index=0), metallicFactor=0.0,
            roughnessFactor=0.75), name="skin", doubleSided=False)],
        textures=[Texture(source=0, sampler=0)],
        images=[Image(bufferView=img_view, mimeType="image/png")],
        samplers=[Sampler(magFilter=9729, minFilter=9987, wrapS=10497, wrapT=33071)],
        bufferViews=views,
        accessors=accessors,
        buffers=[Buffer(byteLength=len(blob))],
    )
    g.set_binary_blob(bytes(blob))
    g.save_binary(str(path))

    print(f"wrote {path}  {Path(path).stat().st_size} bytes  "
          f"{len(verts)} verts  {len(idx)//3} tris  {len(names)} morph targets")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Stylised ARKit-51 head, no 3D software needed")
    ap.add_argument("out", type=Path)
    ap.add_argument("--rings", type=int, default=40)
    ap.add_argument("--segments", type=int, default=56)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        return build(args.out, args.rings, args.segments)
    except ImportError:
        print("pygltflib is required:  pip install pygltflib", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
