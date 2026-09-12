// Emit a plain glTF 2.0 .glb with all 51 ARKit-drivable morph targets.
// stdlib only: no Blender, no glTF libs, no GUI.
const fs = require('fs');

const ARKIT51 = `browDownLeft browDownRight browInnerUp browOuterUpLeft browOuterUpRight
cheekPuff cheekSquintLeft cheekSquintRight
eyeBlinkLeft eyeBlinkRight eyeLookDownLeft eyeLookDownRight eyeLookInLeft eyeLookInRight
eyeLookOutLeft eyeLookOutRight eyeLookUpLeft eyeLookUpRight eyeSquintLeft eyeSquintRight
eyeWideLeft eyeWideRight
jawForward jawLeft jawOpen jawRight
mouthClose mouthDimpleLeft mouthDimpleRight mouthFrownLeft mouthFrownRight mouthFunnel
mouthLeft mouthLowerDownLeft mouthLowerDownRight mouthPressLeft mouthPressRight mouthPucker
mouthRight mouthRollLower mouthRollUpper mouthShrugLower mouthShrugUpper
mouthSmileLeft mouthSmileRight mouthStretchLeft mouthStretchRight
mouthUpperUpLeft mouthUpperUpRight
noseSneerLeft noseSneerRight`.split(/\s+/).filter(Boolean);

if (ARKIT51.length !== 51) throw new Error(`expected 51 names, got ${ARKIT51.length}`);

// --- UV sphere as a stand-in head, radius 0.10 m (a human head is ~0.20 m across) ---
const SEG_U = 24, SEG_V = 16, R = 0.10;
const pos = [], nrm = [], idx = [];
for (let v = 0; v <= SEG_V; v++) {
  const phi = (v / SEG_V) * Math.PI;
  for (let u = 0; u <= SEG_U; u++) {
    const theta = (u / SEG_U) * 2 * Math.PI;
    const x = Math.sin(phi) * Math.cos(theta), y = Math.cos(phi), z = Math.sin(phi) * Math.sin(theta);
    pos.push(x * R, y * R + 0.10, z * R);   // origin at neck base: head sits above y=0
    nrm.push(x, y, z);
  }
}
for (let v = 0; v < SEG_V; v++) {
  for (let u = 0; u < SEG_U; u++) {
    const a = v * (SEG_U + 1) + u, b = a + SEG_U + 1;
    idx.push(a, b, a + 1, b, b + 1, a + 1);
  }
}
const VCOUNT = pos.length / 3;

// --- 51 morph targets: each pushes a deterministic facial region, so every shape is
// visually distinguishable and none is a no-op (a zero morph passes a name check but
// proves nothing at runtime). Front of the face is +Z here.
function targetDelta(i) {
  const d = new Float32Array(VCOUNT * 3);
  const cy = 0.10 + 0.05 * Math.cos((i / 51) * Math.PI);      // vary height per shape
  const cx = 0.05 * Math.sin((i / 51) * Math.PI * 4);          // vary left/right
  for (let k = 0; k < VCOUNT; k++) {
    const x = pos[k*3], y = pos[k*3+1], z = pos[k*3+2];
    if (z <= 0) continue;                                      // face is the +Z hemisphere
    const dist = Math.hypot(x - cx, y - cy);
    const w = Math.max(0, 1 - dist / 0.055);                   // smooth falloff
    if (w > 0) { d[k*3] += 0.004*w*Math.sign(cx||1); d[k*3+1] += 0.006*w; d[k*3+2] += 0.010*w; }
  }
  return d;
}

// --- buffer assembly ---
const views = [], accessors = [], chunks = [];
let offset = 0;
function pad4(n) { return (4 - (n % 4)) % 4; }
function addAccessor(arr, TypedArray, componentType, type, extra = {}) {
  const data = new TypedArray(arr);
  const buf = Buffer.from(data.buffer, data.byteOffset, data.byteLength);
  chunks.push(buf, Buffer.alloc(pad4(buf.length)));
  const comps = { SCALAR: 1, VEC3: 3 }[type];
  const count = data.length / comps;
  const min = [], max = [];
  for (let c = 0; c < comps; c++) {
    let mn = Infinity, mx = -Infinity;
    for (let k = 0; k < count; k++) { const val = data[k*comps+c]; if (val < mn) mn = val; if (val > mx) mx = val; }
    min.push(mn); max.push(mx);
  }
  views.push({ buffer: 0, byteOffset: offset, byteLength: buf.length });
  offset += buf.length + pad4(buf.length);
  accessors.push({ bufferView: views.length - 1, componentType, count, type, min, max, ...extra });
  return accessors.length - 1;
}

const aPos = addAccessor(pos, Float32Array, 5126, 'VEC3');
const aNrm = addAccessor(nrm, Float32Array, 5126, 'VEC3');
const aIdx = addAccessor(idx, Uint16Array, 5123, 'SCALAR');
const targets = ARKIT51.map((_, i) => ({ POSITION: addAccessor(Array.from(targetDelta(i)), Float32Array, 5126, 'VEC3') }));

const gltf = {
  asset: { version: '2.0', generator: 'zylos handwritten glb writer (stdlib only)' },
  scene: 0,
  scenes: [{ nodes: [0] }],
  nodes: [
    { name: 'head', mesh: 0, children: [] },   // named head node: head rotation applies here
  ],
  meshes: [{
    name: 'head_mesh',
    primitives: [{
      attributes: { POSITION: aPos, NORMAL: aNrm },
      indices: aIdx,
      material: 0,
      targets,
    }],
    weights: new Array(51).fill(0),
    extras: { targetNames: ARKIT51 },          // the line exporters drop
  }],
  materials: [{
    name: 'skin',
    pbrMetallicRoughness: { baseColorFactor: [0.95, 0.80, 0.72, 1.0], metallicFactor: 0.0, roughnessFactor: 0.8 },
  }],
  bufferViews: views,
  accessors,
  buffers: [{ byteLength: offset }],
};

// --- GLB container: header + JSON chunk + BIN chunk ---
const bin = Buffer.concat(chunks);
let json = Buffer.from(JSON.stringify(gltf), 'utf8');
json = Buffer.concat([json, Buffer.alloc(pad4(json.length), 0x20)]);            // pad with spaces
const binPadded = Buffer.concat([bin, Buffer.alloc(pad4(bin.length), 0)]);
const header = Buffer.alloc(12);
header.writeUInt32LE(0x46546C67, 0);                                            // 'glTF'
header.writeUInt32LE(2, 4);
header.writeUInt32LE(12 + 8 + json.length + 8 + binPadded.length, 8);
function chunk(data, type) {
  const h = Buffer.alloc(8); h.writeUInt32LE(data.length, 0); h.writeUInt32LE(type, 4);
  return Buffer.concat([h, data]);
}
const out = Buffer.concat([header, chunk(json, 0x4E4F534A), chunk(binPadded, 0x004E4942)]);
const dest = process.argv[2] || 'arkit51-test-head.glb';
fs.writeFileSync(dest, out);
console.log(`wrote ${dest}  ${out.length} bytes  ${VCOUNT} verts  ${idx.length/3} tris  ${ARKIT51.length} morph targets`);
