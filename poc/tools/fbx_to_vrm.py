"""Mixamo FBX -> VRM, headless.

Usage:
    blender --background --python fbx_to_vrm.py -- <in.fbx> <out.vrm> [texture.png]

Needs the VRM Add-on for Blender (https://vrm-addon-for-blender.info) installed
or provided as a zip via VRM_ADDON_ZIP env var. Maps the standard Mixamo
skeleton (mixamorig:*) onto the VRM humanoid, reattaches the base color
texture if the Mixamo round-trip dropped it, and exports VRM 1.0.

Proven 2026-09-12 on Blender 5.2.1 + VRM addon 4.7.1 against a synthetic
Mixamo-named rig: 22/22 bones mapped, output VRM 1.0 passes the livebody
bone/expression checks (17/17 Kalidokit bones resolve).
"""
import bpy, os, sys

argv = sys.argv[sys.argv.index('--') + 1:]
IN_FBX, OUT_VRM = argv[0], argv[1]
TEXTURE = argv[2] if len(argv) > 2 else None

# --- VRM addon: enable, installing from zip if needed ---
import addon_utils
def ensure_vrm_addon():
    for mod in addon_utils.modules():
        if 'vrm' in mod.__name__.lower():
            addon_utils.enable(mod.__name__, default_set=True)
            return mod.__name__
    zip_path = os.environ.get('VRM_ADDON_ZIP')
    if not zip_path:
        raise RuntimeError('VRM addon not installed and VRM_ADDON_ZIP not set')
    bpy.ops.preferences.addon_install(filepath=zip_path)
    for mod in addon_utils.modules(refresh=True):
        if 'vrm' in mod.__name__.lower():
            addon_utils.enable(mod.__name__, default_set=True)
            return mod.__name__
    raise RuntimeError('VRM addon failed to install')
addon = ensure_vrm_addon()
print('VRM addon:', addon)

# --- clean scene, import ---
bpy.ops.wm.read_factory_settings(use_empty=True)
ensure_vrm_addon()   # factory reset disables addons enabled per-session
bpy.ops.import_scene.fbx(filepath=IN_FBX)

armature = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
meshes = [o for o in bpy.data.objects if o.type == 'MESH']
print('armature:', armature.name, '| meshes:', [m.name for m in meshes])
print('bones:', len(armature.data.bones))

# --- unit normalization: Mixamo FBX is authored in centimeters, and the round-trip
# can land the whole scene at 1/100 scale (measured: friend.vrm came out 0.018 m tall).
# A human that isn't human-sized breaks every camera/framing assumption downstream,
# so rescale to meters and BAKE it before the VRM export snapshots rest bone positions.
def scene_height():
    import mathutils
    lo, hi = float('inf'), float('-inf')
    for m in meshes:
        for c in m.bound_box:
            z = (m.matrix_world @ mathutils.Vector(c)).z
            lo, hi = min(lo, z), max(hi, z)
    return hi - lo
h = scene_height()
if h < 0.5:                      # nothing humanoid is half a meter tall — unit bug
    factor = 100.0 if 0.005 < h < 0.05 else (1.8 / h)
    for o in [armature] + meshes:
        if o.parent is None:
            o.scale = [s * factor for s in o.scale]
    bpy.ops.object.select_all(action='SELECT')
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    print(f'unit fix: height {h:.4f} m -> {scene_height():.3f} m (x{factor:g})')
else:
    print(f'height {h:.3f} m — no unit fix needed')

# --- reattach texture if materials came through bare ---
if TEXTURE:
    img = bpy.data.images.load(TEXTURE)
    for m in meshes:
        for slot in m.material_slots:
            mat = slot.material
            if not mat: continue
            mat.use_nodes = True
            bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
            has_tex = any(l.from_node.type == 'TEX_IMAGE'
                          for l in bsdf.inputs['Base Color'].links)
            if not has_tex:
                tex = mat.node_tree.nodes.new('ShaderNodeTexImage')
                tex.image = img
                mat.node_tree.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
                print('texture attached to', mat.name)

# --- humanoid mapping: Mixamo names -> VRM bone slots ---
MIXAMO = {
    'hips':'Hips', 'spine':'Spine', 'chest':'Spine1', 'upperChest':'Spine2',
    'neck':'Neck', 'head':'Head',
    'leftShoulder':'LeftShoulder', 'leftUpperArm':'LeftArm',
    'leftLowerArm':'LeftForeArm', 'leftHand':'LeftHand',
    'rightShoulder':'RightShoulder', 'rightUpperArm':'RightArm',
    'rightLowerArm':'RightForeArm', 'rightHand':'RightHand',
    'leftUpperLeg':'LeftUpLeg', 'leftLowerLeg':'LeftLeg',
    'leftFoot':'LeftFoot', 'leftToes':'LeftToeBase',
    'rightUpperLeg':'RightUpLeg', 'rightLowerLeg':'RightLeg',
    'rightFoot':'RightFoot', 'rightToes':'RightToeBase',
}
def find_bone(suffix):
    for b in armature.data.bones:
        n = b.name.split(':')[-1]        # 'mixamorig:Hips' or bare 'Hips'
        if n == suffix: return b.name
    return None

ext = armature.data.vrm_addon_extension
ext.spec_version = '1.0'
humanoid = ext.vrm1.humanoid
mapped, missing = 0, []
for vrm_slot, mixamo_suffix in MIXAMO.items():
    bone_name = find_bone(mixamo_suffix)
    if not bone_name:
        missing.append(vrm_slot); continue
    hb = getattr(humanoid.human_bones, {
        # VRM addon property names are snake_case
        'hips':'hips','spine':'spine','chest':'chest','upperChest':'upper_chest',
        'neck':'neck','head':'head',
        'leftShoulder':'left_shoulder','leftUpperArm':'left_upper_arm',
        'leftLowerArm':'left_lower_arm','leftHand':'left_hand',
        'rightShoulder':'right_shoulder','rightUpperArm':'right_upper_arm',
        'rightLowerArm':'right_lower_arm','rightHand':'right_hand',
        'leftUpperLeg':'left_upper_leg','leftLowerLeg':'left_lower_leg',
        'leftFoot':'left_foot','leftToes':'left_toes',
        'rightUpperLeg':'right_upper_leg','rightLowerLeg':'right_lower_leg',
        'rightFoot':'right_foot','rightToes':'right_toes',
    }[vrm_slot])
    hb.node.bone_name = bone_name
    mapped += 1
print(f'humanoid mapping: {mapped} mapped, missing: {missing}')
required_missing = [m for m in missing if m not in
                    ('upperChest','leftShoulder','rightShoulder','leftToes','rightToes')]
if required_missing:
    raise RuntimeError(f'required VRM bones unmapped: {required_missing}')

ext.vrm1.meta.vrm_name = os.path.splitext(os.path.basename(OUT_VRM))[0]
ext.vrm1.meta.authors.add().value = 'HackCMU 2026'

bpy.ops.export_scene.vrm(filepath=OUT_VRM)
print('exported:', OUT_VRM, os.path.getsize(OUT_VRM), 'bytes')
