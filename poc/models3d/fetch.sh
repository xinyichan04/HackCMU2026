#!/usr/bin/env bash
# Test avatars for the 3D route. Binaries are gitignored - re-fetch them with this.
set -euo pipefail
cd "$(dirname "$0")"
get(){ [ -f "$2" ] || curl -sSfL --max-time 60 -o "$2" "$1"; echo "  $2"; }

get "https://threejs.org/examples/models/gltf/RobotExpressive/RobotExpressive.glb" RobotExpressive.glb
get "https://threejs.org/examples/models/gltf/facecap.glb" facecap.glb
get "https://raw.githubusercontent.com/pixiv/three-vrm/dev/packages/three-vrm/examples/models/VRM1_Constraint_Twist_Sample.vrm" vrm-sample.vrm
echo "done - see README.md for what each one is for"
