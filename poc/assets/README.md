# Vendored assets

`canonical_face_model.obj` — MediaPipe's canonical face mesh: 468 vertices, 898 triangles, and the
UV layout that `facepaint.py` textures against. Vertex indices match the first 468 landmarks the
Face Landmarker emits, which is what makes a texture drop straight onto a tracked face.

It is **not shipped in the `mediapipe` pip wheel**, so it is vendored here. Source (Apache 2.0):

    https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj
