"""Headless Blender script: Mixamo FBX → glTF Binary (.glb).

Invoked by `convert.sh`; not intended to be run directly.

Defaults match the AAA-quality character pipeline:
  - Mixamo skeleton preserved (`automatic_bone_orientation` keeps the
    `mixamorig:` bone hierarchy intact for our class-defined bone-name
    lookups).
  - All animation tracks exported (`export_animations=True`).
  - Skin / armature exported (`export_skins=True`).
  - Materials exported (`export_materials='EXPORT'`) — Mixamo body
    glb's diffuse texture, the Y-bot tone, etc.
  - GLB binary (`export_format='GLB'`) — single-file, embedded
    textures; matches what `bevy_gltf` expects from
    `Handle<Scene>::load`.
  - Y-up (`export_yup=True`) — glTF spec orientation; bevy_gltf reads
    Y-up directly.

Mixamo source FBX is in cm units (scale 100). Blender's FBX importer
applies a unit conversion automatically; the resulting glb is in
metres, matching every other engine constant in the project.
"""

import bpy
import os
import sys


def _arg_after_dashdash():
    """Return the args after the standalone `--` Blender uses to separate
    its own args from the script's args."""
    argv = sys.argv
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1:]


def _clear_scene():
    """Wipe the default cube + camera + light. We rebuild the scene
    from the imported FBX every run so the converter is idempotent."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _import_mixamo_fbx(path):
    """Import a Mixamo FBX (character or animation-only)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"input FBX does not exist: {path}")
    bpy.ops.import_scene.fbx(
        filepath=path,
        # Mixamo bones export with the `mixamorig:` prefix; this option
        # preserves their hierarchy + roll without re-orienting (which
        # would break later glTF skin imports in bevy_gltf).
        automatic_bone_orientation=True,
        # Keep all bones, including end-effector leaves — our class
        # `fp_cull_bones` may reference any of them.
        ignore_leaf_bones=False,
        use_anim=True,
        use_custom_normals=True,
        # Reset emptys to identity rotation; Mixamo sometimes ships a
        # 90° X rotation on the root that bevy_gltf would re-apply.
        use_image_search=True,
    )


def _export_glb(path):
    """Export the current scene as glTF Binary (.glb)."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        export_animations=True,
        export_skins=True,
        export_apply=True,
        export_yup=True,
        # No morph targets in Mixamo rigs — disable to keep the glb
        # lean and avoid bevy_gltf issuing `MorphTargetsCount` warnings.
        export_morph=False,
    )


def main():
    args = _arg_after_dashdash()
    if len(args) != 2:
        sys.stderr.write(
            "convert.py: expected exactly 2 positional args after --\n"
            "  usage: blender --background --python convert.py -- INPUT.fbx OUTPUT.glb\n"
        )
        sys.exit(2)
    input_fbx, output_glb = args

    _clear_scene()
    _import_mixamo_fbx(input_fbx)
    _export_glb(output_glb)

    print(f"convert.py: OK  {input_fbx} -> {output_glb}")


if __name__ == "__main__":
    main()
