# `fbx_to_glb` — Mixamo asset pipeline

Mixamo (as of 2026) only ships character + animation downloads in
**FBX** format. Bevy's `bevy_gltf` only reads **glTF / GLB**. This tool
converts Mixamo FBX → GLB headlessly via Blender, preserving the
skeleton, skin, and animation tracks. AAA-quality defaults; one
command per character class.

## Prerequisites

- **Blender 4.x** (3.6+ may also work). Install from
  [blender.org/download](https://www.blender.org/download/) or your
  package manager.
- The wrapper script auto-detects:
  - `blender` on PATH
  - `/Applications/Blender.app/Contents/MacOS/Blender` (macOS bundle)
  - Flatpak `org.blender.Blender` (Linux)

## Workflow

### 1. Download Mixamo FBX files

For the `human_default` character class:

1. Sign in at [mixamo.com](https://www.mixamo.com/).
2. Download the **Y Bot** character: search the character library,
   *Use This Character*, then *Download* with **Format: FBX Binary
   (.fbx)**, **Pose: T-pose**.  Save as `Y_Bot.fbx`.
3. For each animation in the table below, search the clip name on
   Mixamo, click *Download* with these settings:
   - **Format**: FBX Binary (.fbx)
   - **Skin**: Without Skin (animation only — saves bandwidth)
   - **Frames per Second**: 30
   - **Keyframe Reduction**: none
   - **In Place**: ON for Walking / Running (so root motion doesn't
     slide the character — the body translation comes from the server)
4. Rename each download per the table — the importer matches on these
   filenames.

| Mixamo clip name (search term)     | Save as                      |
|------------------------------------|------------------------------|
| "Idle"                             | `Idle.fbx`                   |
| "Walking" (in-place)               | `Walking.fbx`                |
| "Running" (in-place)               | `Running.fbx`                |
| "Jumping Up" (in-place)            | `Jumping_Up.fbx`             |
| "Falling Idle"                     | `Falling_Idle.fbx`           |
| "Sitting Idle"                     | `Sitting_Idle.fbx`           |
| "Standing Turn 90 Left"            | `Standing_Turn_90_Left.fbx`  |
| "Standing Turn 90 Right"           | `Standing_Turn_90_Right.fbx` |

Place all 9 files in any directory — referred to as
`<MIXAMO_DOWNLOAD_DIR>` below. Suggested layout:

```
~/Downloads/mixamo/
├── Y_Bot.fbx
├── Idle.fbx
├── Walking.fbx
└── …
```

### 2. Convert to GLB

```bash
tools/fbx_to_glb/import_class.sh human_default ~/Downloads/mixamo
```

The importer:
1. Reads `manifest.human_default.txt` (the source-name → output-path
   mapping).
2. Verifies every required FBX is present in the source directory.
   If anything is missing, lists the missing names and bails BEFORE
   converting anything (no partial state).
3. Runs Blender once per file to convert. Total runtime ~10–20 s for
   the 9 files.
4. Writes outputs into `client/assets/characters/human_default/`
   (creating sub-directories as needed).

When it finishes the `client/assets/characters/human_default/` tree
matches what `client/assets/characters/README.md` documents and the
client's loader will succeed on next launch.

### 3. Adding a new character class

1. Define the class in `core/src/character/class.rs` (a new
   `CharacterClass` const + an entry in `ALL_CLASSES`).
2. Create `tools/fbx_to_glb/manifest.<class_name>.txt` with the
   per-clip source/destination mapping. Use
   `manifest.human_default.txt` as a template.
3. Drop the Mixamo FBX downloads into a directory and run
   `import_class.sh <class_name> <download_dir>`.

No converter / loader code changes.

## Single-file conversion (debugging)

If you need to convert just one file (e.g. a swap-out replacement
animation):

```bash
tools/fbx_to_glb/convert.sh \
    ~/Downloads/Idle.fbx \
    client/assets/characters/human_default/animations/idle.glb
```

## What the converter does (under the hood)

`convert.py` is a headless Blender script. Per file it:
- Clears the default scene.
- Imports the FBX with Mixamo-friendly options
  (`automatic_bone_orientation=True`, `ignore_leaf_bones=False`,
  `use_anim=True`).
- Exports as GLB with skin + all animation tracks
  (`export_skins=True`, `export_animations=True`, `export_apply=True`,
  `export_yup=True`).
- Disables morph targets (Mixamo rigs don't have any) to keep the GLB
  lean.

Mixamo FBX is in cm units; Blender's importer auto-converts to metres,
which matches every other engine constant in the project.
