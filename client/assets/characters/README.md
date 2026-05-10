# Character assets

Skeletal humanoid asset pipeline for the client. One directory per
**character class** (race / body type); the loader reads
`voxeldust_core::character::ALL_CLASSES` and enqueues every class
registered there.

## Layout

```
client/assets/characters/
├── README.md                       (this file)
└── <class_name>/
    ├── <skeleton>.glb              base mesh + skin + skeleton
    └── animations/
        ├── idle.glb
        ├── walk.glb
        ├── run.glb
        ├── jump.glb
        ├── falling.glb
        ├── sit_chair.glb
        ├── turn_l_90.glb
        └── turn_r_90.glb
```

The exact path strings are declared per-class in
`core/src/character/class.rs` (`asset_path`, `clips`); changing a
filename is one edit there. The loader fails fast and prints the
missing paths if any file is absent.

## Character classes

### `human_default` — Mixamo Y-bot (Phase B target)

The default humanoid uses Mixamo's free **Y-bot** rig (~22 bones) so
every Mixamo animation works without retargeting.

#### How to populate `human_default/`

Mixamo (as of 2026) only ships downloads in **FBX** format; Bevy reads
**GLB**. The repo includes a headless Blender converter at
`tools/fbx_to_glb/` that handles this in one command.

1. **Install Blender 4.x** from
   [blender.org/download](https://www.blender.org/download/) if you
   don't have it already.
2. **Sign in to [mixamo.com](https://www.mixamo.com/)** (free Adobe ID).
3. **Download the Y-bot character + every animation** listed in the
   table below as **FBX Binary (.fbx)**. Save them all into one
   directory (suggested: `~/Downloads/mixamo/`). Use the exact
   filenames in the right column — the converter reads its mapping
   from `tools/fbx_to_glb/manifest.human_default.txt`.

   Per-clip Mixamo download settings:
   - **Format**: FBX Binary (.fbx)
   - **Skin**: *With Skin* for the character (`Y_Bot.fbx`),
     *Without Skin* for every animation
   - **Frames per Second**: 30
   - **Keyframe Reduction**: none
   - **In Place**: ON for `Walking`, `Running` (root motion stays at
     the origin — body translation comes from the server)

   | Mixamo clip (search term)       | Save as                      |
   |----------------------------------|------------------------------|
   | "Y Bot" character                | `Y_Bot.fbx`                  |
   | "Idle" (the basic standing one)  | `Idle.fbx`                   |
   | "Walking" (in-place)             | `Walking.fbx`                |
   | "Running" (in-place)             | `Running.fbx`                |
   | "Jumping Up" (in-place)          | `Jumping_Up.fbx`             |
   | "Falling Idle"                   | `Falling_Idle.fbx`           |
   | "Sitting Idle"                   | `Sitting_Idle.fbx`           |
   | "Standing Turn 90 Left"          | `Standing_Turn_90_Left.fbx`  |
   | "Standing Turn 90 Right"         | `Standing_Turn_90_Right.fbx` |

4. **Run the converter** from the repo root:
   ```bash
   tools/fbx_to_glb/import_class.sh human_default ~/Downloads/mixamo
   ```
   It validates every file is present BEFORE starting (so a missing
   clip never produces a half-state asset tree), then converts each
   one (~1–2 s per file). Output lands directly in
   `client/assets/characters/human_default/`.

If any Mixamo clip name has shifted between releases, picking any clip
with the same gait pattern is fine — the rig is shared across every
Mixamo character, so all animations retarget cleanly.

See `tools/fbx_to_glb/README.md` for the full converter docs (single-
file conversion, debugging, error handling).

### Adding a new class

1. Define a new `CharacterClass` const in
   `core/src/character/class.rs`. Reuse `HUMAN_DEFAULT` as a template
   for the look-limit / locomotion / bone-name fields; the only
   per-class fields that always differ are `id`, `name`,
   `asset_path`, and `clips`.
2. Add a reference to the new const in the `ALL_CLASSES` slice.
3. Create `tools/fbx_to_glb/manifest.<new_class_name>.txt` with the
   per-clip source-FBX → output-GLB mapping (use
   `manifest.human_default.txt` as a template).
4. Run `tools/fbx_to_glb/import_class.sh <new_class_name>
   <mixamo_download_dir>`.

The client picks up the new class on next launch — no loader / render
code changes.

## License notes

Mixamo's standard license permits commercial and non-commercial use
with no royalty obligations (see [Mixamo FAQ — General Use Terms](
https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html)). Do
*not* commit Mixamo source files (`.fbx`) here — only the redistributable
`.glb` exports. Mixamo source files live under `art/characters/` (not
in this client/assets tree).
