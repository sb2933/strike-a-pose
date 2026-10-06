# Computer Vision Dance Challenge

A Python computer-vision game that scores how closely your body pose matches a
target dance pose. It detects a skeleton in a photo with MediaPipe, compares
joint angles and joint positions to a target, and gives you a score out of 100,
per-body-part feedback ("Left arm: ⚠ too low (elbow off by 18°)") and a
side-by-side figure.

The project is planned in 6 phases:

| Phase | What | Status |
|---|---|---|
| 1 | Static pose matching from photos | **Done** (this README) |
| 2 | Live webcam detection | placeholder `src/dance_challenge/live/` |
| 3 | Movement sequences | placeholder `src/dance_challenge/movement/` |
| 4 | Advanced scoring | — |
| 5 | Game system | placeholder `src/dance_challenge/game/` |
| 6 | Professional dance video input | placeholder `src/dance_challenge/reference_video/` |

## Setup

Requires **Python 3.10–3.12** (MediaPipe does not support every newer Python
version yet; this was built and tested with Python 3.11 on Windows).

```powershell
cd cv-dance-challenge
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .          # makes `dance_challenge` importable
python scripts/download_model.py    # fetches models/pose_landmarker_full.task (~9 MB)
pytest                              # should print: 43 passed
```

All commands below assume the virtual environment is active and you are in the
`cv-dance-challenge` folder.

## How to run each script

### `scripts/download_model.py`
Downloads the MediaPipe PoseLandmarker model into `models/` if it isn't there yet.

### `scripts/detect_pose.py` — steps 1.1 and 1.2
```powershell
python scripts/detect_pose.py data/raw/my_photo.jpg               # skeleton overlay -> outputs/my_photo_skeleton.png
python scripts/detect_pose.py data/raw/my_photo.jpg --save-json   # also outputs/my_photo.json (13 joints)
```
If there is no person in the photo it prints a message instead of crashing.

### `scripts/create_target.py` — step 1.3
```powershell
python scripts/create_target.py data/raw/star_jump.jpg --name star_jump   # -> data/targets/star_jump.json
python scripts/create_target.py --synthetic                              # rebuild arms_out, reach_up, squat
python scripts/create_target.py --list                                   # list available targets
```
Creating a target from a photo also saves `outputs/<name>_target_overlay.png`.
Open it to confirm the right person was detected.

### `scripts/compare_pose.py` — steps 1.4–1.9 end to end
```powershell
python scripts/compare_pose.py --target reach_up --user data/user/me.jpg
python scripts/compare_pose.py --target arms_out --user-json data/targets/reach_up.json   # no photo needed
python scripts/compare_pose.py --list
```
It prints every joint angle ("Left elbow: target 90°, you 105°, off by 15°"),
the overall / upper-body / lower-body scores and per-body-part feedback, then
saves `outputs/compare_<target>_vs_<you>.png`. When you use `--user`, your
detected pose is also saved to `outputs/<photo name>.json`.

It also saves two overlay images so you can check the right person was used:
`outputs/<target>_target_overlay.png` and `outputs/<you>_user_overlay.png`.
The chosen person is drawn in white/red; anyone else detected is drawn in grey.
Warnings are printed if more than one person was found, if average joint
visibility is low, or if a joint is at the image edge (possibly cut off).

### `scripts/calibrate.py` — tuning the scoring
```powershell
python scripts/calibrate.py --target mypose                 # every photo in data/user/
python scripts/calibrate.py --target mypose --folder some/other/folder
```
Prints one row per photo: overall, upper, lower, angle score, position score.
Run it before and after changing a threshold in `config.py` to see the effect.

### Common flags
- `--verbose` (all scripts): show MediaPipe/TensorFlow's own INFO/warning
  log lines, which are hidden by default. Use it when something goes wrong.
- `--mirror` (detect/create/compare): flip the photo before detection.

Example output (synthetic poses):

```
Score: 73/100
  Upper body: 43   Lower body: 100
Feedback
  Left arm: ✗ too high (shoulder off by 81°)
  Right arm: ✗ too high (shoulder off by 81°)
  Left leg: ✓
  Right leg: ✓
  Torso: ✓
```

## How it works

```
photo ─► pose_detector ─► landmarks (13 joints, JSON) ─┬─► angles     (8 joint angles)
                                                       └─► normalize ─► positions (distance per joint)
                                                                   └─► scoring ─► feedback ─► visualize
```

| Module | Job |
|---|---|
| `config.py` | Joint list, bones, angle definitions, thresholds, weights, `MIRROR_INPUT` |
| `pose_detector.py` | Load image, run MediaPipe `PoseLandmarker` (up to 3 people), pick the largest, draw the skeleton |
| `photo.py` | Photo → pose + warnings, and the overlay images the scripts save |
| `quality.py` | Photo-quality warnings (low visibility, joints at the image edge) |
| `logs.py` | Hide MediaPipe's native log lines unless `--verbose` |
| `landmarks.py` | Keep 13 joints, save/load/validate the pose JSON format |
| `targets.py` | Create targets from photos, list/load targets, build synthetic targets |
| `angles.py` | 8 angles (elbows, shoulders, hips, knees) in degrees, target vs user |
| `normalize.py` | Hip midpoint → (0, 0), torso length → 1 |
| `positions.py` | Distance per joint in torso lengths |
| `scoring.py` | Scores, visibility handling, upper/lower-body sub-scores |
| `feedback.py` | ✓ / ⚠ / ✗ per body part with direction hints |
| `visualize.py` | Matplotlib side-by-side figure |

**Pose JSON format** (targets and user poses):

```json
{
  "name": "reach_up",
  "source_image": "data/raw/reach_up.jpg",
  "image_width": 1080,
  "image_height": 1920,
  "landmarks": {
    "left_elbow": {"x": 0.42, "y": 0.31, "visibility": 0.98}
  }
}
```
`x`, `y` are MediaPipe's normalized image coordinates (0–1, y increases
downward). Synthetic targets also have `"synthetic": true`.

**Scoring** (all numbers live in `config.py`):
- Per-angle score: 100 up to 10° off, then linear down to 0 at 60° or more.
- Per-joint position score: 100 up to 0.1 torso lengths, then linear down to 0 at 0.4 or more.
- Overall = 0.6 × mean angle score + 0.4 × mean position score, rounded to 0–100.
- Joints with visibility < 0.5 in either pose are skipped, as is any angle using them.
  Fewer than 8 usable joints → "pose not fully visible" instead of a score.

## Phase 1 checklist

- [x] 1.1 Detect a skeleton on one photo (overlay saved; "no person" handled)
- [x] 1.2 Turn the skeleton into numbers (13-joint JSON, `--save-json`)
- [x] 1.3 Create target poses (from photos + 3 synthetic: arms_out, reach_up, squat)
- [x] 1.4 Compare joint angles (90° / 180° / self-compare tests; terminal report)
- [x] 1.5 Normalize coordinates (shift + 2× scale invariant to 1e-6)
- [x] 1.6 Compare joint positions (raised arm only moves that elbow + wrist)
- [x] 1.7 Score (identical = 100; reach_up vs arms_out lower; squat low on Lower Body)
- [x] 1.8 Per-body-part feedback (lowered arm → "too low", others ✓)
- [x] 1.9 Visual result (figure + terminal report; full test suite passes)

## Design decisions

- **Python 3.11.** It was installed, is inside the 3.10–3.12 range, and MediaPipe 1.0.1 ships wheels for it.
- **MediaPipe Tasks API (`PoseLandmarker`)**, "full" model, single-image mode. The
  legacy `mp.solutions.pose` doesn't exist in MediaPipe 1.x.
- **`opencv-contrib-python` instead of `opencv-python`.** MediaPipe already depends on the contrib
  build; installing both puts two copies of `cv2` in the same place and they can break each other.
- **Left/right = the person's own (anatomical) side**, as MediaPipe labels it. Facing the camera,
  your left arm appears on the *right* of the image. `MIRROR_INPUT` (in `config.py`, default
  `False`, or `--mirror` on the scripts) flips the image before detection, for mirrored selfies.
  Don't mirror only one of target/user, or left and right will swap.
- **Angles are computed in pixels** (x·width, y·height) so tall phone photos don't distort them.
- **Undefined angles** (two joints at exactly the same point) are skipped, not treated as 0°.
- **Shoulders and hips must be visible.** Normalization needs them, so if any of the four is
  below the visibility threshold the result is "pose not fully visible", even with 8+ joints.
- **Upper/Lower Body sub-scores** use the same 0.6/0.4 mix, over the angles *centred* on those
  joints (elbows + shoulders, or hips + knees) and the positions of those joints.
- **Per-joint colour in the figure** = the joint's position score, mixed 0.6/0.4 with the angle
  score at that joint when it has one (elbows, shoulders, hips, knees).
- **Body-part groups:** arm = elbow + wrist positions, elbow + shoulder angles; leg = knee +
  ankle positions, knee + hip angles; torso = nose, shoulders, hips positions.
- **Feedback hints appear only when a part isn't ✓**, and only for clear differences
  (≥ 0.15 torso lengths in height, ≥ 10° in angle) so you don't get nagged about noise.
  Bend hints name the joint ("straighten the knee more") because the "off by" number
  in brackets is the *largest* error in that part, which may be a different joint.
- **Height hints use the wrist** (or the elbow if the wrist isn't visible).
- **"Clearly lower" for reach_up vs arms_out** means below the ✓ band (73/100) with Upper Body
  at 43. Only the arms differ, so the legs keep the overall score from collapsing. That
  follows from the specified formula; nothing was tuned to make it lower.
- **Synthetic skeleton:** a pretend 1000 × 1400 px photo, torso 250 px, each arm bone 120 px.
  The squat has thighs out sideways and shins vertical, so hips and knees are exactly 90° in
  2D (a front-view squat with thighs pointing at the camera can't be shown in 2D).
- **Detected user poses are saved** to `outputs/<photo>.json`, so you can re-compare with
  `--user-json` without re-running detection.
- **Status colours** (green `#0ca30c`, yellow `#fab219`, red `#d03b3b`) also use different
  marker shapes (●, ▲, ✕) so the figure can be read without relying on colour alone.

### Changes after testing with real photos

- **Scoring tolerances (calibrated).** Small errors are now free: angles score 100 up to
  10° off, then fall linearly to 0 at 60° (was: 0° → 45°). Positions score 100 up to
  0.1 torso lengths, then fall to 0 at 0.4 (was: 0 → 0.5). The 0.5 cutoff was
  re-checked and *tightened* to 0.4: with a tolerance added, 0.5 let the "wrong" photos
  creep up toward 40, and 0.4 keeps them lower while the close copy still scores high.
  Calibration against target `mypose` (`scripts/calibrate.py`):

  | photo | before | after |
  |---|---|---|
  | goodone.png (close copy) | 76 | 89 |
  | good.jpg (hands + legs deliberately different) | 26 | 37 |
  | wrong.jpg (wrong pose) | 24 | 30 |

  Tuned on only three photos, so treat the numbers as a starting point and re-run
  `calibrate.py` as you add more. No existing tests changed: the synthetic test poses
  differ by 45–90°, far beyond the tolerances. New tests cover the tolerance itself.
- **Up to 3 people, use the largest.** The detector looks for up to `MAX_PEOPLE = 3`
  people and uses the one with the biggest bounding box (clipped to the image). If more
  than one is found, a warning names which one (left/centre/right of the photo).
  Biggest-body is a simple rule that matches "the person posing is closest to the camera".
- **Photo-quality warnings** (don't change the score): average visibility of the 13 joints
  below `MIN_AVG_VISIBILITY = 0.7`, or any joint within `EDGE_MARGIN = 2%` of the edge.
- **Overlays for target and user** are saved on every `compare_pose.py` and
  `create_target.py` run. A target's overlay is redrawn from its JSON onto its source
  photo, so the JSON now records `"mirrored": true` when `--mirror` was used.
- **No background removal.** Scoring only uses joint coordinates, so segmenting out a TV or
  sofa wouldn't change any score; the real risk is the *wrong person* being picked, which
  the multi-person check and overlays address.
- **Every ⚠/✗ line has a direction.** New hints: shoulder ("lift the arm away from your
  side" / "bring the arm closer to your side"), hip ("bend more at the hip" / "open up at
  the hip"), foot spread ("step the foot wider" / "bring the foot in"), and torso lean
  and turn. If none of them apply, the part falls back to its worst angle or worst joint,
  e.g. "move the left wrist up and toward the photo's right". At most 2 hints per line.
  Sideways directions say "the photo's left/right" rather than "your left/right" so they
  can't be misread when the person faces away from the camera.
- **MediaPipe log lines hidden.** They come from C++ code that writes straight to stderr,
  so Python's `logging` can't silence them; `logs.py` points stderr at the null device
  only while MediaPipe runs. `--verbose` turns this off. Python errors still show.
- **Personal photos are gitignored** (`data/raw/*`, `data/user/*`) so they are never pushed.

## Known limitations

- **2D only.** MediaPipe's depth (z) is ignored, so an arm pointing at the camera looks
  short, and a front-on photo can't tell a forward bend from a sideways one.
- **Single person scored.** Only the largest person in the photo is scored. MediaPipe can
  also *miss* small people in the background entirely (so no warning appears); the
  overlay image is the reliable check.
- **Crossed legs.** In poses like cross-legged sitting, knees and ankles overlap in 2D, so
  leg angles and positions are less reliable than for standing poses.
- **Sensitive to camera angle.** Target and user photos should be taken from roughly the same
  viewpoint (e.g. both straight-on, full body, similar height). A side-on photo compared to a
  front-on target will score badly even with a perfect pose.
- **Body proportions affect position scores.** Positions are scaled by torso length only, so
  someone with longer arms relative to their torso loses a few position points on the wrists
  even with matching angles.
- **Mirroring.** Selfie/front-camera apps often mirror the image, which swaps left and right.
  Use `--mirror` (or `MIRROR_INPUT = True`) consistently, or not at all.
- **Small or partial people.** If the person is tiny in the frame, cropped, or in silhouette,
  detection may fail or visibility may be too low to score.
