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
| 2 | Live webcam detection | **In progress** — steps 2.1–2.3 done (camera input, live skeleton, smoothing) |
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
python scripts/download_model.py --model lite   # optional: faster model for live mode (~6 MB)
pytest                              # should print: 83 passed
```

All commands below assume the virtual environment is active and you are in the
`cv-dance-challenge` folder.

### Using an iPhone as the camera (live mode)

Live mode works with any camera Windows can see. An iPhone gives a much better
picture than most laptop webcams. All the processing still happens on the
laptop in Python; the phone only sends video.

1. **Install a phone-as-webcam app** on both the iPhone (App Store) and Windows
   (its desktop/driver app): for example **Camo**, **iVCam** or **DroidCam**.
2. **Connect over USB** (more reliable and lower lag than Wi-Fi). Plug the phone in,
   unlock it, tap *Trust this computer* if asked, and open the app on both sides.
   The Windows app should show the phone's picture.
3. **Use the back camera** (better quality) and set the app to 720p or 1080p.
4. **Find the camera number:** close the phone app's preview window if it holds the
   camera, then run
   ```powershell
   python scripts/list_cameras.py
   ```
   The phone usually appears as an extra camera after the built-in webcam (e.g. `1`
   or `2`), often at 1280x720 or 1920x1080. If the laptop has no other camera
   (or it's disabled), the phone is camera `0`. For each number the script tries
   every backend/format and says which gives a **PICTURE**.
   Keep the phone app's Windows program running: it feeds the virtual camera.
5. **Run live mode** with that number:
   ```powershell
   python scripts/live.py --source 1 --debug
   ```
   - Picture sideways (phone held upright)? Add `--rotate 90` (or `270`).
   - Black screen while the phone app shows a picture? Live mode already retries
     other backends/formats when it sees only black frames; you can force one with
     `--backend msmf` or `--backend dshow` (use what `list_cameras.py` marked PICTURE).
   - Wi-Fi apps that give a URL instead of a camera (e.g. DroidCam's
     `http://<phone-ip>:4747/video`) also work: `--source http://...`.

**Mirroring.** The live display is mirrored by default so it behaves like a mirror:
raise your right hand and it goes up on the right of the screen. The iPhone **back
camera is not mirrored**, so the default looks right with it. If your app mirrors
the picture itself (some do for the front/selfie camera), turn mirroring **off in
the phone app**, not with `--no-mirror`: pose detection needs the real,
un-mirrored picture to tell your left from your right.

## How to run each script

### `scripts/download_model.py`
Downloads the MediaPipe PoseLandmarker model into `models/` if it isn't there yet.
`--model lite` downloads the smaller, faster model used by `live.py --model lite`.

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

### `scripts/list_cameras.py` — find your camera / iPhone
```powershell
python scripts/list_cameras.py               # tries cameras 0-5
python scripts/list_cameras.py --max-index 9
```
Opens each camera number briefly and prints which ones work and at what resolution.

### `scripts/live.py` — live mode (Phase 2, in progress)
```powershell
python scripts/live.py --source 1                    # camera 1 (e.g. iPhone app)
python scripts/live.py --source http://IP:4747/video # a stream URL
python scripts/live.py --source clip.mp4             # a video file
```
| Flag | What it does |
|---|---|
| `--source` | Camera number, `http://`/`https://`/`rtsp://` stream URL, or video file (default `0`) |
| `--backend {auto,dshow,msmf}` | Camera backend. `auto` tries DirectShow (with MJPG, then the camera's native format), then Media Foundation, until frames aren't black |
| `--camera-size WxH` | Resolution to ask a camera for (default 1280x720); the startup line shows what it actually delivers |
| `--rotate {0,90,180,270}` | Rotate frames clockwise before anything else (phone upright, picture sideways) |
| `--no-mirror` | Show the picture un-mirrored |
| `--debug` | Start with the debug overlay: joint labels (L/R), source, camera vs detection resolution, mirroring, model, people found, timings, camera delivery FPS |
| `--model {full,lite}` | Pose model: `full` (default, more accurate) or `lite` (faster; download it first) |
| `--detect-size PX` | Longest side of the frame copy used for detection (default 640) |
| `--verbose` | Show MediaPipe/OpenCV log lines |

Keys: **Q** or **Esc** quit, **D** toggles the debug overlay. If the camera drops
(cable unplugged, app closed), the window shows "Camera disconnected" and retries
every 3 seconds.

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

### Phase 2 (live mode)

- **iPhone as a webcam, not a native iPhone app.** A phone-as-webcam app (Camo, iVCam,
  DroidCam) makes the iPhone look like an ordinary camera to Windows, so the same
  Python + OpenCV + MediaPipe code works with the iPhone, a laptop webcam, a USB
  webcam or a video file. A native iOS app would mean rewriting everything in Swift
  (or running models on the phone), needing a Mac and Xcode, and keeping two code
  bases in sync — far more work for no gain at this stage. The phone gives a sharper,
  faster picture; the laptop does all the computing.
- **USB preferred over Wi-Fi**: lower and steadier lag, and no dropped frames when the
  Wi-Fi is busy. Stream URLs are still supported for apps that only do Wi-Fi.
- **Rotation happens first.** `--rotate` turns the frame as soon as it's read, so
  detection, drawing and the debug overlay all see an upright picture.
- **Detection runs on a smaller copy** (longest side `DETECTION_MAX_SIDE = 640`, same
  aspect ratio, never upscaled); the display keeps full camera quality. MediaPipe
  returns landmarks as fractions (0–1) of the image, so the same fractions map straight
  back onto the full-size frame — no extra conversion that could go wrong.
- **Capture size 1280x720 requested, MJPG format.** Big enough to look sharp, small
  enough for USB to carry at 30 FPS. Cameras may deliver something else; the startup
  line prints what you actually got (`asked for ...`).
- **Camera backends on Windows: DirectShow, then Media Foundation, checked for black
  frames.** DirectShow usually opens faster. But asking a camera for MJPG when it doesn't
  support it, or using the wrong backend for a virtual camera (Camo on Windows 11 is a
  Media Foundation virtual camera), can give a camera that "opens" yet sends only black
  frames. So when opening, live mode reads up to `CAMERA_WARMUP_FRAMES = 30` frames and,
  if they're all black (brightest pixel <= `BLANK_FRAME_MAX_VALUE = 8`; a real dark room
  still has sensor noise above that), moves on to: DirectShow + MJPG, DirectShow + native
  format, Media Foundation + native format. If all are black it opens anyway (maybe the
  lens is covered) and prints a warning. `--backend` forces one.
- **Reconnecting instead of crashing.** If the camera/stream stops sending frames, the
  window shows "Camera disconnected" and retries every `RECONNECT_SECONDS = 3`. Each
  attempt runs in a background thread because opening a missing stream can block for
  several seconds (`STREAM_TIMEOUT_MS = 5000`), and the window must keep responding to Q.
  Timestamps keep increasing across reconnects (MediaPipe's video mode requires that).
- **Live detection: VIDEO mode, un-mirrored frame.** PoseLandmarker runs in VIDEO mode
  (`detect_for_video` with increasing millisecond timestamps), which tracks the person
  between frames — simpler than LIVE_STREAM's callbacks. It always sees the original,
  un-mirrored frame, so its anatomical left/right stay correct; only the displayed copy
  is flipped, and skeleton x coordinates are mirrored to match (x → 1 − x).
- **Live poses use the Phase 1 format.** Each frame's main person goes through Phase 1's
  `choose_main_person` (largest body, up to `LIVE_MAX_PEOPLE = 2` searched) and
  `landmarks_to_pose`, labelled with the *full* frame size so Phase 1's pixel-scaled maths
  works unchanged.
- **Speed (measured on the development laptop, 1280x720 test video, processing only):**

  | model | detect size 640 | 480 | 320 |
  |---|---|---|---|
  | full | 28 ms → 36 FPS | 29 ms → 35 FPS | 28 ms → 36 FPS |
  | lite | 24 ms → 42 FPS | 25 ms → 41 FPS | 24 ms → 42 FPS |

  Detection size barely matters because MediaPipe shrinks every frame to its own small
  input size internally; the model choice saves ~4 ms. Both are faster than a 30 FPS
  camera, so the defaults are `full` at 640 and the camera sets the real frame rate.
  `--model lite` is there for slower machines.
- **Camera read in a background thread, newest frame only.** A plain loop waits for the
  camera, *then* detects, so the two times add up. `LatestFrameGrabber` keeps reading in
  a thread and the loop always takes the newest frame (older ones are dropped, not
  queued, so the picture never lags behind). On a local 30 FPS test stream this raised
  processing from 26 to 29 FPS (the stream's own limit). The grabber also measures the
  camera's own delivery rate, shown in the debug panel ("camera delivers N FPS"), to tell
  a slow camera apart from slow processing. Video files are still read one frame at a
  time, in order, so no frames are skipped.
- **Smoothing: per-joint exponential moving average** (`SMOOTHING_ALPHA = 0.5`, the weight
  of the newest frame). Simple, no extra packages, and easy to reason about. Only
  confident joints (visibility >= `MIN_VISIBILITY`) enter the average; a joint that drops
  below that is cleared and restarts at its new position when it's seen again, so it
  never "slides" across the screen. No person → all history cleared and "Step into frame"
  shown. On the test video, smoothing halved frame-to-frame jitter (e.g. left wrist
  1.2 px → 0.5 px per frame) while standing still. Visibility is passed through
  unsmoothed so Phase 1's skip rules see what the detector reported. Trade-off: a lower
  alpha is calmer but lags behind fast movements; 0.5 lags ~1–2 frames at 30 FPS.
- **Hidden joints aren't drawn.** Joints below `MIN_VISIBILITY` (and bones touching them)
  are skipped so a hidden arm doesn't flail around on screen.
- **On-screen text scales with the frame's shorter side**, so it's the same size in
  landscape and portrait (upright phone) and long lines are trimmed to fit.

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
  Use `--mirror` (or `MIRROR_INPUT = True`) consistently, or not at all. In live mode, if the
  phone app mirrors the picture, turn that off in the app (see "Using an iPhone").
- **Camera names aren't shown.** OpenCV can only list cameras by number, so
  `list_cameras.py` shows numbers and resolutions, not "iPhone" or "Integrated Webcam".
- **Small or partial people.** If the person is tiny in the frame, cropped, or in silhouette,
  detection may fail or visibility may be too low to score.
