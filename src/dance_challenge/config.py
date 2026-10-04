"""Shared settings for the dance challenge.

Everything you might want to tweak (joints, thresholds, weights) lives here so
the rest of the code never hard-codes a magic number.
"""

from pathlib import Path

# --- Paths -----------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
MODELS_DIR: Path = PROJECT_ROOT / "models"
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
USER_DIR: Path = DATA_DIR / "user"
TARGETS_DIR: Path = DATA_DIR / "targets"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

# --- Model -----------------------------------------------------------------

MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MODEL_PATH: Path = MODELS_DIR / "pose_landmarker_full.task"

# --- Input handling --------------------------------------------------------

# If True, flip the image horizontally before detection. Leave this False for
# normal photos. Turn it on if your photos come from a mirrored front camera
# and you want "left" to mean the side that *looks* left on screen.
MIRROR_INPUT: bool = False

# --- Joints ----------------------------------------------------------------

# The 13 joints we keep, mapped to their MediaPipe landmark index.
# Left/right are the person's own (anatomical) left and right.
JOINTS: dict[str, int] = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}
JOINT_NAMES: list[str] = list(JOINTS.keys())

# Lines drawn between joints when plotting a skeleton.
BONES: list[tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

# Each angle is measured at the middle joint of the three: (a, vertex, c).
ANGLES: dict[str, tuple[str, str, str]] = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder": ("right_elbow", "right_shoulder", "right_hip"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
}

# Joints counted towards the Upper Body / Lower Body sub-scores.
UPPER_BODY_JOINTS: list[str] = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]
LOWER_BODY_JOINTS: list[str] = [
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# Body-part groups used for feedback: which joints and angles belong to each.
BODY_PARTS: dict[str, dict[str, list[str]]] = {
    "left_arm": {
        "joints": ["left_elbow", "left_wrist"],
        "angles": ["left_elbow", "left_shoulder"],
    },
    "right_arm": {
        "joints": ["right_elbow", "right_wrist"],
        "angles": ["right_elbow", "right_shoulder"],
    },
    "left_leg": {
        "joints": ["left_knee", "left_ankle"],
        "angles": ["left_knee", "left_hip"],
    },
    "right_leg": {
        "joints": ["right_knee", "right_ankle"],
        "angles": ["right_knee", "right_hip"],
    },
    "torso": {
        "joints": ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"],
        "angles": [],
    },
}

# --- Scoring ---------------------------------------------------------------

MIN_VISIBILITY: float = 0.5        # joints below this are skipped
MIN_VISIBLE_JOINTS: int = 8        # fewer than this -> "pose not fully visible"

ANGLE_ZERO_SCORE_DEG: float = 45.0     # angle error at which the score hits 0
POSITION_ZERO_SCORE_DIST: float = 0.5  # distance (torso lengths) at which score hits 0

ANGLE_WEIGHT: float = 0.6
POSITION_WEIGHT: float = 0.4

# --- Feedback --------------------------------------------------------------

GOOD_SCORE: float = 80.0   # >= this -> ✓ / green
OKAY_SCORE: float = 50.0   # >= this -> ⚠ / yellow, otherwise ✗ / red

# Differences smaller than these are not worth a direction hint.
DIRECTION_MIN_Y_DIFF: float = 0.15     # torso lengths
DIRECTION_MIN_ANGLE_DIFF: float = 10.0  # degrees
