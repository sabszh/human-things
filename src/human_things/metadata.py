"""Shared labels and constants for reports and figures."""

MODEL_LABELS = {
    "baseline": "Image-only\nbaseline",
    "human_informed": "v1 human",
    "shuffled_control": "v1 shuffled",
    "v1_human": "v1 human",
    "v1_shuffled": "v1 shuffled",
    "v2_1200": "v2 current\nbatch",
    "v3_human": "v3 strong\nhuman",
    "v3_shuffled": "v3 shuffled",
    "joint_matrix": "joint matrix",
    "joint_matrix_shuffled": "joint matrix\nshuffled",
}

MODEL_ORDER = [
    "baseline",
    "human_informed",
    "shuffled_control",
    "v1_human",
    "v1_shuffled",
    "v2_1200",
    "v3_human",
    "v3_shuffled",
    "joint_matrix",
    "joint_matrix_shuffled",
]

MODEL_COLORS = {
    "baseline": "#4B5563",
    "human_informed": "#2563EB",
    "v1_human": "#2563EB",
    "shuffled_control": "#93C5FD",
    "v1_shuffled": "#93C5FD",
    "v2_1200": "#059669",
    "v3_human": "#DC2626",
    "v3_shuffled": "#FCA5A5",
    "joint_matrix": "#0F766E",
    "joint_matrix_shuffled": "#99F6E4",
}

FIG_BG = "#fbfcfe"
GRID_COLOR = "#CBD5E1"
TEXT_COLOR = "#111827"
MUTED_TEXT = "#475569"
