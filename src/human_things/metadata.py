"""Shared labels and constants for reports and figures."""

MODEL_ALIASES = {
    "human_informed": "fixed_prototype_triplets",
    "shuffled_control": "fixed_prototype_control",
    "v1_human": "fixed_prototype_triplets",
    "v1_shuffled": "fixed_prototype_control",
    "v2_1200": "batch_prototype_triplets",
    "v3_human": "high_pressure_triplets",
    "v3_shuffled": "high_pressure_control",
    "joint_matrix": "joint_matrix_alignment",
    "joint_matrix_shuffled": "matrix_control",
}

MODEL_LABELS = {
    "baseline": "Image-only\nclassifier",
    "fixed_prototype_triplets": "Fixed-prototype\ntriplets",
    "fixed_prototype_control": "Fixed-prototype\ncontrol",
    "batch_prototype_triplets": "Batch-prototype\ntriplets",
    "high_pressure_triplets": "High-pressure\ntriplets",
    "high_pressure_control": "High-pressure\ncontrol",
    "joint_matrix_alignment": "Joint matrix\nalignment",
    "matrix_control": "Matrix\ncontrol",
}
for old, new in MODEL_ALIASES.items():
    MODEL_LABELS[old] = MODEL_LABELS[new]

MODEL_ORDER = [
    "baseline",
    "fixed_prototype_triplets",
    "fixed_prototype_control",
    "batch_prototype_triplets",
    "high_pressure_triplets",
    "high_pressure_control",
    "joint_matrix_alignment",
    "matrix_control",
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
    "fixed_prototype_triplets": "#2563EB",
    "fixed_prototype_control": "#B8C0CC",
    "batch_prototype_triplets": "#059669",
    "high_pressure_triplets": "#DC2626",
    "high_pressure_control": "#B8C0CC",
    "joint_matrix_alignment": "#0F766E",
    "matrix_control": "#B8C0CC",
}
for old, new in MODEL_ALIASES.items():
    MODEL_COLORS[old] = MODEL_COLORS[new]

CONTROL_MODELS = {
    "fixed_prototype_control",
    "high_pressure_control",
    "matrix_control",
    "shuffled_control",
    "v1_shuffled",
    "v3_shuffled",
    "joint_matrix_shuffled",
}

FIG_BG = "#fbfcfe"
GRID_COLOR = "#CBD5E1"
TEXT_COLOR = "#111827"
MUTED_TEXT = "#475569"
