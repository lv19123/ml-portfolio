"""Command-line demonstration for the selectable OpenCV face detectors."""

import argparse
from pathlib import Path
import sys

from .face_cropper import extract_face_from_document
from .face_detector import DEFAULT_YUNET_MODEL_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract the largest face from a document photograph."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input document image.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path where the cropped face will be saved.",
    )
    parser.add_argument(
        "--detector",
        choices=("haar", "yunet"),
        default="yunet",
        help="Face detector to use (default: yunet).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional path to the YuNet ONNX model.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.9,
        help="YuNet confidence threshold from 0 to 1 (default: 0.9).",
    )
    parser.add_argument(
        "--auto-rotate",
        action="store_true",
        help="Check clockwise rotations 0, 90, 180, and 270 degrees.",
    )
    parser.add_argument(
        "--padding-left",
        type=float,
        default=0.30,
        help="Left padding relative to face width (default: 0.30).",
    )
    parser.add_argument(
        "--padding-right",
        type=float,
        default=0.30,
        help="Right padding relative to face width (default: 0.30).",
    )
    parser.add_argument(
        "--padding-top",
        type=float,
        default=0.40,
        help="Top padding relative to face height (default: 0.40).",
    )
    parser.add_argument(
        "--padding-bottom",
        type=float,
        default=0.50,
        help="Bottom padding relative to face height (default: 0.50).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        result = extract_face_from_document(
            input_path=args.input,
            output_path=args.output,
            padding_left=args.padding_left,
            padding_right=args.padding_right,
            padding_top=args.padding_top,
            padding_bottom=args.padding_bottom,
            detector=args.detector,
            yunet_model_path=args.model_path,
            score_threshold=args.score_threshold,
            auto_rotate=args.auto_rotate,
        )
    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        RuntimeError,
        OSError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    x, y, width, height = result.detected_box
    expanded_x, expanded_y, expanded_width, expanded_height = (
        result.expanded_box
    )
    crop_height, crop_width = result.crop_shape

    print(f"Detector: {result.detector_name}")
    print(f"Auto-rotate: {'enabled' if args.auto_rotate else 'disabled'}")
    checked_angles = "0°, 90°, 180°, 270°" if args.auto_rotate else "0°"
    print(f"Checked rotations: {checked_angles}")
    print(f"Selected rotation: {result.rotation_angle}° clockwise")
    print(f"Input image: {Path(args.input)}")
    if result.detector_name == "yunet":
        model_path = (
            DEFAULT_YUNET_MODEL_PATH
            if args.model_path is None
            else args.model_path.expanduser()
        )
        print(f"YuNet model: {model_path}")
        print(f"YuNet score threshold: {args.score_threshold:.2f}")
    print(f"Faces detected: {result.faces_count}")
    print(
        "Original face box (x, y, width, height): "
        f"{x}, {y}, {width}, {height}"
    )
    print(
        "Expanded crop box (x, y, width, height): "
        f"{expanded_x}, {expanded_y}, {expanded_width}, {expanded_height}"
    )
    print(
        "Padding ratios (left, right, top, bottom): "
        f"{args.padding_left:.2f}, {args.padding_right:.2f}, "
        f"{args.padding_top:.2f}, {args.padding_bottom:.2f}"
    )
    print(f"Saved crop size (width x height): {crop_width} x {crop_height}")
    print(f"Detection time: {result.detection_time_ms:.3f} ms")
    print(f"Saved output: {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
