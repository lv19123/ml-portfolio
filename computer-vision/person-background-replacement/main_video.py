"""CLI for video background replacement."""

import argparse

from src.inference import load_segmentation_model
from src.video import replace_video_background


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace the background in a video using a segmentation model."
    )
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument(
        "--background",
        required=True,
        help="Path to the replacement background image.",
    )
    parser.add_argument("--model", required=True, help="Path to the .keras or .h5 model.")
    parser.add_argument("--output", required=True, help="Path for the output video.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Model input image size. Default: 256.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Mask threshold in the range [0, 1]. Default: 0.5.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit for processed frames. Default: process the full video.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = load_segmentation_model(args.model)
    replace_video_background(
        model,
        video_path=args.video,
        background_path=args.background,
        output_path=args.output,
        image_size=args.image_size,
        threshold=args.threshold,
        max_frames=args.max_frames,
    )

    print(f"Video saved to: {args.output}")


if __name__ == "__main__":
    main()
