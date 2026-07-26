"""CLI for image background replacement."""

import argparse

from src.inference import (
    load_segmentation_model,
    predict_mask_from_path,
    replace_background,
    save_mask,
    save_rgb_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace the background in one image using a segmentation model."
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--background",
        required=True,
        help="Path to the replacement background image.",
    )
    parser.add_argument("--model", required=True, help="Path to the .keras or .h5 model.")
    parser.add_argument("--output", required=True, help="Path for the output image.")
    parser.add_argument(
        "--mask-output",
        default=None,
        help="Optional path for saving the predicted binary mask as PNG.",
    )
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = load_segmentation_model(args.model)
    image_rgb, mask = predict_mask_from_path(
        model,
        args.image,
        image_size=args.image_size,
        threshold=args.threshold,
    )
    result_rgb = replace_background(image_rgb, mask, args.background)
    save_rgb_image(result_rgb, args.output)
    if args.mask_output:
        save_mask(mask, args.mask_output)
        print(f"Mask saved to: {args.mask_output}")

    print(f"Result saved to: {args.output}")


if __name__ == "__main__":
    main()
