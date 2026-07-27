"""Video inference helpers for frame-by-frame background replacement."""

from pathlib import Path

import numpy as np

from src.inference import predict_mask_from_array


def _import_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for video inference. "
            "Install dependencies from requirements.txt first."
        ) from exc
    return cv2


def blend_frame_with_background(frame_rgb, mask, background_rgb):
    """Replace a single frame background using a binary 0/1 mask."""
    if frame_rgb is None or frame_rgb.ndim != 3 or frame_rgb.shape[-1] != 3:
        raise ValueError("frame_rgb must be an RGB frame with shape (H, W, 3)")
    if background_rgb is None or background_rgb.ndim != 3 or background_rgb.shape[-1] != 3:
        raise ValueError("background_rgb must be an RGB image with shape (H, W, 3)")
    if mask is None or mask.ndim != 2:
        raise ValueError("mask must be a 2D binary array")

    height, width = frame_rgb.shape[:2]
    cv2 = _import_cv2()

    if background_rgb.shape[:2] != (height, width):
        background_rgb = cv2.resize(background_rgb, (width, height))
    if mask.shape != (height, width):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    person_mask = (mask == 1)[..., np.newaxis]
    return np.where(person_mask, frame_rgb, background_rgb).astype(frame_rgb.dtype)


def replace_video_background(
    model,
    video_path: str,
    background_path: str,
    output_path: str,
    image_size: int = 256,
    threshold: float = 0.5,
    max_frames: int | None = None,
) -> None:
    """Replace a video background frame by frame using a segmentation model."""
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    background = Path(background_path)
    if not background.exists():
        raise FileNotFoundError(f"Background image not found: {background_path}")

    if max_frames is not None and max_frames < 0:
        raise ValueError("max_frames must be None or a non-negative integer")

    cv2 = _import_cv2()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {video_path}")

    writer = None
    processed_frames = 0

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError(f"Could not read video size: {video_path}")

        background_bgr = cv2.imread(str(background), cv2.IMREAD_COLOR)
        if background_bgr is None:
            raise ValueError(f"Could not read background image: {background_path}")

        background_rgb = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB)
        background_rgb = cv2.resize(background_rgb, (width, height))

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise ValueError(f"Could not create output video: {output_path}")

        while True:
            if max_frames is not None and processed_frames >= max_frames:
                break

            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mask = predict_mask_from_array(
                model,
                frame_rgb,
                image_size=image_size,
                threshold=threshold,
            )
            result_rgb = blend_frame_with_background(frame_rgb, mask, background_rgb)
            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
            writer.write(result_bgr)
            processed_frames += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    print(f"Processed {processed_frames} frame(s).")
