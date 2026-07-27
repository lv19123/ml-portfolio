"""Image inference helpers for person segmentation and background replacement."""

from pathlib import Path

import numpy as np


def _import_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for image inference. "
            "Install dependencies from requirements.txt first."
        ) from exc
    return cv2


def load_segmentation_model(model_path: str):
    """Load a saved Keras segmentation model with custom losses and metrics."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Segmentation model not found: {model_path}")

    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is required to load the segmentation model. "
            "Install dependencies from requirements.txt first."
        ) from exc

    from src.losses import bce_dice_loss, dice_coef, dice_loss, iou_metric

    custom_objects = {
        "dice_coef": dice_coef,
        "Custom>dice_coef": dice_coef,
        "dice_loss": dice_loss,
        "Custom>dice_loss": dice_loss,
        "bce_dice_loss": bce_dice_loss,
        "Custom>bce_dice_loss": bce_dice_loss,
        "iou_metric": iou_metric,
        "Custom>iou_metric": iou_metric,
    }
    return tf.keras.models.load_model(str(path), custom_objects=custom_objects)


def preprocess_image(image_rgb: np.ndarray, image_size: int = 256) -> np.ndarray:
    """Resize and normalize an RGB image for model prediction."""
    if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
        raise ValueError("image_rgb must be an RGB image with shape (H, W, 3)")

    cv2 = _import_cv2()
    resized = cv2.resize(image_rgb, (image_size, image_size))
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)


def predict_mask_from_array(
    model,
    image_rgb: np.ndarray,
    image_size: int = 256,
    threshold: float = 0.5,
) -> np.ndarray:
    """Predict a binary person mask for an RGB image array."""
    if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
        raise ValueError("image_rgb must be an RGB image with shape (H, W, 3)")

    height, width = image_rgb.shape[:2]
    batch = preprocess_image(image_rgb, image_size=image_size)
    prediction = model.predict(batch, verbose=0)

    mask = np.asarray(prediction)
    if mask.ndim == 4:
        mask = mask[0, :, :, 0]
    elif mask.ndim == 3:
        mask = mask[0]
    else:
        raise ValueError(f"Unexpected model prediction shape: {mask.shape}")

    binary_mask = (mask >= threshold).astype(np.uint8)
    cv2 = _import_cv2()
    resized_mask = cv2.resize(
        binary_mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized_mask.astype(np.uint8)


def predict_mask_from_path(
    model,
    image_path: str,
    image_size: int = 256,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Load an image from disk and predict its binary person mask."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    cv2 = _import_cv2()
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read input image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask = predict_mask_from_array(
        model,
        image_rgb,
        image_size=image_size,
        threshold=threshold,
    )
    return image_rgb, mask


def replace_background(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    background_path: str,
) -> np.ndarray:
    """Replace image background using a binary 0/1 mask."""
    if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
        raise ValueError("image_rgb must be an RGB image with shape (H, W, 3)")
    if mask is None or mask.ndim != 2:
        raise ValueError("mask must be a 2D binary array")

    path = Path(background_path)
    if not path.exists():
        raise FileNotFoundError(f"Background image not found: {background_path}")

    cv2 = _import_cv2()
    background_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if background_bgr is None:
        raise ValueError(f"Could not read background image: {background_path}")

    height, width = image_rgb.shape[:2]
    background_rgb = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB)
    background_rgb = cv2.resize(background_rgb, (width, height))

    if mask.shape != (height, width):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    person_mask = (mask == 1)[..., np.newaxis]
    return np.where(person_mask, image_rgb, background_rgb).astype(image_rgb.dtype)


def save_rgb_image(image_rgb: np.ndarray, output_path: str) -> None:
    """Save an RGB image to disk with OpenCV."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    cv2 = _import_cv2()
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    saved = cv2.imwrite(str(output), image_bgr)
    if not saved:
        raise ValueError(f"Could not save output image: {output_path}")


def save_mask(mask: np.ndarray, output_path: str) -> None:
    """Save a binary 0/1 mask as an 8-bit PNG image."""
    if mask is None or mask.ndim != 2:
        raise ValueError("mask must be a 2D binary array")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    cv2 = _import_cv2()
    mask_png = (mask > 0).astype(np.uint8) * 255
    saved = cv2.imwrite(str(output), mask_png)
    if not saved:
        raise ValueError(f"Could not save mask image: {output_path}")
