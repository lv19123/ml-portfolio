"""Image loading, face selection, cropping, and saving helpers."""

from dataclasses import dataclass
from math import floor, isfinite
from numbers import Integral, Real
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .face_detector import detect_faces


class FaceNotFoundError(ValueError):
    """Raised when no face is found in any checked orientation."""


@dataclass(frozen=True)
class OrientationDetectionResult:
    """Best face detection and its selected clockwise orientation."""

    rotation_angle: int
    rotated_image: np.ndarray
    detected_box: tuple[int, int, int, int]
    faces_count: int
    detection_time_ms: float


@dataclass(frozen=True)
class ExtractionResult:
    """Metadata returned by one face extraction pipeline run."""

    output_path: Path
    detected_box: tuple[int, int, int, int]
    expanded_box: tuple[int, int, int, int]
    faces_count: int
    detector_name: str
    detection_time_ms: float
    crop_shape: tuple[int, int]
    rotation_angle: int


def _validate_bgr_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array.")
    if image.size == 0:
        raise ValueError("image must not be empty.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR image with three color channels.")


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image from a file and return it as a BGR NumPy array."""
    if not isinstance(image_path, (str, Path)):
        raise TypeError("image_path must be a string or pathlib.Path.")

    path = Path(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Input image does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"OpenCV could not read the input image: {path}")

    return image


def select_largest_face(
    faces: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    """Select the face bounding box with the largest area."""
    if not faces:
        raise ValueError("No faces were detected in the input image.")

    return max(faces, key=lambda face: face[2] * face[3])


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """Return an independent clockwise rotation by 0, 90, 180, or 270°."""
    _validate_bgr_image(image)
    if not isinstance(angle, Integral) or isinstance(angle, bool):
        raise ValueError("angle must be one of: 0, 90, 180, 270.")
    if angle == 0:
        return image.copy()
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("angle must be one of: 0, 90, 180, 270.")


def detect_face_with_orientation(
    image: np.ndarray,
    detector: str = "yunet",
    auto_rotate: bool = True,
    yunet_model_path: str | Path | None = None,
    score_threshold: float = 0.9,
) -> OrientationDetectionResult:
    """Detect the best largest face across the enabled orientations."""
    _validate_bgr_image(image)
    if not isinstance(auto_rotate, bool):
        raise TypeError("auto_rotate must be a boolean.")

    angles = (0, 90, 180, 270) if auto_rotate else (0,)
    angle_priority = {angle: index for index, angle in enumerate(angles)}
    candidates: list[
        tuple[
            float,
            int,
            np.ndarray,
            tuple[int, int, int, int],
            int,
        ]
    ] = []
    detection_time_ms = 0.0

    for angle in angles:
        rotated_image = rotate_image(image, angle)
        detection_started = perf_counter()
        faces = detect_faces(
            image=rotated_image,
            detector=detector,
            yunet_model_path=yunet_model_path,
            score_threshold=score_threshold,
        )
        detection_time_ms += (perf_counter() - detection_started) * 1000.0
        if not faces:
            continue

        detected_box = select_largest_face(faces)
        _, _, width, height = detected_box
        image_height, image_width = rotated_image.shape[:2]
        relative_area = (width * height) / (image_width * image_height)
        candidates.append(
            (
                relative_area,
                angle,
                rotated_image,
                detected_box,
                len(faces),
            )
        )

    if not candidates:
        checked_angles = ", ".join(f"{angle}°" for angle in angles)
        raise FaceNotFoundError(
            f"No faces were detected at checked rotations: {checked_angles}."
        )

    best_candidate = max(
        candidates,
        key=lambda candidate: (
            candidate[0],
            -angle_priority[candidate[1]],
        ),
    )
    _, rotation_angle, rotated_image, detected_box, faces_count = (
        best_candidate
    )
    return OrientationDetectionResult(
        rotation_angle=rotation_angle,
        rotated_image=rotated_image,
        detected_box=detected_box,
        faces_count=faces_count,
        detection_time_ms=detection_time_ms,
    )


def expand_bounding_box(
    image_shape: tuple[int, ...],
    bounding_box: tuple[int, int, int, int],
    padding_left: float = 0.30,
    padding_right: float = 0.30,
    padding_top: float = 0.40,
    padding_bottom: float = 0.50,
) -> tuple[int, int, int, int]:
    """Expand a face box by relative padding and clamp it to the image."""
    if not isinstance(image_shape, tuple) or len(image_shape) < 2:
        raise TypeError("image_shape must be a tuple with at least two values.")
    if any(
        not isinstance(dimension, Integral) or isinstance(dimension, bool)
        for dimension in image_shape
    ):
        raise TypeError("image_shape values must be integers.")
    if any(dimension <= 0 for dimension in image_shape):
        raise ValueError("image_shape dimensions must be positive.")

    if not isinstance(bounding_box, tuple) or len(bounding_box) != 4:
        raise TypeError(
            "bounding_box must be a tuple of (x, y, width, height)."
        )
    if any(
        not isinstance(value, Integral) or isinstance(value, bool)
        for value in bounding_box
    ):
        raise TypeError("bounding_box values must be integers.")

    x, y, width, height = (int(value) for value in bounding_box)
    if width <= 0 or height <= 0:
        raise ValueError("bounding_box width and height must be positive.")

    image_height = int(image_shape[0])
    image_width = int(image_shape[1])
    if (
        x >= image_width
        or y >= image_height
        or x + width <= 0
        or y + height <= 0
    ):
        raise ValueError("bounding_box must intersect the image.")

    padding_values = {
        "padding_left": padding_left,
        "padding_right": padding_right,
        "padding_top": padding_top,
        "padding_bottom": padding_bottom,
    }
    normalized_padding: dict[str, float] = {}
    for name, value in padding_values.items():
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError(f"{name} must be a real number.")
        numeric_value = float(value)
        if not isfinite(numeric_value):
            raise ValueError(f"{name} must be finite.")
        if numeric_value < 0:
            raise ValueError(f"{name} must be non-negative.")
        normalized_padding[name] = numeric_value

    # Round to the nearest pixel, with exact half-pixels rounded upward.
    left_pixels = floor(normalized_padding["padding_left"] * width + 0.5)
    right_pixels = floor(normalized_padding["padding_right"] * width + 0.5)
    top_pixels = floor(normalized_padding["padding_top"] * height + 0.5)
    bottom_pixels = floor(
        normalized_padding["padding_bottom"] * height + 0.5
    )

    x1 = max(x - left_pixels, 0)
    y1 = max(y - top_pixels, 0)
    x2 = min(x + width + right_pixels, image_width)
    y2 = min(y + height + bottom_pixels, image_height)

    expanded_width = x2 - x1
    expanded_height = y2 - y1
    if expanded_width <= 0 or expanded_height <= 0:
        raise ValueError("The expanded bounding_box is empty.")

    return x1, y1, expanded_width, expanded_height


def crop_face(
    image: np.ndarray,
    bounding_box: tuple[int, int, int, int],
    padding_left: float = 0.30,
    padding_right: float = 0.30,
    padding_top: float = 0.40,
    padding_bottom: float = 0.50,
) -> np.ndarray:
    """Return an independent crop made from a relatively expanded face box."""
    _validate_bgr_image(image)

    x, y, width, height = expand_bounding_box(
        image_shape=image.shape,
        bounding_box=bounding_box,
        padding_left=padding_left,
        padding_right=padding_right,
        padding_top=padding_top,
        padding_bottom=padding_bottom,
    )

    cropped_image = image[y : y + height, x : x + width].copy()
    if cropped_image.size == 0:
        raise ValueError("The calculated face crop is empty.")

    return cropped_image


def save_image(image: np.ndarray, output_path: str | Path) -> Path:
    """Save a BGR image and return its output path."""
    _validate_bgr_image(image)
    if not isinstance(output_path, (str, Path)):
        raise TypeError("output_path must be a string or pathlib.Path.")

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        saved = cv2.imwrite(str(path), image)
    except cv2.error as error:
        raise OSError(f"OpenCV could not save the image to: {path}") from error
    if not saved:
        raise OSError(f"Failed to save the output image: {path}")

    return path


def extract_face_from_document(
    input_path: str | Path,
    output_path: str | Path,
    padding_left: float = 0.30,
    padding_right: float = 0.30,
    padding_top: float = 0.40,
    padding_bottom: float = 0.50,
    detector: str = "yunet",
    yunet_model_path: str | Path | None = None,
    score_threshold: float = 0.9,
    auto_rotate: bool = False,
) -> ExtractionResult:
    """Detect, crop, and save the largest face found in a document image."""
    image = load_image(input_path)
    orientation_result = detect_face_with_orientation(
        image=image,
        detector=detector,
        auto_rotate=auto_rotate,
        yunet_model_path=yunet_model_path,
        score_threshold=score_threshold,
    )
    selected_face = orientation_result.detected_box
    rotated_image = orientation_result.rotated_image
    expanded_face = expand_bounding_box(
        image_shape=rotated_image.shape,
        bounding_box=selected_face,
        padding_left=padding_left,
        padding_right=padding_right,
        padding_top=padding_top,
        padding_bottom=padding_bottom,
    )
    cropped_face = crop_face(
        image=rotated_image,
        bounding_box=selected_face,
        padding_left=padding_left,
        padding_right=padding_right,
        padding_top=padding_top,
        padding_bottom=padding_bottom,
    )
    saved_path = save_image(cropped_face, output_path)

    return ExtractionResult(
        output_path=saved_path,
        detected_box=selected_face,
        expanded_box=expanded_face,
        faces_count=orientation_result.faces_count,
        detector_name=detector,
        detection_time_ms=orientation_result.detection_time_ms,
        crop_shape=(
            int(cropped_face.shape[0]),
            int(cropped_face.shape[1]),
        ),
        rotation_angle=orientation_result.rotation_angle,
    )
