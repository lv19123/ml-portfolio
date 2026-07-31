"""Sequential directory processing and CSV reporting."""

import argparse
import csv
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
import sys

from .face_cropper import ExtractionResult, extract_face_from_document
from .face_detector import DEFAULT_YUNET_MODEL_PATH


SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
CSV_FIELDS = (
    "input_file",
    "status",
    "output_file",
    "detector_name",
    "auto_rotate",
    "rotation_angle",
    "faces_count",
    "detected_box",
    "expanded_box",
    "crop_width",
    "crop_height",
    "detection_time_ms",
    "error_type",
    "error_message",
)


@dataclass(frozen=True)
class BatchItemResult:
    input_file: str
    status: str
    output_file: str
    detector_name: str
    auto_rotate: bool
    rotation_angle: int | None
    faces_count: int | None
    detected_box: tuple[int, int, int, int] | None
    expanded_box: tuple[int, int, int, int] | None
    crop_width: int | None
    crop_height: int | None
    detection_time_ms: float | None
    error_type: str
    error_message: str


@dataclass(frozen=True)
class BatchSummary:
    total_supported: int
    successful: int
    failed: int
    skipped: int
    ignored: int
    total_detection_time_ms: float
    report_path: Path
    items: tuple[BatchItemResult, ...]


def _path_from_argument(value: str | Path, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{name} must be a string or pathlib.Path.")
    return Path(value).expanduser()


def _validate_batch_configuration(
    detector: str,
    auto_rotate: bool,
    model_path: Path,
    score_threshold: float,
    padding_values: dict[str, float],
    overwrite: bool,
) -> None:
    if detector not in {"yunet", "haar"}:
        raise ValueError(
            f"Unknown detector {detector!r}; expected 'haar' or 'yunet'."
        )
    if not isinstance(auto_rotate, bool):
        raise TypeError("auto_rotate must be a boolean.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")

    if detector == "yunet":
        if not isinstance(score_threshold, Real) or isinstance(
            score_threshold, bool
        ):
            raise TypeError("score_threshold must be a real number.")
        normalized_threshold = float(score_threshold)
        if not isfinite(normalized_threshold):
            raise ValueError("score_threshold must be finite.")
        if not 0.0 <= normalized_threshold <= 1.0:
            raise ValueError(
                "score_threshold must be between 0 and 1 inclusive."
            )
        if not model_path.exists():
            raise FileNotFoundError(
                f"YuNet model does not exist: {model_path}"
            )
        if not model_path.is_file():
            raise ValueError(
                f"YuNet model path is not a file: {model_path}"
            )

    for name, value in padding_values.items():
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError(f"{name} must be a real number.")
        normalized_padding = float(value)
        if not isfinite(normalized_padding):
            raise ValueError(f"{name} must be finite.")
        if normalized_padding < 0:
            raise ValueError(f"{name} must be non-negative.")


def _format_optional(value: object | None) -> object:
    return "" if value is None else value


def _format_box(
    bounding_box: tuple[int, int, int, int] | None,
) -> str:
    return "" if bounding_box is None else str(bounding_box)


def _item_to_csv_row(item: BatchItemResult) -> dict[str, object]:
    detection_time = (
        ""
        if item.detection_time_ms is None
        else f"{item.detection_time_ms:.3f}"
    )
    return {
        "input_file": item.input_file,
        "status": item.status,
        "output_file": item.output_file,
        "detector_name": item.detector_name,
        "auto_rotate": item.auto_rotate,
        "rotation_angle": _format_optional(item.rotation_angle),
        "faces_count": _format_optional(item.faces_count),
        "detected_box": _format_box(item.detected_box),
        "expanded_box": _format_box(item.expanded_box),
        "crop_width": _format_optional(item.crop_width),
        "crop_height": _format_optional(item.crop_height),
        "detection_time_ms": detection_time,
        "error_type": item.error_type,
        "error_message": item.error_message,
    }


def _short_error_message(
    error: Exception,
    replacements: tuple[tuple[Path, str], ...],
) -> str:
    message = " ".join(str(error).split()) or error.__class__.__name__
    path_strings: list[tuple[str, str]] = []
    for path, replacement in replacements:
        expanded_path = path.expanduser()
        path_strings.append((str(expanded_path), replacement))
        path_strings.append((str(expanded_path.resolve()), replacement))

    for path_string, replacement in sorted(
        set(path_strings),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if path_string:
            message = message.replace(path_string, replacement)
    return message


def _successful_item(
    input_name: str,
    output_name: str,
    auto_rotate: bool,
    result: ExtractionResult,
) -> BatchItemResult:
    crop_height, crop_width = result.crop_shape
    return BatchItemResult(
        input_file=input_name,
        status="success",
        output_file=output_name,
        detector_name=result.detector_name,
        auto_rotate=auto_rotate,
        rotation_angle=result.rotation_angle,
        faces_count=result.faces_count,
        detected_box=result.detected_box,
        expanded_box=result.expanded_box,
        crop_width=crop_width,
        crop_height=crop_height,
        detection_time_ms=result.detection_time_ms,
        error_type="",
        error_message="",
    )


def process_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    report_path: str | Path | None = None,
    detector: str = "yunet",
    auto_rotate: bool = False,
    yunet_model_path: str | Path | None = None,
    score_threshold: float = 0.9,
    padding_left: float = 0.30,
    padding_right: float = 0.30,
    padding_top: float = 0.40,
    padding_bottom: float = 0.50,
    overwrite: bool = False,
) -> BatchSummary:
    """Process supported top-level images and write a deterministic CSV report."""
    input_path = _path_from_argument(input_dir, "input_dir")
    output_path = _path_from_argument(output_dir, "output_dir")
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(
            f"Input path is not a directory: {input_path}"
        )
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_dir and output_dir must be different paths.")
    if output_path.exists() and not output_path.is_dir():
        raise NotADirectoryError(
            f"Output path is not a directory: {output_path}"
        )

    model_path = (
        DEFAULT_YUNET_MODEL_PATH
        if yunet_model_path is None
        else _path_from_argument(yunet_model_path, "yunet_model_path")
    )
    _validate_batch_configuration(
        detector=detector,
        auto_rotate=auto_rotate,
        model_path=model_path,
        score_threshold=score_threshold,
        padding_values={
            "padding_left": padding_left,
            "padding_right": padding_right,
            "padding_top": padding_top,
            "padding_bottom": padding_bottom,
        },
        overwrite=overwrite,
    )

    output_path.mkdir(parents=True, exist_ok=True)

    if report_path is None:
        csv_path = output_path / "batch_report.csv"
    else:
        csv_path = _path_from_argument(report_path, "report_path")
    if csv_path.exists() and csv_path.is_dir():
        raise IsADirectoryError(f"Report path is a directory: {csv_path}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    entries = sorted(input_path.iterdir(), key=lambda path: path.name)
    supported_files = [
        path
        for path in entries
        if not path.name.startswith(".")
        and path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    ignored = len(entries) - len(supported_files)
    items: list[BatchItemResult] = []

    for input_file in supported_files:
        output_name = f"{input_file.stem}_face{input_file.suffix}"
        output_file = output_path / output_name
        if output_file.is_file() and not overwrite:
            items.append(
                BatchItemResult(
                    input_file=input_file.name,
                    status="skipped",
                    output_file=output_name,
                    detector_name=detector,
                    auto_rotate=auto_rotate,
                    rotation_angle=None,
                    faces_count=None,
                    detected_box=None,
                    expanded_box=None,
                    crop_width=None,
                    crop_height=None,
                    detection_time_ms=None,
                    error_type="",
                    error_message="",
                )
            )
            continue

        try:
            extraction_result = extract_face_from_document(
                input_path=input_file,
                output_path=output_file,
                detector=detector,
                auto_rotate=auto_rotate,
                yunet_model_path=yunet_model_path,
                score_threshold=score_threshold,
                padding_left=padding_left,
                padding_right=padding_right,
                padding_top=padding_top,
                padding_bottom=padding_bottom,
            )
        except Exception as error:
            error_message = _short_error_message(
                error,
                (
                    (input_file, input_file.name),
                    (output_file, output_name),
                    (input_path, "."),
                    (output_path, "."),
                    (csv_path, csv_path.name),
                    (model_path, f"models/{model_path.name}"),
                ),
            )
            items.append(
                BatchItemResult(
                    input_file=input_file.name,
                    status="failed",
                    output_file="",
                    detector_name=detector,
                    auto_rotate=auto_rotate,
                    rotation_angle=None,
                    faces_count=None,
                    detected_box=None,
                    expanded_box=None,
                    crop_width=None,
                    crop_height=None,
                    detection_time_ms=None,
                    error_type=error.__class__.__name__,
                    error_message=error_message,
                )
            )
            continue

        items.append(
            _successful_item(
                input_name=input_file.name,
                output_name=output_name,
                auto_rotate=auto_rotate,
                result=extraction_result,
            )
        )

    with csv_path.open("w", encoding="utf-8", newline="") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_item_to_csv_row(item) for item in items)

    successful = sum(item.status == "success" for item in items)
    failed = sum(item.status == "failed" for item in items)
    skipped = sum(item.status == "skipped" for item in items)
    total_detection_time_ms = sum(
        item.detection_time_ms
        for item in items
        if item.status == "success" and item.detection_time_ms is not None
    )
    return BatchSummary(
        total_supported=len(supported_files),
        successful=successful,
        failed=failed,
        skipped=skipped,
        ignored=ignored,
        total_detection_time_ms=total_detection_time_ms,
        report_path=csv_path,
        items=tuple(items),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract portraits from supported images in one directory."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument(
        "--detector",
        choices=("yunet", "haar"),
        default="yunet",
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.9)
    parser.add_argument("--auto-rotate", action="store_true")
    parser.add_argument("--padding-left", type=float, default=0.30)
    parser.add_argument("--padding-right", type=float, default=0.30)
    parser.add_argument("--padding-top", type=float, default=0.40)
    parser.add_argument("--padding-bottom", type=float, default=0.50)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = process_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            report_path=args.report,
            detector=args.detector,
            auto_rotate=args.auto_rotate,
            yunet_model_path=args.model_path,
            score_threshold=args.score_threshold,
            padding_left=args.padding_left,
            padding_right=args.padding_right,
            padding_top=args.padding_top,
            padding_bottom=args.padding_bottom,
            overwrite=args.overwrite,
        )
    except (
        FileNotFoundError,
        NotADirectoryError,
        IsADirectoryError,
        TypeError,
        ValueError,
        OSError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"Input directory: {Path(args.input_dir)}")
    print(f"Output directory: {Path(args.output_dir)}")
    print(f"Detector: {args.detector}")
    print(f"Auto-rotate: {'enabled' if args.auto_rotate else 'disabled'}")
    print(f"Supported files: {summary.total_supported}")
    print(f"Successful: {summary.successful}")
    print(f"Failed: {summary.failed}")
    print(f"Skipped: {summary.skipped}")
    print(f"Ignored: {summary.ignored}")
    print(
        "Total detection time: "
        f"{summary.total_detection_time_ms:.3f} ms"
    )
    print(f"CSV report: {summary.report_path}")
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
