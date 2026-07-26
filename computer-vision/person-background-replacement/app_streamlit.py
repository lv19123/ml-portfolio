"""Streamlit UI for image and video background replacement."""

from io import BytesIO
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import streamlit as st

from src.inference import (
    load_segmentation_model,
    predict_mask_from_array,
    replace_background,
    save_mask,
    save_rgb_image,
)
from src.video import replace_video_background


DEFAULT_MODEL_PATH = Path("models/mobile_unet_model.keras")
OUTPUT_DIR = Path("outputs") / "streamlit"
IMAGE_RESULT_PATH = OUTPUT_DIR / "result_image.png"
IMAGE_MASK_PATH = OUTPUT_DIR / "mask_image.png"
VIDEO_RESULT_PATH = OUTPUT_DIR / "result_video.mp4"

st.set_page_config(page_title="Background Replacement App", layout="wide")


@st.cache_resource(show_spinner=False)
def load_cached_model(model_path: str):
    """Load and cache the segmentation model for Streamlit reruns."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load_segmentation_model(str(path))


def read_uploaded_image_rgb(uploaded_file) -> np.ndarray:
    """Read a Streamlit uploaded image as an RGB numpy array."""
    with Image.open(BytesIO(uploaded_file.getvalue())) as image:
        return np.array(image.convert("RGB"))


def save_uploaded_image_as_png(uploaded_file, output_path: Path) -> Path:
    """Save an uploaded image as PNG so OpenCV can read it by path."""
    image_rgb = read_uploaded_image_rgb(uploaded_file)
    Image.fromarray(image_rgb).save(output_path)
    return output_path


def save_uploaded_file(uploaded_file, output_path: Path) -> Path:
    """Save an uploaded binary file to disk."""
    output_path.write_bytes(uploaded_file.getvalue())
    return output_path


def show_download_button(label: str, file_path: Path, mime: str, key: str) -> None:
    """Render a download button for an existing output file."""
    st.download_button(
        label=label,
        data=file_path.read_bytes(),
        file_name=file_path.name,
        mime=mime,
        key=key,
    )


def render_image_tab(model_path: str, image_size: int, threshold: float) -> None:
    st.subheader("Image")
    source_file = st.file_uploader(
        "Upload source image",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_source_upload",
    )
    background_file = st.file_uploader(
        "Upload new background image",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_background_upload",
    )

    preview_left, preview_right = st.columns(2)
    image_rgb = None
    background_rgb = None

    if source_file is not None:
        try:
            image_rgb = read_uploaded_image_rgb(source_file)
            preview_left.image(image_rgb, caption="Source image", use_container_width=True)
        except Exception as exc:
            preview_left.error(f"Could not read source image: {exc}")

    if background_file is not None:
        try:
            background_rgb = read_uploaded_image_rgb(background_file)
            preview_right.image(
                background_rgb,
                caption="New background",
                use_container_width=True,
            )
        except Exception as exc:
            preview_right.error(f"Could not read background image: {exc}")

    if st.button("Replace image background", type="primary"):
        if source_file is None or background_file is None:
            st.error("Upload both a source image and a background image.")
            return
        if image_rgb is None or background_rgb is None:
            st.error("One of the uploaded images could not be read.")
            return

        try:
            with st.spinner("Processing image..."):
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory() as temp_dir:
                    background_path = save_uploaded_image_as_png(
                        background_file,
                        Path(temp_dir) / "background.png",
                    )
                    model = load_cached_model(model_path)
                    mask = predict_mask_from_array(
                        model,
                        image_rgb,
                        image_size=image_size,
                        threshold=threshold,
                    )
                    result_rgb = replace_background(
                        image_rgb,
                        mask,
                        str(background_path),
                    )
                    save_rgb_image(result_rgb, str(IMAGE_RESULT_PATH))
                    save_mask(mask, str(IMAGE_MASK_PATH))

            result_col, mask_col = st.columns(2)
            result_col.image(
                result_rgb,
                caption="Result image",
                use_container_width=True,
            )
            mask_col.image(
                mask * 255,
                caption="Predicted mask",
                use_container_width=True,
            )

            download_col, mask_download_col = st.columns(2)
            with download_col:
                show_download_button(
                    "Download result_image.png",
                    IMAGE_RESULT_PATH,
                    "image/png",
                    "download_result_image",
                )
            with mask_download_col:
                show_download_button(
                    "Download mask_image.png",
                    IMAGE_MASK_PATH,
                    "image/png",
                    "download_mask_image",
                )
        except Exception as exc:
            st.error(f"Image processing failed: {exc}")


def render_video_tab(model_path: str, image_size: int, threshold: float) -> None:
    st.subheader("Video")
    video_file = st.file_uploader(
        "Upload source video",
        type=["mp4", "mov", "avi", "mkv"],
        key="video_source_upload",
    )
    background_file = st.file_uploader(
        "Upload new background image",
        type=["jpg", "jpeg", "png", "webp"],
        key="video_background_upload",
    )
    max_frames_input = st.number_input(
        "Max frames for quick test",
        min_value=0,
        value=120,
        step=1,
        help="Use 0 to process the full video. Larger values take more time.",
    )

    preview_left, preview_right = st.columns(2)
    background_rgb = None

    if video_file is not None:
        preview_left.video(video_file.getvalue())

    if background_file is not None:
        try:
            background_rgb = read_uploaded_image_rgb(background_file)
            preview_right.image(
                background_rgb,
                caption="New background",
                use_container_width=True,
            )
        except Exception as exc:
            preview_right.error(f"Could not read background image: {exc}")

    if st.button("Replace video background", type="primary"):
        if video_file is None or background_file is None:
            st.error("Upload both a source video and a background image.")
            return
        if background_rgb is None:
            st.error("The uploaded background image could not be read.")
            return

        try:
            with st.spinner("Processing video..."):
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                VIDEO_RESULT_PATH.unlink(missing_ok=True)
                max_frames = None if int(max_frames_input) == 0 else int(max_frames_input)

                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    video_suffix = Path(video_file.name).suffix or ".mp4"
                    video_path = save_uploaded_file(
                        video_file,
                        temp_path / f"source{video_suffix}",
                    )
                    background_path = save_uploaded_image_as_png(
                        background_file,
                        temp_path / "background.png",
                    )

                    model = load_cached_model(model_path)
                    replace_video_background(
                        model,
                        video_path=str(video_path),
                        background_path=str(background_path),
                        output_path=str(VIDEO_RESULT_PATH),
                        image_size=image_size,
                        threshold=threshold,
                        max_frames=max_frames,
                    )

            if VIDEO_RESULT_PATH.exists():
                video_bytes = VIDEO_RESULT_PATH.read_bytes()
                st.video(video_bytes)
                st.download_button(
                    label="Download result_video.mp4",
                    data=video_bytes,
                    file_name=VIDEO_RESULT_PATH.name,
                    mime="video/mp4",
                    key="download_result_video",
                )
            else:
                st.error("Video processing finished, but result_video.mp4 was not created.")
        except Exception as exc:
            st.error(f"Video processing failed: {exc}")


def main() -> None:
    st.title("Background Replacement App")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with st.sidebar:
        st.header("Settings")
        model_path = st.text_input("Model path", value=str(DEFAULT_MODEL_PATH))
        image_size = st.number_input(
            "Image size",
            min_value=32,
            max_value=2048,
            value=256,
            step=32,
        )
        threshold = st.slider(
            "Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
        )

        if not Path(model_path).exists():
            st.warning("Model file does not exist yet.")

    image_tab, video_tab = st.tabs(["Image", "Video"])
    with image_tab:
        render_image_tab(model_path, int(image_size), float(threshold))
    with video_tab:
        render_video_tab(model_path, int(image_size), float(threshold))


if __name__ == "__main__":
    main()
