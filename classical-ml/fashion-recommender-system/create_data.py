from pathlib import Path

import kagglehub

RAW_DATA_DIR = Path.cwd() / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

path = kagglehub.dataset_download(
    "sohyunjun0401/h-and-m-personalized-fashion-data",
    output_dir=str(RAW_DATA_DIR),
)

print("Датасет сохранён в:", path)

print("\nСкачанные файлы:")
for file_path in sorted(RAW_DATA_DIR.iterdir()):
    if file_path.is_file():
        size_mb = file_path.stat().st_size / 1024**2
        print(f"{file_path.name}: {size_mb:.2f} MB")