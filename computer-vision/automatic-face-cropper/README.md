# Automatic Face Cropper

Автоматическое обнаружение и вырезание лица с фотографии с помощью Haar Cascade и YuNet.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Face%20Detection-2F855A)

Проект создан для прикладной фриланс-задачи: получить фотографию, автоматически найти на ней лица, выбрать крупнейшее из них и сохранить его отдельным изображением.

На вход программа принимает одно изображение или папку с изображениями. На выходе создаётся прямоугольный crop области лица; при пакетной обработке дополнительно формируется CSV-отчёт.

> Проект выполняет детекцию и прямоугольное вырезание области лица. Это не сегментация по контуру и не удаление фона.

## Пример работы

<p align="center">
  <strong>Исходное изображение</strong>
</p>

<p align="center">
  <img src="examples/document.jpg" alt="Исходное изображение document.jpg" width="500">
</p>

<table>
  <tr>
    <th align="center">Haar Cascade</th>
    <th align="center">YuNet</th>
  </tr>
  <tr>
    <td align="center">
      <img src="examples/haar_face.jpg" alt="Результат Haar Cascade — haar_face.jpg" width="250">
    </td>
    <td align="center">
      <img src="examples/yunet_face.jpg" alt="Результат YuNet — yunet_face.jpg" width="250">
    </td>
  </tr>
</table>

Оба детектора получили одно исходное изображение `document.jpg`, нашли находящиеся на нём лица и выбрали лицо с крупнейшим bounding box. Полученные области сохранены отдельно как `haar_face.jpg` и `yunet_face.jpg`.

## Как работает программа

```text
Фотография
    ↓
Обнаружение лиц
    ↓
Выбор крупнейшего лица
    ↓
Расширение и ограничение bounding box
    ↓
Вырезание прямоугольной области
    ↓
Сохранение результата
```

Если детектор находит несколько лиц, программа сравнивает площади bounding box и выбирает крупнейший. Перед вырезанием рамка расширяется пропорциональными отступами и ограничивается границами исходного изображения.

## Возможности

- обработка одного изображения;
- выбор между Haar Cascade и YuNet;
- автоматический выбор крупнейшего найденного лица;
- сохранение отдельного crop-изображения;
- пакетная обработка папки;
- проверка поворотов на `0°`, `90°`, `180°` и `270°`;
- формирование CSV-отчёта;
- продолжение пакетной обработки после повреждённого изображения;
- пропуск файлов с уже существующим результатом;
- принудительная повторная обработка через `--overwrite`.

## Использованные методы

### Haar Cascade

Классический детектор лиц из OpenCV. Подходит как простой и быстрый вариант, но может быть чувствителен к качеству изображения, освещению и повороту головы.

Для работы используется фронтальный Haar Cascade, поставляемый вместе с OpenCV.

### YuNet

Нейросетевой детектор лиц, запускаемый через OpenCV и ONNX-модель:

```text
models/face_detection_yunet_2023mar.onnx
```

Модель хранится в проекте и загружается по относительному пути. YuNet используется как более современный вариант детекции, но проект не предполагает, что он всегда превосходит Haar Cascade на любом изображении.

Haar Cascade и YuNet — два доступных способа решения одной прикладной задачи, а не академическое сравнение алгоритмов.

## Структура проекта

```text
automatic-face-cropper/
├── .gitignore
├── README.md
├── requirements.txt
├── examples/
│   ├── document.jpg
│   ├── haar_face.jpg
│   └── yunet_face.jpg
├── models/
│   ├── README.md
│   └── face_detection_yunet_2023mar.onnx
└── src/
    ├── __init__.py
    ├── batch.py
    ├── demo.py
    ├── face_cropper.py
    └── face_detector.py
```

Основные файлы:

- `src/demo.py` — обработка одного изображения;
- `src/batch.py` — последовательная обработка папки и создание CSV-отчёта;
- `src/face_detector.py` — детекция лиц через Haar Cascade и YuNet;
- `src/face_cropper.py` — загрузка, выбор лица, обрезка и сохранение результата;
- `examples/` — исходное демонстрационное изображение и два готовых результата;
- `models/` — ONNX-модель YuNet и информация о ней.

## Установка

Создайте виртуальное окружение:

```bash
python -m venv .venv
```

Активируйте его на macOS или Linux:

```bash
source .venv/bin/activate
```

На Windows:

```bash
.venv\Scripts\activate
```

Установите зависимости:

```bash
pip install -r requirements.txt
```

Все команды ниже выполняются из корневой папки проекта.

## Обработка одного изображения

### Haar Cascade

```bash
python -m src.demo \
  --input examples/document.jpg \
  --output examples/haar_face.jpg \
  --detector haar
```

Одной строкой:

```bash
python -m src.demo --input examples/document.jpg --output examples/haar_face.jpg --detector haar
```

### YuNet

```bash
python -m src.demo \
  --input examples/document.jpg \
  --output examples/yunet_face.jpg \
  --detector yunet
```

Одной строкой:

```bash
python -m src.demo --input examples/document.jpg --output examples/yunet_face.jpg --detector yunet
```

Для YuNet по умолчанию используется модель `models/face_detection_yunet_2023mar.onnx` и порог уверенности `0.9`. Параметры можно изменить через `--model-path` и `--score-threshold`.

Флаг `--auto-rotate` включает проверку четырёх ориентаций изображения:

```bash
python -m src.demo --input examples/document.jpg --output local_data/face.jpg --detector yunet --auto-rotate
```

Детектор запускается для каждого поворота, после чего выбирается ориентация с крупнейшей относительной площадью найденного лица.

## Пакетная обработка

Создайте локальную папку `local_data/input` и поместите в неё изображения `.jpg`, `.jpeg` или `.png`. Вложенные каталоги не обрабатываются.

```bash
python -m src.batch \
  --input-dir local_data/input \
  --output-dir local_data/output \
  --report local_data/batch_report.csv \
  --detector yunet \
  --auto-rotate
```

При таком запуске:

- изображения читаются из `local_data/input`;
- crops сохраняются в `local_data/output`;
- к имени результата добавляется суффикс `_face`;
- сведения об обработке записываются в `local_data/batch_report.csv`;
- повреждённый файл получает статус `failed`, но не останавливает остальные;
- уже существующие результаты получают статус `skipped`.

Для принудительной повторной обработки добавьте `--overwrite`:

```bash
python -m src.batch \
  --input-dir local_data/input \
  --output-dir local_data/output \
  --report local_data/batch_report.csv \
  --detector yunet \
  --auto-rotate \
  --overwrite
```

В CSV сохраняются имена файлов, статус, выбранный поворот, координаты исходной и расширенной рамок, размер crop, время детекции и информация об ошибке.

Папка `local_data/` предназначена для локальных входных файлов, результатов и отчётов и не добавляется в Git.

## Результат

Программа сохраняет отдельный прямоугольный crop лица без принудительного изменения его пропорций. Такой результат можно использовать как вход для последующей обработки изображений.

> Проект выполняет детекцию и прямоугольное вырезание области лица. Он не выполняет точную сегментацию лица или головы по контуру.

## Ограничения

- качество результата зависит от разрешения и освещения фотографии;
- размытие и сильный поворот головы могут ухудшить детекцию;
- очень маленькие лица могут определяться хуже;
- при нескольких найденных лицах выбирается крупнейший bounding box;
- прямоугольный crop может содержать часть окружающего фона;
- Haar Cascade может быть менее устойчивым на сложных изображениях;
- автоматический поворот проверяет только углы, кратные `90°`;
- выбор ориентации основан на площади найденной рамки и не гарантирует визуально правильный поворот на каждом изображении;
- пакетная обработка выполняется последовательно и не обходит вложенные папки.
