 # ITMO Acoustic Event Detection 2026
Пайплайн классификации аудиособытий для [соревнования на Kaggle](https://www.kaggle.com/competitions/itmo-acoustic-event-detectin-2026).

## Порядок запуска

### 1. Настройка окружения
Убедитесь, что у вас установлен [uv](https://github.com/astral-sh/uv).

```bash
uv sync
uv run pre-commit install
```

### 2. Подготовка данных и конфига
1. Положите ваши аудиофайлы и CSV-файлы в папку `data/`.
2. Настройте пути к данным и параметры модели в файле `config.yaml`.

### 3. Извлечение признаков
Кэширование Log-Mel спектрограмм в `data/features_cache/` перед обучением или инференсом.

```bash
# тренировочные признаки
uv run python preprocess.py

# тестовые признаки
uv run python preprocess.py split=test
```

### 4. Обучение модели
Запуск тренировки с автоматическим сохранением лучших чекпоинтов в папку `models/`. Модель автоматически скачивает предобученные веса AST при первом запуске.

```bash
uv run python train.py
```

*логи обучения доступны в TensorBoard:* `tensorboard --logdir logs/ast_tb`

### 5. Получение предсказаний
Генерация файла `submission.csv` для Kaggle. Скрипт по умолчанию берет самый свежий чекпоинт из папки `models/`.

```bash
# использовать последний чекпоинт
uv run python predict.py

# использовать конкретный чекпоинт
uv run python predict.py +prediction.checkpoint_path=models/best-epoch=05-val_f1=0.850.ckpt
```

## Примечания

* можно использовать аугментации Mixin (смешивание с шумом) и SpecAugment (параметр в config.yaml).
* для воспроизводимости экспериментов не меняйте training.random_state.
* основные гиперпараметры и пути хранятся в config.yaml.
* `submission.csv` перезаписывается на месте исходного файла.
