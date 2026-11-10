# Лабораторная работа 4 — вариант 2: фильтрация SMS-спама

Основной результат — **[lab4_mlops.ipynb](lab4_mlops.ipynb)**, выполненный отчёт по всем 15 пунктам.
Отдельный отчёт с оформлением по образцу ЛР2: **[Word](Отчёт_ЛР4_MLOps.docx)**,
**[PDF](Отчёт_ЛР4_MLOps.pdf)**. В титульном листе указан Бикмухаметов Таир Олегович.
Для чтения отчёта в VS Code повторно запускать ячейки не требуется. Дополнительно сохранён
`lab4_mlops.html`, который можно открыть в обычном браузере.
Титульный лист ноутбука можно дополнить персональными данными; в Word они уже заполнены.

Word-отчёт формируется из сохранённых результатов без повторного обучения:

```bash
source .venv/bin/activate
python -m pip install -r requirements-report.txt
python scripts/build_word_report.py
```

Параметры оформления берутся из файла-образца, путь можно указать через `--template`.
PDF экспортирован установленным Microsoft Word с обновлением номеров страниц.

Проект содержит реальные данные UCI, шесть завершённых MLflow runs, две Registry-версии,
экспорты, API, Dockerfile, Compose, тесты и протокол реального обновления/отката.
На зафиксированном test у MultinomialNB(alpha=0.1): F1 spam 0.925620,
precision 0.982456, recall 0.875000. Эти числа относятся к сохранённому разбиению.

## Быстрый запуск через bash

Из корня репозитория:

```bash
./lab4/run.sh
```

Скрипт сам переходит в `lab4`, при необходимости создаёт `.venv` и устанавливает
зафиксированные зависимости. Он проверяет сохранённые модели, собирает и запускает
API с релизом **v2** через Compose, ожидает healthcheck и поднимает MLflow в фоне.
Открыть: http://127.0.0.1:8000/docs и http://127.0.0.1:5000.

```bash
./lab4/run.sh status             # состояние Compose и MLflow
./lab4/run.sh check              # проверить работающий HTTP API v2
./lab4/run.sh test               # локальные проверки API и pytest
./lab4/run.sh logs               # логи контейнера; Ctrl+C завершает просмотр
./lab4/run.sh logs mlflow        # логи фонового MLflow
./lab4/run.sh down               # остановить API и управляемый скриптом MLflow
./lab4/run.sh up v1              # запустить предыдущий релиз
./lab4/run.sh check v1           # проверить API предыдущего релиза
./lab4/run.sh up v2 --no-build   # использовать уже собранный образ
./lab4/run.sh demo               # сборка, обновление, откат и Compose
./lab4/run.sh demo --no-build    # повторить демонстрацию на готовых образах
./lab4/run.sh train --force      # новый цикл обучения, 6 runs + 2 версии
./lab4/run.sh report             # обновить выполненный IPYNB и HTML
./lab4/run.sh help               # все команды
```

После `up v1` стандартный `./lab4/run.sh` возвращает API на v2. Сам `up` не меняет
alias champion: он запускает выбранный сохранённый экспорт. Для демонстрации изменения
alias и отката предназначен `demo`.

Внутри `lab4` можно писать `./run.sh`. Ручная активация окружения не нужна.
Команда `setup` готовит только Python-окружение, `mlflow` запускает только интерфейс MLflow.
Зависимости проверяются локально перед запуском; при совпадении с lock-файлом
повторной установки пакетов не происходит.

Если стандартные порты заняты, используйте одни и те же значения для запуска,
проверки и остановки, например:

```bash
API_PORT=8001 MLFLOW_PORT=5001 ./lab4/run.sh up
API_PORT=8001 MLFLOW_PORT=5001 ./lab4/run.sh check
API_PORT=8001 MLFLOW_PORT=5001 ./lab4/run.sh down
```

Сценарий `demo` использует фиксированные порты 8000/18000 и стандартный Compose-проект.
MLflow, запущенный скриптом, хранит PID и лог в `.runtime/runner/`. Если на выбранном
порту уже работает MLflow **этого проекта**, он используется повторно. Сервер,
запущенный вручную в другом терминале, останавливается там через Ctrl+C.

`.gitignore` исключает окружения, кеши, локальные PID/логи, временные SQLite-файлы,
блокировки Word/LibreOffice и случайные каталоги `mlruns`/`mlartifacts`.
Данные, разбиение, настоящая база `mlflow-data`, экспорты, результаты и готовые отчёты
доступны для добавления в Git как материалы лабораторной.

## Подготовка в WSL Ubuntu 24.04

Docker Desktop должен работать с Linux engine. Включите
Settings → Resources → WSL Integration → Ubuntu → Apply.
Выполняйте команды в Ubuntu, из каталога проекта:

```bash
cd /home/tair/NETI/AI/lab4
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
docker version
docker compose version
```

Python — 3.12. Проверенный полный lock — `requirements.txt`, прямые зависимости —
`requirements.in`, минимальный runtime lock — `requirements-runtime.txt`.
Системный Python и окружения других лабораторных не изменяются.
Исходный файл уже сохранён, повторная загрузка UCI не требуется.

## Запуск полного цикла

```bash
source .venv/bin/activate
python -m sms_mlops.train
python -m sms_mlops.verify
python -m pytest -q
python -m sms_mlops.docker_demo --require
```

`train` при наличии результатов использует сохранённые runs и модели.
Для **нового обучения**: `python -m sms_mlops.train --force`.
Это добавит шесть runs и две новые Registry-версии, обновит локальные релизы `v1/v2`
и потребует заново собрать образы/выполнить отчёт. Номер Registry не равен тегу образа.

`docker_demo` собирает `sms-spam:v1` и `sms-spam:v2`, запускает их на порту 18000
в порядке v1 → v2 → v1, сверяет предсказания с экспортом, меняет alias и проверяет
неизменность числа runs и SHA-256 моделей. Затем возвращает champion версии v2
и оставляет её работающей в Compose на **127.0.0.1:8000**.
Существующий контейнер `lab4-sms-demo` автоматически не удаляется: сценарий откажется
использовать занятое имя. Детальные команды и stdout/stderr — `results/docker_checks.json`.

## Просмотр MLflow

В отдельном терминале из `lab4`:

```bash
source .venv/bin/activate
python -m sms_mlops.relocate
mlflow server --backend-store-uri sqlite:///mlflow-data/mlflow.db \
  --default-artifact-root ./mlflow-data/artifacts --no-serve-artifacts \
  --host 127.0.0.1 --port 5000 --workers 1
```

Откройте http://127.0.0.1:5000. Эксперимент `lab4-sms-spam`, модель `sms-spam-filter`,
alias `champion`. SQLite используется непосредственно при обучении; сервер нужен
для интерфейса. Запускайте его из `lab4`, чтобы пути совпадали.

При переносе всей папки на другой компьютер выполните `python -m sms_mlops.relocate`
перед запуском UI: это обновит абсолютные пути к локальным артефактам в SQLite,
сохранив резервную копию. `train` также делает это автоматически. Переносите **всю**
папку `mlflow-data`, включая артефакты. Остановите UI перед переносом/резервным копированием.

## API и Compose

```bash
MODEL_RELEASE=v2 docker compose up -d --no-build --wait --wait-timeout 90
docker compose ps
curl -f http://127.0.0.1:8000/health
curl -sS -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' --data-binary @examples/valid.json
curl -i -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' --data-binary @examples/invalid_empty.json
python -m sms_mlops.verify --url http://127.0.0.1:8000 --release v2
```

Если образы ещё не собраны, замените `--no-build` на `--build`.
Swagger — http://127.0.0.1:8000/docs, метаданные — `/metadata`.
Вход: `{"texts": ["Hi, see you soon!", "Claim your free prize now!"]}`.
API возвращает `predictions`, `probabilities_spam`, `model_version`, `release`, `run_id`,
`latency_ms`. Принимается 1–100 непустых строк длиной до 5000 символов;
неверные типы и лишние поля возвращают 422. Модель загружается один раз при старте.
Неизвестные слова/Unicode допустимы, но качество на русском языке не оценивалось.

Ручная сборка и запуск конкретного снимка:

```bash
docker build --build-arg MODEL_RELEASE=v1 -t sms-spam:v1 .
docker build --build-arg MODEL_RELEASE=v2 -t sms-spam:v2 .
docker run --rm -d --name lab4-sms-demo -p 127.0.0.1:18000:8000 sms-spam:v1
python -m sms_mlops.verify --url http://127.0.0.1:18000 --release v1
docker stop lab4-sms-demo
```

Откат без обучения и пересборки: запустите сохранённый `sms-spam:v1` вместо v2.
Полная автоматическая демонстрация с alias и проверками — `docker_demo`.
Остановка Compose: `docker compose down`. MLflow в своём терминале останавливается Ctrl+C.

## Ноутбук

Откройте `lab4_mlops.ipynb` в VS Code и выберите интерпретатор `lab4/.venv/bin/python`.
**Run All** проверит сохранённые runs и локальный API, покажет контейнерный протокол
с датой запуска. Чтобы повторить контейнерные проверки, задайте `LAB4_RUN_DOCKER=1`
до запуска ядра: ноутбук проверит существующие образы, обновление, откат и Compose.
Для повторной сборки используйте отдельно `python -m sms_mlops.docker_demo --require`.

Выполнение без интерфейса и экспорт в HTML:

```bash
source .venv/bin/activate
JUPYTER_CONFIG_DIR="$PWD/.runtime/jupyter-config" \
JUPYTER_DATA_DIR="$PWD/.runtime/jupyter-data" \
JUPYTER_RUNTIME_DIR="$PWD/.runtime/jupyter-runtime" \
IPYTHONDIR="$PWD/.runtime/ipython" \
python scripts/execute_report.py
```

`scripts/build_report.py` заново создаёт **невыполненный** ноутбук из исходного текста.
Для обычного просмотра/повторного исполнения этот шаг не нужен.
Скриншоты — настоящие снимки локального MLflow и Swagger, встроены в выполненные
ячейки и сохранены в `screenshots`. Registry (`02_registry.png`), validation-метрики
(`06_mlflow_validation_metrics.png`), обзор обучения/test (`07_mlflow_metrics_overview.png`)
и test precision/recall (`08_mlflow_test_metrics.png`) включены в отчёты.
Оригиналы PNG сохранены без изменений; их источники и SHA-256 находятся в
`screenshots/capture_metadata.json`. Эти четыре изображения включены в IPYNB, HTML,
Word и PDF. Зафиксированные снимки защищены от перезаписи автоматической съёмкой.
Инструмент обновления остальных снимков — `scripts/capture_screenshots.py`
(отдельно установить Playwright 1.51.0 и Chromium).

## Воспроизводимость и данные

UCI SMS Spam Collection: https://archive.ics.uci.edu/dataset/228/sms+spam+collection,
Almeida & Hidalgo, DOI 10.24432/C5CC84, CC BY 4.0.
Сохранены исходный архив, TSV и readme; SHA-256 — `data/manifest.json`.
Из 5574 строк исключено 415 повторов после нормализации; осталось 5159 SMS.
Точное стратифицированное разбиение 3095/1032/1032 — `data/splits.json`.
TF-IDF обучается только на train; параметры выбираются на validation; test оценивает
только финальный замороженный Pipeline. Переобучение на train+validation не выполняется.

Базовый Python-образ зафиксирован тегом и SHA-256 digest, фактические ID/digest образов и build-вывод —
в Docker-протоколе. Прямые зависимости не обновляйте без повторной проверки lock-файлов:
SQLAlchemy 2.1.8 несовместима с используемым MLflow, поэтому закреплена 2.0.44.
`scripts/freeze_requirements.py` обновляет lock-файлы из активного проверенного окружения.
