"""Build the report source. Execute the notebook afterward to embed real results."""
from pathlib import Path
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))


md("""
# 1. Титульный лист

**Лабораторная работа № 4. MLOps**  
**Вариант 2. Сервис фильтрации SMS-спама**

Образовательная организация: ______________________________  
Кафедра: ______________________________  
Выполнил(а): ______________________________  
Группа: ______________________________  
Проверил(а): ______________________________

Новосибирск, 2026

Отчёт содержит выполненные ячейки, фактические идентификаторы MLflow, метрики,
выводы Docker и скриншоты реальных страниц. Поля титульного листа заполняются студентом.
Все пути команд далее отсчитываются от `lab4`.
""")
md("""
## 2. Цель работы и постановка задачи

Освоить воспроизводимый цикл ML: подготовку данных, учёт и сравнение экспериментов
в MLflow, Model Registry и aliases, экспорт модели, API, Docker, обновление, откат
и Docker Compose с healthcheck.

Индивидуальная задача: по тексту SMS определить `ham` (обычное сообщение) или `spam`.
Исследуются `TF-IDF + MultinomialNB` с изменением `alpha` и
`TF-IDF + LogisticRegression` с изменением `C`. Основной критерий — **F1 для spam**;
дополнительно сравниваются precision и recall для spam. `spam=1`, `ham=0`.

Правило выбора задаётся до итогового тестирования: максимум validation F1; при равенстве —
максимум precision, далее порядок конфигураций в исходном коде. Порог 0,5 фиксирован.
Test не используется для изменения гиперпараметров или порога.
""")
code("""
from pathlib import Path
import os
import sys
import json
import subprocess
import importlib.metadata

candidates = [Path.cwd(), Path.cwd() / 'lab4']
ROOT = next((p.resolve() for p in candidates if (p / 'sms_mlops/train.py').exists()), None)
if ROOT is None:
    raise RuntimeError('Откройте ноутбук из lab4 или корня репозитория')
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.runtime/matplotlib'))

import pandas as pd
import numpy as np
from IPython.display import display, Markdown, Image
from sms_mlops.common import read_json, sha256, MODEL_NAME

def table(frame):
    display(frame.style.format(precision=6).set_properties(**{'text-align': 'left'}))

def screenshot(name, caption):
    path = ROOT / 'screenshots' / name
    if path.exists():
        display(Markdown(caption))
        display(Image(filename=str(path), width=1200))
    else:
        display(Markdown('Скриншот ещё не снят: `' + name + '`.'))

print('Каталог проекта:', ROOT)
print('Интерпретатор:', sys.executable)
print('Python:', sys.version.split()[0])
""")
md("""
## 3. Набор данных, подготовка и разбиение

Источник: [SMS Spam Collection, UCI](https://archive.ics.uci.edu/dataset/228/sms+spam+collection),
Almeida и Hidalgo, DOI [10.24432/C5CC84](https://doi.org/10.24432/C5CC84), лицензия CC BY 4.0.
Признак — исходная строка `text`, цель — метка `label`.
Идентификатор исходной строки `row_id` используется только для воспроизведения разбиения.

Оригинальный TSV читается разделением по **первой табуляции**, без CSV-интерпретации кавычек.
Пропуски, пустые тексты и неизвестные метки проверяются явно. Дубликаты определяются по
NFKC-нормализации, `casefold` и свёртке пробелов; сохраняется первое сообщение.
Группы с противоречащими метками исключаются полностью. Эта нормализация служит
только для проверки повторов, в классификатор передаётся исходный текст.
Решения об исключении сохранены в `data/excluded.csv`.

После очистки выполняется стратифицированное разбиение примерно 60/20/20 при `random_state=42`.
`data/splits.json` хранит точные исходные `row_id` и SHA-256 исходного файла.
Множества строк и хешей сообщений попарно не пересекаются. Близкие по смыслу шаблоны
могут оставаться — удаление повторов не решает проблему всех разновидностей похожих SMS.

TF-IDF обучается только внутри Pipeline на train. Validation и test преобразуются
уже обученным векторизатором. Англоязычный исторический корпус не даёт оснований
утверждать такое же качество на современных русскоязычных SMS.
""")
code("""
from sms_mlops.data import prepare_data
frames, dataset = prepare_data()
display(pd.DataFrame([
    {'этап': 'Исходный файл', 'сообщений': dataset['raw_rows'], **dataset['raw_classes']},
    {'этап': 'После очистки', 'сообщений': dataset['clean_rows'], **dataset['clean_classes']},
]))
display(pd.DataFrame(dataset['splits']).T)
print('Исключено:', dataset['excluded_rows'])
print('Групп с противоречивыми метками:', dataset['conflicting_hashes'])
print('SHA-256 исходного файла:', dataset['raw_sha256'])
print('SHA-256 разбиения:', dataset['split_sha256'])
assert sha256(ROOT / 'data/raw/SMSSpamCollection') == dataset['raw_sha256']
assert all(set(frames[a].text_hash).isdisjoint(frames[b].text_hash)
           for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')])
""")
md("""
Набор несбалансирован: обычных сообщений значительно больше. Поэтому accuracy
может быть высокой даже при пропуске значительной части спама.
Precision для spam = TP/(TP+FP): насколько редко обычная SMS блокируется ошибочно.
Recall = TP/(TP+FN): какую долю спама обнаруживает модель.
F1 = 2PR/(P+R) учитывает оба вида ошибок.
""")
md("""
## 4. Окружение, зависимости и структура проекта

Рабочая среда — Ubuntu 24.04 под WSL 2, Python 3.12, Docker Desktop с интеграцией WSL.
Для этой реализации зафиксированы и проверены версии ниже. Они отличаются от версий
учебного примера в PDF: код и API MLflow соответствуют именно установленной версии.
SQLAlchemy 2.0.44 закреплена явно: версия 2.1.8 оказалась несовместимой с MLflow 2.22.2.

`requirements.in` содержит прямые зависимости, `requirements.txt` — полный снимок,
`requirements-runtime.txt` — транзитивные зависимости только API и инференса.
В образ не включаются MLflow, Jupyter и исходные данные. Векторизатор и классификатор
имеют одинаковую версию scikit-learn при обучении и обслуживании.
""")
code("""
packages = ['mlflow', 'sqlalchemy', 'scikit-learn', 'numpy', 'scipy', 'pandas',
            'fastapi', 'pydantic', 'uvicorn', 'joblib', 'nbconvert']
display(pd.DataFrame({'пакет': packages,
                      'версия': [importlib.metadata.version(p) for p in packages]}))
checked = subprocess.run([sys.executable, '-m', 'pip', 'check'], text=True, capture_output=True)
print(checked.stdout.strip())
assert checked.returncode == 0, checked.stderr
print((ROOT / 'compose.yaml').read_text())
""")
md("""
```text
lab4/
  lab4_mlops.ipynb               выполненный отчёт
  README.md                     запуск под WSL
  sms_mlops/                    данные, обучение, API, проверки, Docker-сценарий
  scripts/                      генерация отчёта, lock-файлов, скриншотов
  tests/                        контракт API и контроль целостности модели
  data/raw/                     неизменённый оригинал UCI и описание
  data/clean.csv, excluded.csv   данные и журнал очистки
  data/splits.json              идентификаторы train/validation/test
  mlflow-data/mlflow.db          настоящая SQLite-база Tracking и Registry
  mlflow-data/artifacts/         Pipeline, signature, примеры, код, графики
  models/v1/, models/v2/         экспортированные Pipeline и metadata.json
  results/                      CSV, JSON, графики и протоколы проверок
  screenshots/                  реальные скриншоты браузера
  examples/                     корректные и некорректные JSON-запросы
  requirements*.in, *.txt        прямые и полные зависимости
  Dockerfile, compose.yaml, .dockerignore
```
""")
md("""
## 5. Последовательность выполнения

Команды выполняются в bash из `lab4`. Docker переустанавливать не требуется.
В Docker Desktop должны быть запущен Linux engine и включена
Settings → Resources → WSL Integration → Ubuntu.

```bash
cd /home/tair/NETI/AI/lab4
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
docker version
docker compose version
python -m sms_mlops.train
python -m sms_mlops.verify
python -m pytest -q
python -m sms_mlops.docker_demo --require
```

Обучение пишет в SQLite непосредственно; отдельный работающий сервер для этого не нужен.
Для просмотра тех же данных в браузере запустить в другом терминале:

```bash
source .venv/bin/activate
mlflow server --backend-store-uri sqlite:///mlflow-data/mlflow.db \\
  --default-artifact-root ./mlflow-data/artifacts --no-serve-artifacts \\
  --host 127.0.0.1 --port 5000 --workers 1
```

Открыть <http://127.0.0.1:5000>. API обслуживается на <http://127.0.0.1:8000/docs>.
Ниже ноутбук использует сохранённые запуски. Если результатов ещё нет,
`train.run()` создаёт шесть экспериментов; `--force` создаёт **новые** run и номера Registry.
""")
code("""
from sms_mlops.train import run, setup_tracking
summary = run()
client, experiment_id = setup_tracking()
experiments = pd.read_csv(ROOT / 'results/experiments.csv')
for run_id in experiments.run_id:
    assert client.get_run(run_id).info.status == 'FINISHED'
print('Проверено завершённых экспериментов:', len(experiments))
screenshot('01_mlflow_experiments.png', '**Рисунок 1. Настоящие шесть запусков в MLflow.**')
""")
md("""
## 6. Сравнение экспериментов

Во всех опытах: word TF-IDF, униграммы и биграммы, `max_features=20000`,
`min_df=1`, `sublinear_tf=True`, `lowercase=True`, L2-нормировка.
MultinomialNB исследуется при `alpha ∈ {0.1, 0.5, 1}`;
LogisticRegression — при `C ∈ {0.5, 2, 10}`, `solver=liblinear`, `max_iter=2000`.
Для всех моделей используются одни и те же train и validation.

В каждом run сохранены параметры, seed, размеры выборок, четыре validation-метрики,
время обучения, размер словаря, теги варианта и контрольных сумм, входные datasets,
матрица ошибок, разбиение, исходный код, зависимости и MLflow-модель с signature/input_example.
Только после выбора чемпиону дополнительно записываются итоговые test-метрики.
""")
code("""
columns = ['algorithm', 'parameter', 'value', 'val_f1_spam', 'val_precision_spam',
           'val_recall_spam', 'val_accuracy', 'fit_seconds', 'run_id']
table(experiments[columns].sort_values(['val_f1_spam', 'val_precision_spam'], ascending=False))
display(Image(filename=str(ROOT / 'results/figures/experiments.png'), width=1100))
display(Markdown('**Рисунок 2. Сравнение validation F1, precision и recall для spam.**'))
screenshot('06_mlflow_validation_metrics.png',
           '**Рисунок 3. Validation-метрики и размер словаря шести запусков MLflow.**')
run_example = client.get_run(summary['champion']['run_id'])
print('Теги выбранного запуска:', json.dumps(run_example.data.tags, ensure_ascii=False, indent=2))
print('Артефакты:', [item.path for item in client.list_artifacts(run_example.info.run_id)])
""")
md("""
Увеличение `alpha` усиливает сглаживание NB. В полученном сравнении модель становится
более консервативной: precision достигает 1, но recall и F1 падают.
У LogisticRegression рост `C` ослабляет регуляризацию и в исследованном диапазоне
повышает recall. Это наблюдение относится к данному разбиению, а не является
гарантией монотонного улучшения для любых данных и параметров.
""")
md("""
## 7. Выбор модели и итоговая оценка

Выбирается лучший run по validation F1 для spam. Разница между лидерами мала;
по одному разбиению нельзя утверждать статистически доказанное превосходство.
Сетка и порог после просмотра test не менялись.
Pipeline остаётся обученным на train: дополнительного переобучения на train+validation
в этом протоколе нет. Это сохраняет полное соответствие между сравнённым run,
Registry-версией, экспортом и итоговой оценкой.
""")
code("""
winner = summary['champion']
runner_up = summary['alternative']
final = read_json(ROOT / 'results/test_metrics.json')
display(Markdown(
    f"Выбрана **{winner['algorithm']}**, `{winner['parameter']}={winner['value']:g}`. "
    f"Validation F1 = **{winner['val_f1_spam']:.6f}**, "
    f"precision = {winner['val_precision_spam']:.6f}, recall = {winner['val_recall_spam']:.6f}. "
    f"Лучший представитель другого семейства: {runner_up['algorithm']}, "
    f"validation F1 = {runner_up['val_f1_spam']:.6f}."))
table(pd.DataFrame([{k: v for k, v in final.items() if k.startswith('test_')}]))
display(pd.DataFrame(final['classification_report']).T)
screenshot('07_mlflow_metrics_overview.png',
           '**Рисунок 4. Время обучения шести моделей и test accuracy/F1 выбранной модели.**')
screenshot('08_mlflow_test_metrics.png',
           '**Рисунок 5. Test precision/recall чемпиона и validation accuracy шести моделей.**')
display(Markdown('На графиках test-метрик показан один run — выбранный `nb_alpha_0.1`. '
                 'Остальные модели сравнивались по validation; test не использовался для их выбора. '
                 'Подписи столбцов в MLflow округлены, точные значения приведены в таблицах отчёта.'))
display(Image(filename=str(ROOT / 'results/figures/test_confusion_matrix.png'), width=600))
display(Markdown('**Рисунок 6. Матрица ошибок выбранного Pipeline на test.**'))
tn, fp, fn, tp = np.array(final['confusion_matrix']).ravel()
display(Markdown(
    f"На test из {final['rows']} сообщений правильно найдены **{tp}** spam, "
    f"пропущены **{fn}** spam; **{fp}** обычных SMS ошибочно помечены spam, "
    f"**{tn}** ham распознаны правильно. Recall {final['test_recall_spam']:.3%} "
    "показывает, что высокая точность ещё не означает обнаружение всего спама."))
""")
md("""
## 8. Registry, версии и alias champion

Два уже обученных Pipeline регистрируются под одним именем `sms-spam-filter`;
повторное обучение при регистрации не выполняется.
`v1` — лучший представитель другого семейства алгоритмов, `v2` — выбранный чемпион.
Номер Registry читается из ответа MLflow, а не предполагается по тегу образа.

Alias `champion` назначается `v2`. При экспорте alias разрешается один раз в конкретный
номер версии, затем загружается `models:/sms-spam-filter/<номер>`.
Смена alias сама по себе не меняет уже собранный образ или работающий контейнер.
""")
code("""
registered = []
for release, info in summary['releases'].items():
    version = client.get_model_version(MODEL_NAME, info['model_version'])
    registered.append({'релиз': release, 'Registry version': str(version.version),
                       'run_id': version.run_id, 'algorithm': info['algorithm'],
                       'pipeline_sha256': info['pipeline_sha256']})
display(pd.DataFrame(registered))
alias = client.get_model_version_by_alias(MODEL_NAME, 'champion')
print('champion → Registry version', alias.version)
assert str(alias.version) == summary['releases']['v2']['model_version']
screenshot('02_registry.png', '**Рисунок 7. Две версии одной модели и alias champion.**')
print('Метаданные экспорта v2:')
print(json.dumps(read_json(ROOT / 'models/v2/metadata.json'), ensure_ascii=False, indent=2))
""")
md("""
## 9. API и проверка запросов

`GET /health` — готовность загруженной модели и её версия;
`GET /metadata` — происхождение, версия, run_id и SHA-256;
`POST /predict` — пакетное предсказание;
`GET /docs` — Swagger UI.

Вход: `{"texts": ["See you soon", "Claim your free prize now!"]}`.
Допускается 1–100 строк, каждая содержит 1–5000 символов и не состоит целиком из пробелов.
Числа, `null`, пустые пакеты, лишние поля и повреждённый JSON возвращают **422**.
Ответ содержит метки `ham/spam`, вероятности spam, Registry version, release, run_id
и `latency_ms`. Эта вероятность не проходила отдельную калибровку.

Модель загружается один раз при старте. Проверяется SHA-256 и порядок классов;
повреждённый экспорт не позволяет запустить приложение.
API использует `joblib + scikit-learn`, не обращаясь к MLflow во время предсказания.
При равенстве вероятностей выбирается ham, как при `Pipeline.predict`.
""")
code("""
from sms_mlops.verify import run_local
local_checks = run_local()
local_v2 = local_checks['sequence'][1]
print('Корректный запрос:')
print(json.dumps(local_v2['example_request'], ensure_ascii=False, indent=2))
print('Ответ API:')
print(json.dumps(local_v2['example_response'], ensure_ascii=False, indent=2))
display(pd.DataFrame(local_v2['invalid_requests']))
print('Совпадение с экспортом проверено на', local_v2['parity_rows'], 'сообщениях для каждой версии.')
screenshot('03_swagger_valid.png', '**Рисунок 8. Реальный POST /predict через Swagger, ответ 200.**')
screenshot('04_swagger_invalid.png', '**Рисунок 9. Реальный пустой пакет и ответ 422.**')
""")
md("""
Корректность запроса означает соответствие контракту API. Она не гарантирует правильную
классификацию каждого текста. Качество определяется метриками на истинных метках test.
Smoke-test сравнивает ответы API с экспортированной моделью, включая все 1 032 test-SMS
и пять дополнительных строк (Unicode, неизвестные токены, предельная длина).
Для вероятностей проверяется `allclose(rtol=1e-10, atol=1e-12)`.

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/metadata
curl -sS -X POST http://127.0.0.1:8000/predict \\
  -H 'Content-Type: application/json' --data-binary @examples/valid.json
curl -i -X POST http://127.0.0.1:8000/predict \\
  -H 'Content-Type: application/json' --data-binary @examples/invalid_empty.json
python -m sms_mlops.verify --url http://127.0.0.1:8000 --release v2
```
""")
md("""
## 10. Сборка и запуск Docker, совпадение предсказаний

Dockerfile фиксирует Python 3.12.10 на Debian bookworm, устанавливает только runtime lock,
копирует API и конкретный экспорт. Контейнер работает от пользователя `appuser` (UID 10001).
Uvicorn слушает `0.0.0.0:8000` внутри контейнера; наружу порт публикуется на `127.0.0.1`.
`.dockerignore` исключает данные, SQLite, отчёт и виртуальное окружение из контекста сборки.

```bash
docker build --build-arg MODEL_RELEASE=v1 -t sms-spam:v1 .
docker build --build-arg MODEL_RELEASE=v2 -t sms-spam:v2 .
docker run --rm -d --name lab4-sms-demo -p 127.0.0.1:18000:8000 sms-spam:v1
python -m sms_mlops.verify --url http://127.0.0.1:18000 --release v1
docker stop lab4-sms-demo
```

Следующая ячейка показывает сохранённый протокол настоящего Docker-сценария.
Для повторного запуска контейнерных проверок установить `LAB4_RUN_DOCKER=1`
**перед запуском ядра**. Существующие образы проверяются без повторной сборки.
Для новой сборки с подробным build-протоколом выполнить отдельную команду
`python -m sms_mlops.docker_demo --require`.
""")
code("""
if os.environ.get('LAB4_RUN_DOCKER', '0') == '1':
    completed = subprocess.run([sys.executable, '-m', 'sms_mlops.docker_demo', '--require', '--no-build'],
                               cwd=ROOT, text=True, capture_output=True, timeout=900)
    print(completed.stdout)
    if completed.returncode:
        print(completed.stderr)
        raise RuntimeError('Контейнерные проверки завершились ошибкой')
docker_checks = read_json(ROOT / 'results/docker_checks.json')
print('Статус Docker:', docker_checks['status'])
print('Дата контейнерного протокола (UTC):', docker_checks['created_at'])
if docker_checks['status'] != 'passed':
    display(Markdown('**Docker не проверен:** ' + docker_checks.get('reason', 'нет протокола')))
else:
    display(pd.DataFrame([
        {'release': release, 'image_tag': item['tag'], 'image_id': item['id'],
         'размер, MiB': item['size_bytes'] / 1024**2}
        for release, item in docker_checks['images'].items()]))
    display(pd.DataFrame([
        {'release': item['release'], 'Registry version': item['model_version'],
         'проверено строк': item['parity_rows'],
         'макс. разность вероятностей': item['max_probability_difference'], 'статус': item['status']}
        for item in docker_checks['sequence']]))
""")
md("""
Полный stdout/stderr каждой Docker-команды и время её выполнения сохранены в
`results/docker_checks.json`. Image ID и RepoDigest фиксируют содержимое конкретного
образа; тег `v1/v2` сам по себе может быть перезаписан. Показанный `Size` — значение
из `docker image inspect`, не объём переданных по сети данных.
Базовый образ дополнительно закреплён по `FROM ...@sha256:...`: digest взят из
истории фактически выполненной сборки. Логи сборки доступны также в
`results/docker_build_v1.txt` и `results/docker_build_v2.txt`.
""")
md("""
## 11. Обновление и откат без повторного обучения

Сценарий по очереди запускает `sms-spam:v1 → sms-spam:v2 → sms-spam:v1`.
Перед каждым запуском переключается alias, затем проверяются `/metadata`, версия,
предсказания, вероятности и невалидные запросы. Экспортированные файлы уже существуют:
при откате не выполняются ни обучение, ни экспорт, ни пересборка `v1`.
Возврат исходных ответов проверяется автоматически. Возможна краткая недоступность
при остановке контейнера: бесшовное обновление в этой лабораторной не реализуется.

```bash
docker run --rm -d --name lab4-sms-demo -p 127.0.0.1:18000:8000 sms-spam:v2
python -m sms_mlops.verify --url http://127.0.0.1:18000 --release v2
docker stop lab4-sms-demo
docker run --rm -d --name lab4-sms-demo -p 127.0.0.1:18000:8000 sms-spam:v1
python -m sms_mlops.verify --url http://127.0.0.1:18000 --release v1
docker stop lab4-sms-demo
```

Число run и SHA-256 экспортов до/после сверяются. После демонстрации alias возвращается
выбранному по validation чемпиону `v2`, и эта версия запускается через Compose.
""")
code("""
if docker_checks['status'] == 'passed':
    display(pd.DataFrame([
        {'шаг': i + 1, 'образ': item['image_tag'], 'Registry version': item['model_version'],
         'champion во время шага': item['champion_alias'],
         'пример предсказаний': ', '.join(item['example_response']['predictions']), 'статус': item['status']}
        for i, item in enumerate(docker_checks['sequence'])]))
    proof = docker_checks['no_retraining']
    print('Число run до/после:', proof['runs_before'], '/', proof['runs_after'])
    print('Хеши моделей совпадают:', proof['hashes_before'] == proof['hashes_after'])
    assert proof['runs_before'] == proof['runs_after']
    assert proof['hashes_before'] == proof['hashes_after']
    display(pd.DataFrame([
        {'шаг': i + 1, 'версия': item['release'], **item['latency']}
        for i, item in enumerate(docker_checks['sequence'])]))
    display(Markdown('Медиана `handler_median_ms` измеряет вычисление в обработчике; '
                     '`roundtrip_median_ms` включает HTTP и сериализацию. '
                     'Десять последовательных запросов — smoke-измерение, '
                     'они не заменяют нагрузочный тест и не дают гарантии SLA.'))
""")
md("""
## 12. Docker Compose и healthcheck

```bash
MODEL_RELEASE=v2 docker compose up -d --no-build --wait --wait-timeout 90
docker compose ps
docker inspect --format '{{.State.Health.Status}}' "$(docker compose ps -q api)"
curl -f http://127.0.0.1:8000/health
```

Compose использует уже собранный образ. Проверка каждые 5 секунд запрашивает `/health`
изнутри контейнера, timeout 3 секунды, начальный период 10 секунд, до 5 ошибок.
`--wait` ожидает готовности по healthcheck. `restart: unless-stopped` перезапускает
завершившийся контейнер, но сам по себе не перезапускает контейнер только из-за unhealthy.
Остановка после работы: `docker compose down`.
""")
code("""
if docker_checks['status'] == 'passed':
    print('docker compose ps:')
    print(docker_checks['compose_ps'])
    print('Healthcheck:', docker_checks['compose_health'])
    assert docker_checks['compose_health'] == 'healthy'
    print('API Compose:', docker_checks['compose_checks']['status'])
    print('Registry version:', docker_checks['compose_checks']['model_version'])
    print('Docker и Compose:')
    print(docker_checks['docker_version'])
    print(docker_checks['compose_version'])
    screenshot('05_health.png', '**Рисунок 10. Настоящий /health сервиса в Compose.**')
""")
md("""
## 13. Выводы

Построен воспроизводимый проект варианта 2: исходные байты UCI, журнал очистки, seed,
индексы разбиения, зависимости, MLflow runs, артефакты и два экспортированных Pipeline
сохранены в проекте. Отделение validation от test предотвращает выбор модели по
итоговой оценке, а Pipeline предотвращает обучение TF-IDF на отложенных данных.

Уменьшение alpha улучшило recall NB на данном разбиении; рост C повысил recall
LogisticRegression. Выбор сделан по F1, с отдельным анализом precision/recall.
Высокая precision сочетается с пропусками спама: модель не является безошибочным фильтром.
Исторический корпус и один случайный split ограничивают переносимость оценки.

Model Registry отделяет запуск обучения от версии модели, alias задаёт выбор для следующего
экспорта, а Docker-образ хранит конкретный снимок. Проверка полного пакета test-SMS
подтверждает соответствие API экспорту; проверка некорректного ввода подтверждает контракт.
Статусы контейнерной части и итоговые метрики ниже читаются из реальных протоколов.
""")
code("""
display(Markdown(
    f"**Итог:** {winner['algorithm']} ({winner['parameter']}={winner['value']:g}); "
    f"test F1 = **{final['test_f1_spam']:.4f}**, precision = **{final['test_precision_spam']:.4f}**, "
    f"recall = **{final['test_recall_spam']:.4f}**. "
    f"Локальные проверки: **{local_checks['status']}**; "
    f"Docker: **{docker_checks['status']}**; Compose healthcheck: "
    f"**{docker_checks.get('compose_health', 'не проверен')}**."))
""")
md("""
## 14. Список использованных источников

1. Методические материалы «Лабораторная работа 4. MLOps», PDF, предоставленный к заданию,
   особенно разделы 7–17 и индивидуальный вариант 2. Пример Wine адаптирован под текст SMS.
2. Almeida, T., Hidalgo, J. SMS Spam Collection. UCI Machine Learning Repository, 2011.
   [Страница и лицензия набора](https://archive.ics.uci.edu/dataset/228/sms+spam+collection),
   [DOI](https://doi.org/10.24432/C5CC84).
3. [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry/) и
   [sklearn API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html).
4. [scikit-learn 1.6: Pipeline](https://scikit-learn.org/1.6/modules/compose.html),
   [text feature extraction](https://scikit-learn.org/1.6/modules/feature_extraction.html#text-feature-extraction),
   [model evaluation](https://scikit-learn.org/1.6/modules/model_evaluation.html).
5. [Dockerfile reference](https://docs.docker.com/reference/dockerfile/),
   [Compose services and healthcheck](https://docs.docker.com/reference/compose-file/services/),
   [compose up](https://docs.docker.com/reference/cli/docker/compose/up/),
   [Docker Desktop WSL integration](https://docs.docker.com/desktop/features/wsl/).
6. [FastAPI request body](https://fastapi.tiangolo.com/tutorial/body/),
   [Pydantic validators](https://docs.pydantic.dev/latest/concepts/validators/).

Документация latest может описывать другие версии: воспроизводимое окружение этой
работы задаётся локальными lock-файлами, фактические результаты — сохранёнными протоколами.
""")
md("""
## 15. Приложение: исходные файлы и протокол выполнения

Ниже приводятся полные исходные файлы проекта, зависимости, Dockerfile, Compose и
JSON-примеры. SQLite, исходные данные, бинарные Pipeline и изображения приложены
отдельными файлами в `lab4`.
Скриншоты Registry и графиков метрик отражают состояние локального MLflow; остальные
снимки получены Playwright с реальных локальных страниц. Источники и SHA-256
изображений записаны в `screenshots/capture_metadata.json`. Оригинальные PNG сохранены
без изменений в каталоге `screenshots`.
При повторном эксперименте текст численных выводов обновляется из сохранённых результатов.
""")
code("""
source_paths = [Path('README.md'), Path('requirements.in'), Path('requirements.txt'),
                Path('requirements-runtime.in'), Path('requirements-runtime.txt'),
                Path('Dockerfile'), Path('compose.yaml'), Path('.dockerignore')]
source_paths += sorted(Path('sms_mlops').glob('*.py'))
source_paths += sorted(Path('scripts').glob('*.py'))
source_paths += sorted(Path('tests').glob('*.py'))
source_paths += sorted(Path('examples').glob('*.json'))
for path in source_paths:
    # The report generator itself is linked to avoid recursively embedding the whole report.
    if path.name == 'build_report.py':
        display(Markdown(f'Генератор исходного ноутбука: [{path.as_posix()}]({path.as_posix()}).'))
        continue
    language = 'python' if path.suffix == '.py' else 'json' if path.suffix == '.json' else 'text'
    display(Markdown(f'### {path.as_posix()}\\n\\n```{language}\\n' + path.read_text() + '\\n```'))
""")
code("""
if docker_checks['status'] == 'passed':
    for entry in docker_checks['commands']:
        print('$', ' '.join(entry['argv']))
        print('exit =', entry['returncode'], '; seconds =', round(entry['seconds'], 3))
        if entry['stdout'].strip():
            print(entry['stdout'])
        if entry['stderr'].strip():
            print(entry['stderr'])
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.update({
    "kernelspec": {"display_name": "Python (lab4 .venv)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.3"},
})
nbf.write(notebook, ROOT / "lab4_mlops.ipynb")
print(f"Created {len(cells)} cells: {ROOT / 'lab4_mlops.ipynb'}")
