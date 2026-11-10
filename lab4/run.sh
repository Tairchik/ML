#!/usr/bin/env bash
# WSL / Ubuntu launcher. All paths are resolved relative to this file.
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/.venv/bin/python"
PROJECT_NAME="${COMPOSE_PROJECT_NAME:-lab4-sms-spam}"
API_PORT="${API_PORT:-8000}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
TRACKING_URI="sqlite:///$PROJECT_DIR/mlflow-data/mlflow.db"
RUNTIME_DIR="$PROJECT_DIR/.runtime/runner/$PROJECT_NAME"
PID_FILE="$RUNTIME_DIR/mlflow.pid"
MLFLOW_LOG="$RUNTIME_DIR/mlflow.log"

usage() {
    cat <<'HELP'
ЛР4: сервис фильтрации SMS-спама. Запускать можно из любого каталога.

  ./run.sh                         Запустить API v2 и MLflow (как up).
  ./run.sh up [v1|v2] [--no-build]  Поднять Compose и MLflow; по умолчанию v2.
  ./run.sh down                    Остановить Compose и MLflow этого скрипта.
  ./run.sh status                  Показать состояние API и MLflow.
  ./run.sh logs [api|mlflow]        Смотреть логи (Ctrl+C завершает просмотр).
  ./run.sh setup                   Создать .venv и установить зависимости.
  ./run.sh train [--force]          Проверить сохранённые модели / обучить заново.
  ./run.sh test                    Локальные проверки API и pytest.
  ./run.sh check [v1|v2]            Проверить API, работающий на API_PORT.
  ./run.sh demo [--no-build]        Полная демонстрация Docker v1 → v2 → v1.
  ./run.sh mlflow                  Запустить только интерфейс MLflow в фоне.
  ./run.sh report                  Выполнить ноутбук и обновить IPYNB / HTML.
  ./run.sh help                    Эта справка.

API_PORT=8000 и MLFLOW_PORT=5000 можно переопределить перед командой.
Первый запуск автоматически готовит окружение; train --force создаёт новые runs.
HELP
}

die() { printf 'Ошибка: %s\n' "$*" >&2; exit 1; }
note() { printf '\n%s\n' "$*"; }

validate_config() {
    [[ "$PROJECT_NAME" =~ ^[a-z0-9][a-z0-9_-]*$ ]] || die 'Некорректный COMPOSE_PROJECT_NAME.'
    local port
    for port in "$API_PORT" "$MLFLOW_PORT"; do
        [[ "$port" =~ ^[0-9]{1,5}$ ]] && ((10#$port >= 1 && 10#$port <= 65535)) ||
            die 'API_PORT и MLFLOW_PORT должны быть числами от 1 до 65535.'
    done
    API_PORT="$((10#$API_PORT))"
    MLFLOW_PORT="$((10#$MLFLOW_PORT))"
    [[ "$API_PORT" != "$MLFLOW_PORT" ]] || die 'Порты API и MLflow должны различаться.'
    export API_PORT
    export MPLCONFIGDIR="$PROJECT_DIR/.runtime/matplotlib"
    export JUPYTER_CONFIG_DIR="$PROJECT_DIR/.runtime/jupyter-config"
    export JUPYTER_DATA_DIR="$PROJECT_DIR/.runtime/jupyter-data"
    export JUPYTER_RUNTIME_DIR="$PROJECT_DIR/.runtime/jupyter-runtime"
    export IPYTHONDIR="$PROJECT_DIR/.runtime/ipython"
}

ensure_environment() {
    if [[ ! -x "$PYTHON" ]]; then
        command -v python3 >/dev/null || die 'Нужен Python 3.12 (python3).'
        python3 -c 'import sys; assert sys.version_info[:2] == (3, 12)' ||
            die 'Для зафиксированного окружения нужен Python 3.12.'
        note 'Создаю lab4/.venv…'
        python3 -m venv "$PROJECT_DIR/.venv" ||
            die 'Не удалось создать .venv. Проверьте установку python3-venv в Ubuntu.'
    fi
    "$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12)' ||
        die 'Существующая lab4/.venv должна использовать Python 3.12.'
    # Check the complete lock locally; avoid a network request on every launch.
    if ! "$PYTHON" - "$PROJECT_DIR/requirements.txt" <<'PY'
import importlib.metadata as metadata
import sys
from pathlib import Path

missing = []
for line in Path(sys.argv[1]).read_text().splitlines():
    if not line or line.startswith('#'):
        continue
    name, expected = line.split('==', 1)
    try:
        actual = metadata.version(name)
    except metadata.PackageNotFoundError:
        actual = None
    if actual != expected:
        missing.append(f'{name}=={expected}')
if missing:
    print('Нужно установить: ' + ', '.join(missing))
    raise SystemExit(1)
PY
    then
        note 'Устанавливаю зависимости из requirements.txt…'
        "$PYTHON" -m pip install -r "$PROJECT_DIR/requirements.txt"
        "$PYTHON" -m pip check
    fi
}

ensure_models() {
    # Reuses existing runs by default; the training module itself validates exports.
    "$PYTHON" -m sms_mlops.train
}

require_docker() {
    command -v docker >/dev/null || die 'Docker недоступен. Включите Docker Desktop → WSL Integration → Ubuntu.'
    if ! docker info >/dev/null 2>&1; then
        docker info >&2 || true
        die 'Нет доступа к Docker engine. Проверьте Docker Desktop и интеграцию WSL.'
    fi
    docker compose version >/dev/null || die 'Нужен Docker Compose plugin.'
}

compose() {
    docker compose --project-name "$PROJECT_NAME" --file "$PROJECT_DIR/compose.yaml" "$@"
}

release_value() {
    [[ "$1" == 'v1' || "$1" == 'v2' ]] || die 'Выберите релиз v1 или v2.'
}

mlflow_ready() {
    "$PYTHON" - "$MLFLOW_PORT" "$PROJECT_DIR" <<'PY'
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

url = (f'http://127.0.0.1:{sys.argv[1]}/api/2.0/mlflow/experiments/get-by-name'
       '?experiment_name=lab4-sms-spam')
try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=1) as response:
        experiment = json.load(response)['experiment']
    artifact = experiment['artifact_location']
    if artifact.startswith('file:'):
        artifact = urllib.parse.unquote(urllib.parse.urlparse(artifact).path)
    expected = Path(sys.argv[2]) / 'mlflow-data/artifacts'
    raise SystemExit(0 if Path(artifact).resolve() == expected.resolve() else 1)
except (OSError, ValueError, KeyError):
    raise SystemExit(1)
PY
}

managed_pid() {
    [[ -r "$PID_FILE" ]] || return 1
    local pid command_line group
    read -r pid < "$PID_FILE" || return 1
    [[ "$pid" =~ ^[1-9][0-9]*$ && -r "/proc/$pid/cmdline" ]] || return 1
    command_line="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
    [[ "$command_line" == *' -m mlflow server '* && "$command_line" == *"$TRACKING_URI"* ]] || return 1
    group="$(ps -o pgid= -p "$pid")"
    [[ "${group//[[:space:]]/}" == "$pid" ]] || return 1
    printf '%s\n' "$pid"
}

start_mlflow() {
    if mlflow_ready; then
        note "MLflow доступен: http://127.0.0.1:$MLFLOW_PORT"
        return
    fi
    local pid attempt
    if pid="$(managed_pid)"; then
        die "Этот скрипт уже запустил MLflow (PID $pid). Проверьте MLFLOW_PORT и логи."
    fi
    command -v setsid >/dev/null || die 'Для фонового запуска MLflow нужна команда setsid (util-linux).'
    mkdir -p -- "$RUNTIME_DIR"
    "$PYTHON" -m sms_mlops.relocate
    note "Запускаю MLflow на порту $MLFLOW_PORT…"
    # Separate process group lets down terminate both the CLI and its Gunicorn worker.
    nohup setsid "$PYTHON" -m mlflow server \
        --backend-store-uri "$TRACKING_URI" \
        --default-artifact-root "$PROJECT_DIR/mlflow-data/artifacts" \
        --no-serve-artifacts --host 127.0.0.1 --port "$MLFLOW_PORT" --workers 1 \
        > "$MLFLOW_LOG" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "$pid" > "$PID_FILE"
    for attempt in {1..30}; do
        if mlflow_ready; then
            note "MLflow: http://127.0.0.1:$MLFLOW_PORT (логи: ./run.sh logs mlflow)"
            return
        fi
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    tail -n 30 "$MLFLOW_LOG" >&2
    die 'MLflow не стал готов. Проверьте логи и свободен ли MLFLOW_PORT.'
}

stop_mlflow() {
    local pid attempt
    if ! pid="$(managed_pid)"; then
        note 'Управляемый этим скриптом MLflow не запущен.'
        return
    fi
    note "Останавливаю MLflow (группа процессов $pid)…"
    kill -TERM -- "-$pid"
    for attempt in {1..50}; do
        if ! kill -0 -- "-$pid" 2>/dev/null; then
            rm -f -- "$PID_FILE"
            return
        fi
        sleep 0.2
    done
    die 'MLflow ещё завершает работу. Проверьте ./run.sh logs mlflow.'
}

main() {
    local action="${1:-up}" release='v2' build='--build' target
    if (($#)); then shift; fi
    case "$action" in
        help|-h|--help) usage; return ;;
        setup|up|down|status|logs|train|test|check|demo|mlflow|report) ;;
        *) usage >&2; die "Неизвестная команда: $action" ;;
    esac
    validate_config
    case "$action" in
        setup)
            (($# == 0)) || die 'setup не принимает аргументы.'
            ensure_environment
            "$PYTHON" -m pip check
            note "Окружение готово: $PYTHON"
            ;;
        up)
            for target in "$@"; do
                case "$target" in
                    v1|v2) release="$target" ;;
                    --no-build) build='--no-build' ;;
                    *) die "Неизвестный аргумент up: $target" ;;
                esac
            done
            require_docker
            ensure_environment
            ensure_models
            export MODEL_RELEASE="$release"
            note "Запускаю API $release через Docker Compose…"
            compose up -d "$build" --wait --wait-timeout 90
            start_mlflow
            note "API / Swagger: http://127.0.0.1:$API_PORT/docs"
            note 'Остановка: ./run.sh down'
            ;;
        down)
            (($# == 0)) || die 'down не принимает аргументы.'
            stop_mlflow
            require_docker
            compose down
            ;;
        status)
            (($# == 0)) || die 'status не принимает аргументы.'
            require_docker
            compose ps
            if [[ -x "$PYTHON" ]] && mlflow_ready; then
                note "MLflow готов: http://127.0.0.1:$MLFLOW_PORT"
            else
                note "MLflow не отвечает на порту $MLFLOW_PORT."
            fi
            ;;
        logs)
            (($# <= 1)) || die 'Использование: logs [api|mlflow].'
            case "${1:-api}" in
                api) require_docker; compose logs --follow --tail 100 api ;;
                mlflow)
                    [[ -f "$MLFLOW_LOG" ]] || die 'Нет лога: этот скрипт ещё не запускал MLflow.'
                    tail -n 100 -f "$MLFLOW_LOG"
                    ;;
                *) die 'Выберите логи api или mlflow.' ;;
            esac
            ;;
        train)
            ensure_environment
            "$PYTHON" -m sms_mlops.train "$@"
            ;;
        test)
            (($# == 0)) || die 'test не принимает аргументы.'
            ensure_environment
            ensure_models
            "$PYTHON" -m sms_mlops.verify
            "$PYTHON" -m pytest -q
            ;;
        check)
            (($# <= 1)) || die 'Использование: check [v1|v2].'
            release="${1:-v2}"
            release_value "$release"
            ensure_environment
            "$PYTHON" -m sms_mlops.verify --url "http://127.0.0.1:$API_PORT" --release "$release"
            ;;
        demo)
            (($# == 0)) || [[ "$#" == 1 && "$1" == '--no-build' ]] || die 'Использование: demo [--no-build].'
            [[ "$API_PORT" == 8000 ]] || die 'Демонстрация docker_demo использует API_PORT=8000 и порт 18000.'
            [[ "$PROJECT_NAME" == 'lab4-sms-spam' ]] || die 'demo использует стандартный Compose-проект lab4-sms-spam.'
            require_docker
            ensure_environment
            ensure_models
            "$PYTHON" -m sms_mlops.docker_demo --require "$@"
            ;;
        mlflow)
            (($# == 0)) || die 'mlflow не принимает аргументы.'
            ensure_environment
            ensure_models
            start_mlflow
            ;;
        report)
            (($# == 0)) || die 'report не принимает аргументы.'
            ensure_environment
            ensure_models
            "$PYTHON" scripts/execute_report.py
            ;;
    esac
}

main "$@"
