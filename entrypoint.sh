#!/usr/bin/env bash
set -e

APP_DIR="/app"

mkdir -p "${APP_DIR}/models"
mkdir -p "${APP_DIR}/outputs"

# Если нет моделей, запускаем обучение
if [ -z "$(ls -A ${APP_DIR}/models 2>/dev/null)" ]; then
  echo "Модели не найдены — запускаем обучение..."
  python "${APP_DIR}/src/train.py"
else
  echo "Найдены модели в ${APP_DIR}/models — пропускаем обучение."
fi

# Запускаем FastAPI через uvicorn
exec uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 1