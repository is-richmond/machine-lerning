```markdown
# House Price Prediction (Прогнозирование цен на жильё)

Кратко
-----
Проект прогнозирует медианную стоимость жилья на основе признаков (California Housing). Реализованы:
- Предобработка данных и обучение моделей (Linear Regression, Random Forest, LightGBM).
- REST API на FastAPI для предсказаний.
- Docker / docker-compose для лёгкого развёртывания (entrypoint автоматически тренирует модели при первом запуске).

Структура репозитория
- src/ — исходники (train.py, predict.py, api.py)
- models/ — сохранённые модели (.joblib)
- outputs/ — метрики, примеры
- docs/final_report.md — финальный отчёт (markdown)
- Dockerfile, docker-compose.yml, entrypoint.sh — для контейнеризации

Быстрый старт (локально)
1. Создайте виртуальное окружение и установите зависимости:
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt

2. Обучите модели:
   python src/train.py

3. Запустите API:
   uvicorn src.api:app --host 0.0.0.0 --port 8000

4. Документация API (Swagger): http://localhost:8000/docs

Через Docker (рекомендуется)
1. Собрать и запустить:
   docker-compose up --build

2. Открыть API:
   http://localhost:8000/docs

Формат запроса к /predict (POST JSON)
{
  "model": "rf",            # "rf" | "lr" | "lgbm"
  "features": {
    "MedInc": 3.87,
    "HouseAge": 41.0,
    "AveRooms": 6.984127,
    "AveBedrms": 1.023810,
    "Population": 322.0,
    "AveOccup": 2.555556,
    "Latitude": 37.88,
    "Longitude": -122.23
  }
}

Ответ:
{
  "model": "rf",
  "prediction": 2.653,    # предсказание (в единицах набора данных)
  "unit_notes": "target is MedianHouseValue in dataset units",
  "metrics": { ... }      # метрики обученной модели (из outputs/metrics.json)
}

Контакты команды
- Группа: SIS-2208
- Участники: Abaidullayev Kablan, Dilovarov Nariman, Suleimanov Mintemir

Лицензия
- MIT
```