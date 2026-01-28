# HybridQA Multi-Agent Service

Сервис на FastAPI + LangGraph для вопросов по таблицам HybridQA с multi-hop (таблица + Wikipedia passages).

## Возможности
- Табличные запросы через инструменты (get_row/lookup_cell/filter и т.д.)
- RAG по таблице и связанным passage
- Multi-hop вопросы (birthplace → country)
- Валидный JSON-ответ с function_calls и sources
- Docker Compose запуск

## Запуск (Docker)
```bash
cd agents_practice
docker compose up --build

##Сервис будет доступен:
- http://localhost:8000/health
- http://localhost:8000/docs

##Данные
- Большие данные и модели не входят в репозиторий.
- При запуске контейнер скачивает модели и кэши в data/.
