# План доведения fasteval до образцового hiring-проекта

## Цель

Сделать проект надёжным, воспроизводимым и удобным для нового разработчика: заявленные возможности должны работать end-to-end, критические сценарии должны быть покрыты тестами, а качество должно автоматически проверяться в CI.

## Definition of Done

- [ ] Команда из README запускается в чистом окружении после установки зависимостей.
- [ ] Structured output действительно передаётся всем совместимым провайдерам.
- [ ] CLI, registry и список адаптеров используют единый список провайдеров.
- [ ] Ошибки одного запуска не ломают всю матрицу и представлены структурированно.
- [ ] Основные сценарии покрыты unit-, integration- и CLI-тестами.
- [ ] В CI проходят тесты, lint, форматирование, type checking и coverage threshold.
- [ ] Документация позволяет добавить модель или провайдера без чтения всего исходного кода.

## P0 — исправить функциональные дефекты

### 1. Провести structured output через весь pipeline

- [ ] Передать `config["structured_output"]` из `runner.run()` в `_call_model()`.
- [ ] Передать `response_schema` в `complete()` и `processor.send_message()`.
- [ ] Проверить поддержку схемы в OpenAI, Gemini и OpenRouter.
- [ ] Явно определить поведение для провайдера без structured output.
- [ ] Добавить mock-поведение, возвращающее валидный структурированный JSON.
- [ ] Добавить end-to-end тест: CLI → runner → mock provider → `run.json`.
- [ ] Добавить тест, доказывающий, что schema реально попала в запрос провайдера.

Критерий готовности: запуск с `--structured-output` формирует и валидирует структурированный ответ, а не только сохраняет схему в отчёте.

### 2. Синхронизировать провайдеры

- [ ] Добавить `openrouter` в допустимые значения CLI.
- [ ] Удалить `anthropic` из CLI либо реализовать его адаптер.
- [ ] Убрать дублирование списка провайдеров из CLI и registry.
- [ ] Сделать CLI-валидацию провайдера на основе registry.
- [ ] Добавить тест для каждого доступного через CLI провайдера.

Критерий готовности: любой провайдер из `list_providers()` либо доступен через CLI, либо явно исключён с понятным сообщением.

### 3. Исправить зависимости и чистую установку

- [ ] Проверить все импорты в provider adapters в чистом virtualenv.
- [ ] Добавить в optional dependency group все реально используемые библиотеки, включая `asgiref`, если он нужен.
- [ ] Разделить зависимости по провайдерам, если это возможно.
- [ ] Проверить `pip install -e '.[native]'` и `pip install -e '.[dev]'` с нуля.
- [ ] Добавить smoke test импорта пакета и registry.
- [ ] Зафиксировать поддерживаемые версии Python и SDK.

Критерий готовности: чистая установка не требует ручной установки скрытых зависимостей.

### 4. Проверить расчёт метрик и стоимости

- [ ] Исправить расчёт cached input tokens и cached write tokens.
- [ ] Проверить, что стоимость не считается дважды для cached tokens.
- [ ] Определить, включается ли reasoning в output cost или считается отдельно.
- [ ] Исправить `time_to_first_token_ms`: не выдавать полную latency за TTFT без streaming.
- [ ] Для не-streaming режима возвращать `None` для TTFT.
- [ ] Добавить timeout на вызов провайдера.
- [ ] Добавить unit-тесты на нулевые, отсутствующие и частично известные usage values.

Критерий готовности: метрики и стоимость имеют документированную семантику и подтверждены тестами.

## P1 — укрепить архитектуру и качество

### 5. Ввести типизированную конфигурацию

- [ ] Заменить свободный `dict[str, Any]` на dataclass или Pydantic-модели.
- [ ] Вынести `RunConfig`, `ModelConfig` и `PricingConfig`.
- [ ] Валидировать пути к файлам до запуска моделей.
- [ ] Валидировать несовместимые комбинации аргументов.
- [ ] Добавить поддержку пользовательского registry через CLI.
- [ ] Убрать захардкоженный путь к `config/models.toml`.

Критерий готовности: ошибки конфигурации обнаруживаются до сетевых вызовов и содержат понятное описание.

### 6. Унифицировать provider API

- [ ] Оставить один публичный request model.
- [ ] Оставить один публичный result model.
- [ ] Согласовать типы `content`, usage fields и finish reasons.
- [ ] Убрать лишние дублирующие dataclass-модели либо скрыть их как внутренние.
- [ ] Описать контракт адаптера в документации.
- [ ] Добавить contract tests для всех адаптеров.
- [ ] Явно разделить sync, async и streaming API.

Критерий готовности: добавление нового провайдера требует реализации одного документированного контракта.

### 7. Улучшить обработку ошибок

- [ ] Ввести иерархию ошибок конфигурации, провайдера, валидации и файлов.
- [ ] Сохранять в результате тип ошибки и безопасное сообщение.
- [ ] Не сохранять API keys, request headers и чувствительные данные в raw response.
- [ ] Добавить retry policy с документированными условиями.
- [ ] Добавить ограничение параллелизма.
- [ ] Добавить graceful handling пустого registry и неизвестного provider.
- [ ] Сохранить traceback только в debug-режиме или отдельном локальном логе.

Критерий готовности: ошибка одного model run диагностируема и не скрывает причину сбоя.

### 8. Привести код к единому стилю

- [ ] Запустить Black или Ruff formatter.
- [ ] Настроить Ruff для import order, unused imports и базовых ошибок.
- [ ] Добавить mypy или pyright.
- [ ] Убрать отключённый и мёртвый код.
- [ ] Разбить длинные функции и выражения в runner/CLI.
- [ ] Убрать неиспользуемые поля и импорты.
- [ ] Добавить docstrings только для публичных API и сложной бизнес-логики.

Критерий готовности: стиль и базовые ошибки проверяются одной командой.

## P1 — тестовая стратегия

### Unit tests

- [ ] `_load_registry()` валидирует корректный и повреждённый TOML.
- [ ] `_expand_reasoning_efforts()` корректно обрабатывает строки, пустые значения и default.
- [ ] `RunResult.total_cost_usd` обрабатывает `None`.
- [ ] `_escape()` защищает HTML-спецсимволы.
- [ ] Structured schema parser покрывает quotes, commas, arrays, optional fields и malformed input.
- [ ] Provider registry корректно сообщает неизвестный provider.

### Integration tests

- [ ] Mock run с одной моделью.
- [ ] Mock run с несколькими reasoning efforts.
- [ ] Частичный сбой одной модели.
- [ ] Пустой registry.
- [ ] Structured output.
- [ ] File input.
- [ ] Image input.
- [ ] JSON и HTML report.

### CLI tests

- [ ] `--help`.
- [ ] Отсутствующий `--prompt`.
- [ ] Неизвестный provider.
- [ ] Невалидная schema.
- [ ] Exit code `0` при полном успехе.
- [ ] Exit code `1` при частичной или полной ошибке.
- [ ] Вывод содержит валидный JSON.

### Quality gates

- [ ] Покрытие не ниже 85% для core-модулей.
- [ ] Отдельно измеряется покрытие provider adapters.
- [ ] Нет flaky tests.
- [ ] Тесты не требуют реальных API keys.
- [ ] Интеграционные тесты с API запускаются только по отдельному opt-in marker.

## P2 — сделать проект сильным портфолио

### Evaluation capabilities

- [ ] Поддержать несколько повторных запусков одного кейса.
- [ ] Добавить dataset input: JSONL/CSV.
- [ ] Добавить evaluators: exact match, JSON validation, regex, custom Python evaluator.
- [ ] Добавить LLM-as-a-judge как отдельный opt-in evaluator.
- [ ] Сравнивать модели по качеству, latency, стоимости и стабильности.
- [ ] Добавить aggregate summary в JSON и HTML.

### Reporting

- [ ] Добавить фильтрацию и сортировку в HTML.
- [ ] Показать tokens, cost, latency и finish reason.
- [ ] Добавить статус success/error для каждой строки.
- [ ] Добавить экспорт Markdown и CSV.
- [ ] Добавить ссылку на исходные параметры запуска.
- [ ] Добавить версию fasteval и registry в report metadata.

### Developer experience

- [ ] Добавить `CONTRIBUTING.md`.
- [ ] Добавить `CHANGELOG.md`.
- [ ] Добавить `SECURITY.md`.
- [ ] Добавить `CODE_OF_CONDUCT.md`, если проект публикуется.
- [ ] Добавить `Makefile` или task runner с командами `test`, `lint`, `format`, `typecheck`.
- [ ] Добавить GitHub Actions CI.
- [ ] Добавить badges для CI, Python versions и coverage.
- [ ] Добавить пример custom provider.

## Предлагаемый порядок коммитов

1. `Fix structured output pipeline`
2. `Align provider registry and CLI`
3. `Declare native dependencies`
4. `Add runner and CLI integration tests`
5. `Correct usage and cost metrics`
6. `Introduce typed run configuration`
7. `Unify provider result contract`
8. `Add lint typecheck and CI`
9. `Improve documentation and examples`
10. `Add dataset evaluation and aggregate reports`

## Финальная проверка перед публикацией

- [ ] Новый разработчик устанавливает проект по README без устных пояснений.
- [ ] Demo запускается без API key через mock provider.
- [ ] Все команды README проверены буквально.
- [ ] Нет секретов, локальных артефактов и временных файлов в git.
- [ ] `pytest`, lint, formatter check и typecheck проходят.
- [ ] Coverage опубликован в CI.
- [ ] У каждой публичной возможности есть тест и пример.
- [ ] Ограничения проекта честно описаны в README.
- [ ] В README есть архитектура и инструкция расширения.
