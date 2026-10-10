# Задачи SinglePipe

Список всех задач, которые SinglePipe собирается поддерживать, и их состояние. Это память проекта о задачах: заглушек в коде нет (см. `architecture.md`, раздел 7), поэтому всё, что нужно не забыть, пишется сюда.

Источники истины:

- **Какие задачи известны** — `TaskType` в `singlepipe/tasks.py`. Значения совпадают со строками задач HF `pipeline`, кроме `TABULAR_*` и `OCR`.
- **Какие задачи реализованы** — реестр схем. Член `TaskType` без регистрации в реестре — это задача «в планах», а не «есть».
- Код старых заглушек (только `NotImplementedError`, полей в них не было) остался в теге `v0.0.1`.

Статусы:

- **Есть** — схемы на контракте, зарегистрированы, есть юнит- и интеграционные тесты.
- **В работе** — схемы есть, но ещё не на основах или не доведены.
- **В плане** — привязано к версии из дорожной карты.
- **Идея** — член `TaskType` есть, планов нет.

Когда задача меняет статус, правится эта таблица в том же коммите.

---

## Картинки

| `TaskType` | Значение | Статус | Заметки |
|---|---|---|---|
| `IMAGE_CLASSIFICATION` | `image-classification` | В работе | Схемы и тесты есть, интеграционный тест на крошечной ViT. Перевод на `ImageTaskInput` + `PredictionList[Prediction]` — шаг 6 плана. |
| `OBJECT_DETECTION` | `object-detection` | В работе | Схемы есть, свой `Detection` и фильтры — дубли, удаляются в шаге 6. Остался проблемный `__iter__`. `output_context` с размерами картинки. |
| `IMAGE_SEGMENTATION` | `image-segmentation` | Идея | Понадобится тип-значение для маски. |
| `DEPTH_ESTIMATION` | `depth-estimation` | Идея | Выход — карта глубины, тоже значение, а не запись. |
| `IMAGE_EMBEDDING` | `image-feature-extraction` | Идея | Выход — вектор; общий тип с `TEXT_EMBEDDING`. |
| `ZERO_SHOT_IMAGE_CLASSIFICATION` | `zero-shot-image-classification` | Идея | Вход — картинка + набор меток, поэтому `ImageTaskInput` как есть не подойдёт. Выход совпадает с классификацией. |
| `ZERO_SHOT_OBJECT_DETECTION` | `zero-shot-object-detection` | Идея | Вход — картинка + метки, выход как у детекции. |
| `MASK_GENERATION` | `mask-generation` | Идея | Маски, как у сегментации. |
| `KEYPOINT_MATCHING` | `keypoint-matching` | Идея | Вход — две картинки. |

## Текст

| `TaskType` | Значение | Статус | Заметки |
|---|---|---|---|
| `TEXT_CLASSIFICATION` | `text-classification` | Идея | Выход — `PredictionList[Prediction]`, как у картиночной классификации. |
| `TOKEN_CLASSIFICATION` | `token-classification` | Идея | NER; у предсказания появятся позиции в тексте. |
| `TEXT_GENERATION` | `text-generation` | Идея | |
| `TEXT2TEXT_GENERATION` | `text2text-generation` | Идея | При реализации сверить, что pipeline есть в актуальной версии `transformers`. |
| `FILL_MASK` | `fill-mask` | Идея | Выход — список вариантов со скором. |
| `SUMMARIZATION` | `summarization` | Идея | Сверить с актуальной версией `transformers`. |
| `TRANSLATION` | `translation` | Идея | Сверить с актуальной версией `transformers`; языковая пара в аргументах. |
| `QUESTION_ANSWERING` | `question-answering` | Идея | Вход — вопрос + контекст. |
| `TEXT_EMBEDDING` | `feature-extraction` | Идея | Общий тип выхода с `IMAGE_EMBEDDING`. |
| `ZERO_SHOT_TEXT_CLASSIFICATION` | `zero-shot-classification` | Идея | Вход — текст + метки. |

## Аудио

| `TaskType` | Значение | Статус | Заметки |
|---|---|---|---|
| `AUDIO_CLASSIFICATION` | `audio-classification` | Идея | Нужен тип-значение `Audio`, по образцу `Image`. |
| `SPEECH_TO_TEXT` | `automatic-speech-recognition` | Идея | |
| `TEXT_TO_AUDIO` | `text-to-audio` | Идея | Выход — `Audio`. |
| `ZERO_SHOT_AUDIO_CLASSIFICATION` | `zero-shot-audio-classification` | Идея | Вход — аудио + метки. |

## Видео

| `TaskType` | Значение | Статус | Заметки |
|---|---|---|---|
| `VIDEO_CLASSIFICATION` | `video-classification` | Идея | Нужен тип-значение `Video`. |

## Мультимодальные

| `TaskType` | Значение | Статус | Заметки |
|---|---|---|---|
| `IMAGE_TEXT_TO_TEXT` | `image-text-to-text` | Идея | Вход — `Image` + текст. |
| `DOCUMENT_QUESTION_ANSWERING` | `document-question-answering` | Идея | Вход — `Image` документа + вопрос. |
| `TABLE_QUESTION_ANSWERING` | `table-question-answering` | Идея | Вход — таблица + вопрос; тип таблицы общий с табличными задачами. |

## Табличные

| `TaskType` | Значение | Статус | Заметки |
|---|---|---|---|
| `TABULAR_CLASSIFICATION` | `tabular-classification` | В плане (v0.6) | Бэкенды sklearn / xgboost / lightgbm / catboost, не HF. **До начала решить батчи:** табличка батчевая по природе, а API сейчас «один вход — один выход». |
| `TABULAR_REGRESSION` | `tabular-regression` | В плане (v0.6) | То же. Выход — число, не `Prediction`. |

## OCR

| `TaskType` | Значение | Статус | Заметки |
|---|---|---|---|
| `OCR` | — | Идея | **Члена нет в `TaskType`**: в файле есть комментарий `# OCR`, но под ним пусто, хотя на него ссылаются docstring `TaskType` и удалённая заглушка `ocr_schemas/ocr.py`. HF pipeline нет, нужен свой адаптер. |

---

## Сквозные заметки

- **Детекция → классификация** не стыкуются по типам: выход детекции — список, вход классификатора — одна картинка. Нужен явный шаг fan-out (v0.3, `Node` + `Sequential`).
- **Zero-shot задачи** все устроены одинаково: основной вход + набор меток. Когда появятся две такие задачи, это кандидат на общую базу (правило выноса общего, `architecture.md`, раздел 7).
- **Новые типы-значения** (`Audio`, `Video`, маска, вектор, таблица) делаются по образцу `Image`: обычный класс + `__get_pydantic_core_schema__` (`architecture.md`, раздел 1.2).