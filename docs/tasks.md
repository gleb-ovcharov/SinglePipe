# Задачи SinglePipe

Список всех задач, которые SinglePipe собирается поддерживать, и их состояние. Это память проекта о задачах: заглушек и списка имён «на будущее» в коде нет (см. `architecture.md`, разделы 7 и 12.6), поэтому всё, что нужно не забыть, пишется сюда.

Источники истины:

- **Какие задачи реализованы** — объекты `Task` в `singlepipe/tasks.py` (`image_classification`, `object_detection`). Задача реализована, если для неё есть объект `Task`.
- **Какие задачи в планах** — эта таблица.
- **Имя задачи** совпадает со строкой HF `pipeline`, где такой pipeline есть: так `HFAdapter` берёт его по умолчанию. Для задач без HF pipeline (OCR, табличка) имя выбирается для читаемости. Колонка «Объект» — имя будущего объекта в `sp.tasks`.
- Enum `TaskType` удалён (`architecture.md`, 12.6). Код старых заглушек остался в теге `v0.0.1`.

Статусы:

- **Есть** — объект `Task`, схемы на контракте, юнит- и интеграционные тесты.
- **В работе** — объект `Task` есть, но не всё доведено (например, нет интеграционного теста).
- **В плане** — привязано к шагу плана или пункту дорожной карты.
- **Идея** — планов нет.

Когда задача меняет статус, правится эта таблица в том же коммите.

---

## Картинки

| Имя | Объект | Статус | Заметки |
|---|---|---|---|
| `image-classification` | `image_classification` | Есть | `ImageTaskInput` + `PredictionList[Prediction]`. Интеграционные тесты: крошечная ViT без сети и `google/vit-base-patch16-224` (`slow`), плюс своя задача поверх HF через `hf_task` и батч из 4 картинок через настоящий pipeline. `label_id` пока не заполняется. |
| `object-detection` | `object_detection` | В работе | На основах: `ImageTaskInput` + `output_context` с размерами, `PredictionList[Detection]`. Коробки за краем с любой стороны не обрезаются и не отвергаются (отрицательные координаты разрешены). Юнит-тесты есть; **нет интеграционного теста** на крошечной DETR. |
| `image-segmentation` | `image_segmentation` | Идея | Понадобится тип-значение для маски. |
| `depth-estimation` | `depth_estimation` | Идея | Выход — карта глубины, тоже значение, а не запись. |
| `image-feature-extraction` | `image_embedding` | Идея | Выход — вектор; общий тип с текстовыми эмбеддингами. Имя объекта человеческое, строка HF — по умолчанию для `hf_task`. |
| `zero-shot-image-classification` | `zero_shot_image_classification` | Идея | Вход — картинка + набор меток, поэтому `ImageTaskInput` как есть не подойдёт. Выход совпадает с классификацией. |
| `zero-shot-object-detection` | `zero_shot_object_detection` | Идея | Вход — картинка + метки, выход как у детекции. |
| `mask-generation` | `mask_generation` | Идея | Маски, как у сегментации. |
| `keypoint-matching` | `keypoint_matching` | Идея | Вход — две картинки. |

## Текст

| Имя | Объект | Статус | Заметки |
|---|---|---|---|
| `text-classification` | `text_classification` | Идея | Выход — `PredictionList[Prediction]`, как у картиночной классификации. |
| `token-classification` | `token_classification` | Идея | NER; у предсказания появятся позиции в тексте. |
| `text-generation` | `text_generation` | Идея | |
| `text2text-generation` | `text2text_generation` | Идея | При реализации сверить, что pipeline есть в актуальной версии `transformers`. |
| `fill-mask` | `fill_mask` | Идея | Выход — список вариантов со скором. |
| `summarization` | `summarization` | Идея | Сверить с актуальной версией `transformers`. |
| `translation` | `translation` | Идея | Сверить с актуальной версией `transformers`; языковая пара в `hf_task` (`translation_en_to_fr`). |
| `question-answering` | `question_answering` | Идея | Вход — вопрос + контекст. |
| `feature-extraction` | `text_embedding` | Идея | Общий тип выхода с картиночными эмбеддингами. |
| `zero-shot-classification` | `zero_shot_text_classification` | Идея | Вход — текст + метки. |

## Аудио

| Имя | Объект | Статус | Заметки |
|---|---|---|---|
| `audio-classification` | `audio_classification` | Идея | Нужен тип-значение `Audio`, по образцу `Image`. |
| `automatic-speech-recognition` | `speech_to_text` | Идея | |
| `text-to-audio` | `text_to_audio` | Идея | Выход — `Audio`. |
| `zero-shot-audio-classification` | `zero_shot_audio_classification` | Идея | Вход — аудио + метки. |

## Видео

| Имя | Объект | Статус | Заметки |
|---|---|---|---|
| `video-classification` | `video_classification` | Идея | Нужен тип-значение `Video`. |

## Мультимодальные

| Имя | Объект | Статус | Заметки |
|---|---|---|---|
| `image-text-to-text` | `image_text_to_text` | Идея | Вход — `Image` + текст. |
| `document-question-answering` | `document_question_answering` | Идея | Вход — `Image` документа + вопрос. |
| `table-question-answering` | `table_question_answering` | Идея | Вход — таблица + вопрос; тип таблицы общий с табличными задачами. |

## Табличные

| Имя | Объект | Статус | Заметки |
|---|---|---|---|
| `tabular-classification` | `tabular_classification` | В плане (после 0.2.0) | Бэкенды sklearn / xgboost / lightgbm / catboost, не HF. Строка признаков — pydantic-схема. Батчи решены на уровне адаптера (`architecture.md`, 12.4). Демо «страховой случай». |
| `tabular-regression` | `tabular_regression` | В плане (после 0.2.0) | То же. Выход — число, не `Prediction`. |

## OCR

| Имя | Объект | Статус | Заметки |
|---|---|---|---|
| `ocr` | `ocr` | Идея | HF pipeline нет: свой адаптер или адаптер над функцией (`architecture.md`, 12.7). Нужен в сценариях «номера машин» и «документ». |

---

## Сквозные заметки

- **Детекция → классификация** не стыкуются по типам: выход детекции — список, вход классификатора — одна картинка. Решено явным fan-out: `sp.ops.crops(image, dets)` + `classifier.map(crops)` (`architecture.md`, 12.8).
- **Задачи вне HF** обслуживают другие адаптеры — подклассы `Adapter` с одним методом `run_batch`; универсальный выход — адаптер над функцией (`architecture.md`, 12.7). Своя задача поверх HF-модели — `Task` + `hf_task=...`.
- **Батчинг** у всех задач бесплатный: его делает база `Adapter`, схемы остаются поштучными (`architecture.md`, 12.4). Чтобы он сработал на списке, вызывать через `map`, а не циклом `await`.
- **Zero-shot задачи** все устроены одинаково: основной вход + набор меток. Когда появятся две такие задачи, это кандидат на общую базу (правило выноса общего, `architecture.md`, раздел 7).
- **Новые типы-значения** (`Audio`, `Video`, маска, вектор, таблица) делаются по образцу `Image`: обычный класс + `__get_pydantic_core_schema__` (`architecture.md`, раздел 1.2).