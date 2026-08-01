# SinglePipe

> Типизированный интерфейс для инференса ML-моделей: один API для HuggingFace, PyTorch и (в перспективе) других бэкендов.

⚠️ **Статус:** ранняя разработка. API нестабилен, поддерживается одна задача — детекция объектов через HuggingFace.

## Зачем

Работая с ML-моделями, легко утонуть в разнородности форматов: HuggingFace возвращает `list[dict]`, кастомный `nn.Module` — тензоры, remote-API — JSON со своей схемой. SinglePipe даёт **одну точку входа** и **типизированный выход** независимо от того, что под капотом.

## Возможности

- **Типизированные схемы вход/выход** на pydantic — валидация «из коробки», автодополнение в IDE, готовая JSON-сериализация.
- **Гибкий вход** — картинка может прийти как `PIL.Image`, `numpy.ndarray`, `torch.Tensor`, путь к файлу, URL или байты; SinglePipe сам нормализует к каноническому формату.
- **Единый интерфейс поверх разных бэкендов** — сейчас HuggingFace, дальше PyTorch, ONNX, remote-API.
- **Композиция пайплайнов** *(в разработке)* — объединение нескольких моделей в один типизированный поток (детектор → OCR → классификатор) с автоматической стыковкой по типам.
- **Асинхронность и параллельность независимых узлов** *(в дорожной карте)*.

## Установка

Пока только из исходников:

```bash
git clone https://github.com/<...>/SinglePipe
cd SinglePipe
uv sync
```

Требуется Python 3.10+.

## Быстрый старт

```python
from singlepipe.adapters.hf_adapter import HFAdapter
from singlepipe.model_types import TaskType

adapter = HFAdapter(
    model_name="facebook/detr-resnet-50",
    task=TaskType.OBJECT_DETECTION,
)

result = adapter("http://images.cocodataset.org/val2017/000000039769.jpg")

print(f"Found {len(result)} objects")
for det in result:
    print(f"  {det.label}: score={det.score:.2f}, box={det.box.as_tuple()}")
```

Что можно передать в адаптер: `PIL.Image`, `numpy.ndarray` (HWC), `torch.Tensor` (CHW), путь к файлу (`str` или `Path`), `http(s)://` URL, сырые `bytes`.

## Работа с результатом

Результат — типизированный pydantic-объект с богатым API:

```python
# Фильтрация
confident = result.filter_by_score(0.7)
cats_only = result.filter_by_label("cat")
some = result.filter_by_label(["cat", "dog", "person"])

# Итерация
for det in result:
    print(det.label, det.score)
print(f"Total: {len(result)}")

# Геометрия
box = result.predictions[0].box
box.width, box.height, box.area, box.center
box.clip(image_width=800, image_height=600)
box.iou(other_box)

# Сериализация
result.model_dump()
result.model_dump_json()
```

## Поддерживаемые задачи

| Задача | Бэкенд | Статус |
|---|---|---|
| Object detection | HuggingFace | ✅ |
| Image classification | HuggingFace | 🚧 в планах |
| Text classification | HuggingFace | 🚧 в планах |
| Text generation | HuggingFace | 🚧 в планах |

## Дорожная карта

- [x] HF-адаптер для одиночной модели
- [x] Типизированные схемы для object detection
- [ ] Вторая задача (image classification) — проверка переносимости паттерна
- [ ] Интерфейс `Node` для унифицированных узлов
- [ ] Линейная композиция пайплайнов
- [ ] PyTorch-адаптер (для кастомных `nn.Module`)
- [ ] Полноценный DAG с ветвлениями
- [ ] Async и параллельность независимых узлов
- [ ] ONNX Runtime бэкенд
- [ ] Remote-API бэкенд (OpenAI-совместимый)

## Разработка

```bash
uv sync --group dev
uv run pytest
```

## Лицензия

MIT (черновик — уточнить перед первым релизом).