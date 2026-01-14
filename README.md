# AI Text Detector

Система обнаружения AI-генерированных текстов на основе стилометрических, синтаксических и семантических признаков, которые сложно имитировать AI.

## Описание

Этот проект реализует детектор текстов, написанных AI, используя признаки, которые плохо поддаются имитации:

### 1. Стилометрические признаки
- Средняя длина предложения и её дисперсия
- Разнообразие лексики (Type-Token Ratio)
- Hapax legomena (слова, встречающиеся один раз)
- Частота использования служебных слов (stopwords)
- Плотность пунктуации
- Статистика длины слов

### 2. Синтаксические признаки
- Глубина синтаксического дерева
- Паттерны расстановки запятых
- Длина зависимостей в дереве разбора
- Сложность придаточных предложений
- Разнообразие частей речи

### 3. Семантические признаки
- Плотность именованных сущностей
- Паттерны использования местоимений
- Частота отрицаний
- Использование модальных глаголов
- Соотношение прилагательных и существительных

## Установка

```bash
# Клонировать репозиторий
git clone <repo-url>
cd Detector

# Установить зависимости
pip install -r requirements.txt

# ВАЖНО: Для Python 3.14
# Используется облегченная версия без spaCy (feature_extractor_lite.py)
# из-за несовместимости spaCy с Python 3.14

# Для Python 3.11-3.12 (опционально, для полной функциональности):
python -m spacy download en_core_web_sm  # для английского
python -m spacy download ru_core_news_sm  # для русского
```

## Структура проекта

```
Detector/
├── src/
│   ├── feature_extractor.py  # Извлечение признаков
│   ├── dataset_loader.py      # Загрузка и подготовка данных
│   ├── classifiers.py         # Классификаторы (XGBoost, нейросеть)
│   ├── visualizer.py          # Визуализация результатов
│   └── detector.py            # Главный интерфейс детектора
├── data/                      # Данные для обучения
├── models/                    # Сохраненные модели
├── results/                   # Результаты и визуализации
├── train.py                   # Скрипт обучения
├── detect.py                  # Скрипт детекции
├── example.py                 # Пример использования
└── requirements.txt           # Зависимости
```

## Использование

### 1. Пример работы

Запустите пример для быстрой демонстрации:

```bash
python example.py
```

Это создаст пример датасета, обучит модель и покажет результаты.

### 2. Обучение на своих данных

#### Вариант A: Из директорий с текстовыми файлами

```bash
# Создайте структуру:
# data/
#   human/
#     text1.txt
#     text2.txt
#   ai/
#     text1.txt
#     text2.txt

python train.py --data-dir data --model both
```

#### Вариант B: Из CSV файла

```bash
# CSV должен содержать столбцы 'text' и 'label' (0=человек, 1=AI)
python train.py --csv-path data.csv --model xgboost
```

#### Вариант C: Создать пример датасета

```bash
python train.py --create-sample --model both
```

### 3. Детекция текстов

```bash
# Анализ текста из командной строки
python detect.py --text "Your text here to analyze..."

# Анализ текста из файла
python detect.py --file document.txt

# Интерактивный режим
python detect.py

# С показом всех признаков
python detect.py --text "Text..." --show-features
```

### 4. Программное использование

```python
from src.detector import AITextDetector

# Загрузить обученную модель
detector = AITextDetector(
    model_type='xgboost',
    model_path='models/xgboost_model.pkl',
    language='en'
)

# Анализировать текст
text = "Your text to analyze..."
result = detector.detect(text)

print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"AI probability: {result['ai_probability']:.2%}")

# Получить объяснение
explanation = detector.explain_prediction(text, top_n=5)
for feat in explanation['top_contributing_features']:
    print(f"{feat['feature']}: {feat['value']:.4f}")
```

## Модели

### XGBoost
- Быстрое обучение и предсказание
- Интерпретируемые результаты (feature importance)
- Хорошо работает на небольших датасетах
- Рекомендуется для большинства задач

### Нейронная сеть
- Может выявлять более сложные паттерны
- Требует больше данных для обучения
- Более гибкая архитектура

## Параметры обучения

```bash
python train.py --help

Options:
  --data-dir PATH        Директория с human/ и ai/ поддиректориями
  --csv-path PATH        Путь к CSV файлу
  --model TYPE           Тип модели: xgboost, neural, both (default: both)
  --output-dir PATH      Директория для сохранения моделей (default: models)
  --no-viz               Отключить визуализации
  --create-sample        Создать пример датасета
```

## Результаты

После обучения в директории `results/` создаются:

1. **Feature Importance** - график важности признаков
2. **Confusion Matrix** - матрица ошибок
3. **ROC Curve** - ROC-кривая с AUC
4. **Feature Distributions** - распределения признаков для человеческих и AI текстов
5. **Model Comparison** - сравнение моделей (если обучены обе)
6. **Text Reports** - текстовые отчеты с метриками

## Метрики качества

- **Accuracy** - общая точность
- **Precision** - точность (доля правильных среди предсказанных как AI)
- **Recall** - полнота (доля найденных AI текстов)
- **F1-Score** - гармоническое среднее precision и recall
- **AUC** - площадь под ROC-кривой

## Признаки, устойчивые к имитации

Основные признаки, которые сложно подделать AI:

1. **Вариативность длины предложений** - люди пишут более хаотично
2. **Hapax legomena ratio** - уникальные слова, используемые один раз
3. **Неравномерность использования запятых** - AI более предсказуем
4. **Местоимения первого лица** - люди чаще используют "я", "мы"
5. **Дисперсия глубины синтаксического дерева** - люди варьируют сложность

## Расширение функциональности

### Добавление новых признаков

Отредактируйте [src/feature_extractor.py](src/feature_extractor.py):

```python
def extract_custom_features(self, text: str) -> Dict[str, float]:
    features = {}
    # Ваши признаки
    features['custom_feature'] = calculate_something(text)
    return features
```

### Настройка классификаторов

Параметры в [src/classifiers.py](src/classifiers.py):

```python
# XGBoost
classifier = XGBoostClassifier(
    n_estimators=200,    # количество деревьев
    max_depth=6,         # глубина деревьев
    learning_rate=0.1    # скорость обучения
)

# Neural Network
trainer = NeuralNetTrainer(
    hidden_sizes=[128, 64, 32],  # размеры скрытых слоёв
    dropout=0.3,                  # dropout для регуляризации
    learning_rate=0.001           # скорость обучения
)
```

## Требования

- Python 3.8+
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- torch >= 2.0.0
- spacy >= 3.7.0
- nltk >= 3.8.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

## Лицензия

MIT

## Автор

Разработано для исследования методов детекции AI-генерированных текстов на основе стилометрических и синтаксических признаков.

## Цитирование

Если вы используете этот код в исследованиях, пожалуйста, укажите ссылку на репозиторий.

## Заметки

- Для работы с русским языком используйте параметр `--language ru`
- Рекомендуется минимум 100 примеров каждого класса для хорошего качества
- Модели работают лучше на текстах длиной от 50 слов
- Для оптимальных результатов используйте тексты схожей тематики в обучающей выборке
