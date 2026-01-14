"""
Полный пример использования детектора AI для русского языка.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Force using full version with spaCy
import os
os.environ['USE_FULL_FEATURE_EXTRACTOR'] = '1'

from dataset_loader import DatasetLoader
from classifiers import XGBoostClassifier
from visualizer import Visualizer


def create_russian_sample_data():
    """Создать образцы данных на русском языке."""

    human_texts = [
        "Ну вчера вообще жесть была! Пошли с друзьями в новый бар, так там такая атмосфера классная. Музыка норм, народу немного, в общем отдохнули на ура)",
        "Короче, смотрю я сериал этот новый. Первые серии вроде ничего, потом как-то скучновато стало. Не знаю, досмотрю или нет...",
        "Вчера на работе такое произошло! Начальник вызвал всех на совещание внезапное. Думали, что-то серьезное случилось, а он просто про премии объявил)) Прикол",
        "Ребят, кто-нибудь знает хороший ресторан недалеко от метро? Хочу девушку пригласить, но не знаю куда. Посоветуйте что-нибудь романтичное!",
        "Сегодня попробовала новый рецепт торта - получилось так себе, если честно. Крем вышел слишком сладким, а бисквит суховат. В следующий раз попробую по-другому.",
        "Не могу найти нормальные наушники! Уже третьи за месяц покупаю, все ломаются. Может посоветуете какие надежные?",
        "У меня кот такой смешной! Сегодня в коробку залез, которая явно мала для него, и сидит довольный))) Животные - это нечто",
        "Кстати, насчет той книги что ты советовал - начал читать, пока нравится. Стиль интересный, сюжет закручен неплохо. Спасибо за рекомендацию!",
    ]

    ai_texts = [
        "Внедрение современных информационных технологий в образовательный процесс представляет собой важнейший аспект модернизации системы образования. Цифровизация способствует повышению качества обучения.",
        "Анализ макроэкономических показателей демонстрирует устойчивую тенденцию к росту валового внутреннего продукта. Экспертные оценки прогнозируют дальнейшее развитие экономической ситуации в позитивном направлении.",
        "Исследование климатических изменений требует комплексного междисциплинарного подхода. Научное сообщество уделяет значительное внимание разработке инновационных методов мониторинга экологических параметров.",
        "Развитие искусственного интеллекта открывает новые перспективы для автоматизации бизнес-процессов. Интеграция машинного обучения способствует оптимизации производственных циклов и повышению эффективности организационных структур.",
        "Современная архитектура характеризуется применением экологически устойчивых материалов и энергоэффективных технологий. Градостроительная политика ориентирована на создание комфортной городской среды.",
        "Трансформация потребительского поведения в условиях цифровой экономики обуславливает необходимость адаптации маркетинговых стратегий. Персонализация предложений становится ключевым фактором конкурентоспособности.",
        "Медицинские исследования в области генетики открывают широкие возможности для развития персонализированной терапии. Молекулярная диагностика обеспечивает раннее выявление патологических состояний.",
        "Глобализация образовательного пространства способствует интернационализации академических программ. Межкультурное взаимодействие обогащает образовательный опыт и расширяет профессиональные горизонты.",
    ]

    # Создать директории
    output_path = Path('data/russian_sample')
    human_dir = output_path / 'human'
    ai_dir = output_path / 'ai'

    human_dir.mkdir(parents=True, exist_ok=True)
    ai_dir.mkdir(parents=True, exist_ok=True)

    # Сохранить человеческие тексты
    for i, text in enumerate(human_texts):
        with open(human_dir / f'human_{i:03d}.txt', 'w', encoding='utf-8') as f:
            f.write(text)

    # Сохранить AI тексты
    for i, text in enumerate(ai_texts):
        with open(ai_dir / f'ai_{i:03d}.txt', 'w', encoding='utf-8') as f:
            f.write(text)

    print(f"Создано примеров: {len(human_texts)} человеческих, {len(ai_texts)} AI")
    return str(output_path)


def main():
    print("Детектор AI-текстов - Пример для РУССКОГО языка")
    print("=" * 70)
    print("Используется ПОЛНАЯ версия с глубиной синтаксического дерева")
    print("=" * 70)

    # Шаг 1: Создать образцы данных
    print("\n1. Создание образцов русских текстов...")
    data_path = create_russian_sample_data()

    # Шаг 2: Загрузить данные
    print("\n2. Загрузка и подготовка данных...")
    loader = DatasetLoader(language='ru')
    df = loader.load_from_text_files(
        f'{data_path}/human',
        f'{data_path}/ai'
    )
    print(f"   Загружено {len(df)} примеров")

    data_dict = loader.prepare_training_data(df, test_size=0.25, random_state=42)
    print(f"   Обучающая выборка: {len(data_dict['X_train'])} примеров")
    print(f"   Тестовая выборка: {len(data_dict['X_test'])} примеров")
    print(f"   Признаков: {data_dict['n_features']}")

    # Шаг 3: Обучить XGBoost
    print("\n3. Обучение XGBoost классификатора...")
    classifier = XGBoostClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
    classifier.train(
        data_dict['X_train'],
        data_dict['y_train'],
        data_dict['X_test'],
        data_dict['y_test']
    )

    # Шаг 4: Оценить модель
    print("\n4. Оценка модели...")
    metrics = classifier.evaluate(data_dict['X_test'], data_dict['y_test'])

    print("\n   Метрики качества:")
    print("   " + "-" * 50)
    for metric, value in metrics.items():
        print(f"   {metric.capitalize():15s}: {value:.4f}")

    # Шаг 5: Важность признаков
    print("\n5. Анализ важности признаков...")
    feature_importance = classifier.get_feature_importance(data_dict['feature_names'])
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]

    print("\n   Топ-10 самых важных признаков:")
    print("   " + "-" * 50)
    for idx, (feature, score) in enumerate(sorted_features, 1):
        print(f"   {idx:2d}. {feature:30s}: {score:.4f}")

    # Шаг 6: Тестирование на новых текстах
    print("\n6. Тестирование на новых примерах...")

    test_texts = [
        ("Слушай, ты не поверишь что случилось! Иду я значит по улице, а тут бац - "
         "встречаю свою бывшую одноклассницу. Не виделись лет 10 наверное. "
         "Проболтали минут 20, обменялись номерами. Мир тесен, да?)"),

        ("Разработка стратегии цифровой трансформации предприятия требует "
         "системного анализа существующих бизнес-процессов и технологической "
         "инфраструктуры. Оптимизация организационных структур способствует "
         "повышению операционной эффективности и конкурентоспособности компании.")
    ]

    from feature_extractor import FeatureExtractor
    import numpy as np

    extractor = FeatureExtractor(language='ru')

    for idx, text in enumerate(test_texts, 1):
        print(f"\n   Тест {idx}:")
        print(f"   Текст: {text[:70]}...")

        # Извлечь признаки
        features = extractor.extract_all_features(text)
        feature_vector = np.array([[features[name] for name in data_dict['feature_names']]])

        # Предсказание
        prediction = classifier.predict(feature_vector)[0]
        proba = classifier.predict_proba(feature_vector)[0]

        print(f"   Предсказание: {'AI-генерированный' if prediction else 'Человеческий'}")
        print(f"   Уверенность: {max(proba):.2%}")
        print(f"   Вероятности: Человек={proba[0]:.2%}, AI={proba[1]:.2%}")

    # Шаг 7: Визуализации
    print("\n7. Создание визуализаций...")
    viz = Visualizer(output_dir='results')

    # Важность признаков
    viz.plot_feature_importance(
        feature_importance,
        top_n=15,
        title='Важность признаков (русский язык)',
        filename='russian_feature_importance.png'
    )

    # Матрица ошибок
    y_pred = classifier.predict(data_dict['X_test'])
    viz.plot_confusion_matrix(
        data_dict['y_test'],
        y_pred,
        labels=['Человек', 'AI'],
        filename='russian_confusion_matrix.png'
    )

    # ROC кривая
    y_proba = classifier.predict_proba(data_dict['X_test'])[:, 1]
    viz.plot_roc_curve(
        data_dict['y_test'],
        y_proba,
        filename='russian_roc_curve.png'
    )

    # Распределения признаков
    viz.plot_feature_distributions(
        data_dict['X_train'],
        data_dict['y_train'],
        data_dict['feature_names'],
        filename='russian_feature_distributions.png'
    )

    # Отчет
    viz.create_report(
        'XGBoost (Русский язык)',
        metrics,
        feature_importance,
        filename='russian_report.txt'
    )

    # Сохранить модель
    model_path = Path('models')
    model_path.mkdir(exist_ok=True)
    classifier.save('models/russian_xgboost_model.pkl')
    print(f"\n   Модель сохранена: models/russian_xgboost_model.pkl")

    print("\n" + "=" * 70)
    print("Пример завершен! Проверьте директорию 'results' для визуализаций.")
    print("=" * 70)


if __name__ == '__main__':
    main()
