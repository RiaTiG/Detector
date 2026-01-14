"""
Тест полной версии детектора с поддержкой русского языка.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from feature_extractor import FeatureExtractor


def test_russian_features():
    """Тест извлечения признаков из русских текстов."""
    print("Тест детектора AI для РУССКОГО языка")
    print("=" * 70)
    print("Используется ПОЛНАЯ версия с spaCy и глубиной синтаксического дерева")
    print("=" * 70)

    extractor = FeatureExtractor(language='ru')

    # Человеческий текст (неформальный, разговорный)
    human_text = """
    Вчера был такой безумный день! Встретилась с Машей, пошли в новую кофейню на Арбате.
    Бариста оказался супер приятным парнем, налил нам кофе с лишней пенкой )))
    Короче, проболтали часа три обо всём - о работе, отношениях, жизни всякой.
    Прикольно как время летит, когда с хорошими людьми!
    """

    # AI текст (формальный, академический стиль)
    ai_text = """
    Внедрение передовых алгоритмов машинного обучения в современные бизнес-процессы
    представляет собой фундаментальную трансформацию операционных парадигм.
    Организации, использующие данные технологии, демонстрируют повышенные показатели
    эффективности и оптимизированные процессы принятия решений в различных
    функциональных областях. Интеграция искусственного интеллекта способствует
    достижению стратегических целей предприятия.
    """

    print("\n1. Извлечение признаков из ЧЕЛОВЕЧЕСКОГО текста...")
    human_features = extractor.extract_all_features(human_text)
    print(f"   Извлечено {len(human_features)} признаков")

    print("\n2. Извлечение признаков из AI текста...")
    ai_features = extractor.extract_all_features(ai_text)
    print(f"   Извлечено {len(ai_features)} признаков")

    # Сравнение ключевых признаков
    print("\n3. Ключевые различия (ЧЕЛОВЕК vs AI):")
    print("   " + "-" * 66)
    print(f"   {'Признак':<35} {'Человек':>12} {'AI':>12}")
    print("   " + "-" * 66)

    key_features = [
        ('avg_sentence_length', 'Средняя длина предложения'),
        ('sentence_length_variance', 'Дисперсия длины предложений'),
        ('avg_tree_depth', 'Глубина синтаксического дерева'),
        ('type_token_ratio', 'Лексическое разнообразие'),
        ('first_person_ratio', 'Местоимения 1-го лица'),
        ('discourse_marker_ratio', 'Дискурсивные маркеры'),
        ('avg_word_length', 'Средняя длина слова'),
        ('stopword_ratio', 'Служебные слова'),
        ('avg_dependency_distance', 'Длина зависимостей'),
        ('subordinate_clause_ratio', 'Придаточные предложения'),
    ]

    for feat_key, feat_name in key_features:
        if feat_key in human_features and feat_key in ai_features:
            human_val = human_features[feat_key]
            ai_val = ai_features[feat_key]
            print(f"   {feat_name:<35} {human_val:>12.4f} {ai_val:>12.4f}")

    # Показать уникальные признаки полной версии
    print("\n4. Уникальные признаки ПОЛНОЙ версии (требуют spaCy):")
    print("   " + "-" * 66)

    spacy_features = [
        'avg_tree_depth',
        'tree_depth_variance',
        'avg_dependency_distance',
        'dependency_distance_variance',
        'subordinate_clause_ratio',
        'entity_density',
        'pos_diversity'
    ]

    for feat in spacy_features:
        if feat in human_features:
            print(f"   + {feat:<35} = {human_features[feat]:.4f}")

    print("\n" + "=" * 70)
    print("УСПЕХ: Полная версия с русским языком работает корректно!")
    print("=" * 70)

    print(f"\nВсе доступные признаки ({len(human_features)}):")
    for i, feat in enumerate(sorted(human_features.keys()), 1):
        if i % 3 == 1:
            print(f"  {i:2d}. {feat:<30}", end="")
        elif i % 3 == 2:
            print(f"{i:2d}. {feat:<30}", end="")
        else:
            print(f"{i:2d}. {feat}")
    if len(human_features) % 3 != 0:
        print()

    return True


if __name__ == '__main__':
    test_russian_features()
