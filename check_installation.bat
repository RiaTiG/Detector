@echo off
echo ========================================
echo Проверка установки детектора AI
echo ========================================
echo.

echo [1/5] Проверка виртуального окружения...
if exist venv\Scripts\python.exe (
    echo [OK] Виртуальное окружение найдено
) else (
    echo [ОШИБКА] Виртуальное окружение не найдено!
    echo Запустите: py -3.12 -m venv venv
    pause
    exit /b 1
)

echo.
echo [2/5] Проверка версии Python...
venv\Scripts\python.exe --version
if errorlevel 1 (
    echo [ОШИБКА] Не удалось запустить Python!
    pause
    exit /b 1
)

echo.
echo [3/5] Проверка spaCy и русской модели...
venv\Scripts\python.exe -c "import spacy; nlp = spacy.load('ru_core_news_sm'); print('[OK] spaCy OK')"
if errorlevel 1 (
    echo [ОШИБКА] spaCy или русская модель не установлены!
    echo Запустите: venv\Scripts\python.exe -m spacy download ru_core_news_sm
    pause
    exit /b 1
)

echo.
echo [4/5] Проверка основных библиотек...
venv\Scripts\python.exe -c "import xgboost, torch, sklearn, nltk; print('[OK] All libraries OK')"
if errorlevel 1 (
    echo [ОШИБКА] Не все библиотеки установлены!
    pause
    exit /b 1
)

echo.
echo [5/5] Запуск теста на русском языке...
echo.
venv\Scripts\python.exe test_russian.py
if errorlevel 1 (
    echo.
    echo [ОШИБКА] Тест не прошел!
    pause
    exit /b 1
)

echo.
echo ========================================
echo [SUCCESS] Все проверки пройдены!
echo ========================================
echo.
echo Система готова к использованию:
echo   - Python 3.12 с виртуальным окружением
echo   - spaCy с русской моделью
echo   - Все 25 признаков работают
echo   - Глубина синтаксического дерева: ОК
echo.
echo Для начала работы:
echo   1. activate_venv.bat
echo   2. set USE_FULL_FEATURE_EXTRACTOR=1
echo   3. python example_russian.py
echo.
pause
