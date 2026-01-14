@echo off
REM Активация виртуального окружения Python 3.12 с полной версией spaCy

echo ========================================
echo Активация виртуального окружения
echo Python 3.12 с полной поддержкой spaCy
echo ========================================
echo.

call venv\Scripts\activate.bat

echo.
echo Виртуальное окружение активировано!
echo.
echo Доступные команды:
echo   python test_russian.py          - Тест русского языка
echo   python example_russian.py       - Полный пример на русском
echo   python train.py --help          - Справка по обучению
echo   python detect.py --help         - Справка по детекции
echo.
echo Для деактивации введите: deactivate
echo ========================================
