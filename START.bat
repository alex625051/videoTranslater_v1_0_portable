@echo off
chcp 65001
setlocal

rem Путь к папке с вашим проектом
set PROJECT_DIR=C:\videoTranslater_v1_0_portable

rem Путь к скрипту Python
set PYTHON_SCRIPT=%PROJECT_DIR%\перевод_видео_с_субтитрами(alex)_ver0_1_1.py

rem Путь к виртуальному окружению
set VENV_DIR=%PROJECT_DIR%\venv

rem Проверяем наличие виртуального окружения
if not exist "%VENV_DIR%" (
    echo Виртуальное окружение не найдено
    goto End
)

rem Активация виртуального окружения
call "%VENV_DIR%\Scripts\activate.bat"

rem Запуск Python-скрипта
"%VENV_DIR%\Scripts\python.exe" "%PYTHON_SCRIPT%"

:End
endlocal
pause
