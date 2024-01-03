chcp 65001

@echo off
setlocal

rem Проверка существования виртуального окружения
if exist "venv" (
    echo Уже установлено
    goto End
)




rem Создание виртуального окружения
echo Создание виртуального окружения...
"C:\Users\%username%\AppData\Local\Programs\Python\Python311\python.exe" -m venv venv

rem Проверка успешного создания виртуального окружения
if errorlevel 1 (
    echo Ошибка создания виртуального окружения
    goto End
)

rem Активация виртуального окружения
call venv\Scripts\activate.bat
pip install --upgrade pip setuptools


rem Установка необходимых пакетов
echo Установка пакетов...
pip install git+https://github.com/alex625051/whisper.git
if errorlevel 1 goto Error
pip install silero
if errorlevel 1 goto Error
pip install pydub
if errorlevel 1 goto Error
pip install torch
if errorlevel 1 goto Error
pip install librosa
if errorlevel 1 goto Error
pip install --upgrade --no-deps --force-reinstall git+https://github.com/alex625051/whisper.git
if errorlevel 1 goto Error

echo Окружение установлено
goto End

:Error
echo Ошибка установки окружения

:End
endlocal
pause