# -*- coding: utf-8 -*-
"""перевод_видео_с_субтитрами(alex)_ver0.1.ipynb
Original file is located at
https://colab.research.google.com/drive/16JEgsZ4rDmGRwZThwD048oUDL1AYA4-n
**Получение текста исходного видео**
"""

"""https://github.com/snakers4/silero-models/blob/master/models.yml
https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples.ipynb#scrollTo=nYG1dBgBDN5S
https://tts.readthedocs.io/en/latest/inference.html
https://github.com/snakers4/silero-models/#standalone-use
"""

import os
import subprocess
import sys
import threading
import tkinter as tk
from sys import exit
from tkinter import Tk
from tkinter import filedialog
from tkinter import ttk


def update_progress(value):
    progress_var.set(value)
    root.update_idletasks()


def update_message(message):
    message_label.config(text=message)
    root.update_idletasks()


def open_output_folder():
    # Открытие папки с результатами
    os.startfile(project_output_persistance)


def resource_path(relative_path):
    """
    Получает абсолютный путь к ресурсу, который работает как для разработки, так и для однофайлового исполняемого файла.

    :param relative_path: Относительный путь к файлу или каталогу.
    :return: Абсолютный путь к файлу или каталогу.
    """
    if relative_path == '.' or relative_path == './' or relative_path == '.\\':
        return os.path.abspath(".")
    if relative_path.startswith('.'):
        raise ValueError("Относительный путь НЕ должен начинаться с точки. - Сразу с папки")
    try:
        # PyInstaller создает временную папку и сохраняет путь в sys._MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# Установка языков для транскрибации и перевода
#####################################
source_language = "Russian"
target_language = "English"
tts_model_name = 'v3_en.pt'
this_device_processor_tourch = 'cpu'  # cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mtia
speaker = 'en_11'
tts_url = 'https://models.silero.ai/models/tts/en/v3_en.pt'
trans_variant = '2'  # '1':траскрибация на языке;  '2': транскрибация+перевод
DEBUG = True
current_directory = os.getcwd()
project_persistance_folder = current_directory
#####################################
project_output_persistance = os.path.join(project_persistance_folder, 'output')
project_persistance_folder_tts = resource_path('tts_models')
project_persistance_tts_modelfile = os.path.join(project_persistance_folder_tts, tts_model_name)
os.makedirs(project_output_persistance, exist_ok=True)
os.makedirs(project_persistance_folder_tts, exist_ok=True)
temp_waves_folder = os.path.join(project_persistance_folder, 'waves')
os.makedirs(temp_waves_folder, exist_ok=True)

running = True  # Флаг для управления выполнением потоков

import os
import glob

# Список расширений файлов для удаления
extensions = ["*.txt", "*.vtt", "*.srt", "*.tsv", "*.json"]

# Удаление файлов по заданным расширениям
for ext in extensions:
    for file in glob.glob(os.path.join(project_output_persistance, ext)):
        os.remove(file)
        print(f"Удален файл: {file}")

# Удаление файла source_audio_track.wav
audio_file = os.path.join(project_persistance_folder, "source_audio_track.wav")
if os.path.exists(audio_file):
    os.remove(audio_file)
    print(f"Удален файл: {audio_file}")

# Удаление всех файлов в директории output, но не папок
for file in os.listdir(project_output_persistance):
    file_path = os.path.join(project_output_persistance, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

# Удаление всех файлов в директории waves, но не папок
for file in os.listdir(temp_waves_folder):
    file_path = os.path.join(temp_waves_folder, file)
    if os.path.isfile(file_path):
        os.remove(file_path)
if os.path.exists(audio_file):
    os.remove(audio_file)

# Путь к папке для хранения модели large_v3 на Google Диске
wisper_model_path = resource_path("whisper3/large-v3")
os.makedirs(wisper_model_path, exist_ok=True)


# Блок для загрузки файла


def main():
    import os
    # Запрос файла через диалоговое окно
    input_file = filedialog.askopenfilename(title='Выбрать видео для обработки на рускком языке')

    # Проверка, был ли выбран файл
    if not input_file:
        print("Файл не был выбран.")
        exit(1)

    update_message(f"Выбранный файл: {input_file}")
    # Вывод пути к выбранному файлу
    print(f"Выбранный файл: {input_file}")

    input_file_basename = os.path.basename(input_file)
    input_file_videoname_without_ext = os.path.splitext(input_file_basename)[0]
    input_file_videoname_only_ext = os.path.splitext(input_file_basename)[1]

    # Commented out IPython magic to ensure Python compatibility.
    #########################################
    # Загрузка нейросети и процессы обработки 1
    #########################################

    """**Загрузка модели**"""

    model_file = os.path.join(wisper_model_path, "large-v3.pt")
    #########################################
    # Загрузка нейросети и процессы обработки 2
    #########################################
    import whisper

    update_message(f"Загрузка модели")
    # Проверка наличия модели large_v3
    if not os.path.isfile(model_file):
        # Загрузка модели с OpenAI, если она отсутствует
        model = whisper.load_model("large-v3", download_root=wisper_model_path)
    else:
        # Загрузка модели из кеша
        model = whisper.load_model(model_file)

    update_message(f"Загружена модель: {model_file}")

    # Команда ffmpeg для извлечения аудиодорожки
    command = [
        'ffmpeg',
        "-y",  # Перезапись без подтверждения
        '-i', f'{input_file}',  # Исходный файл
        '-vn',  # Отключение обработки видео
        '-acodec', 'pcm_s16le',  # Кодек аудио
        '-ar', '44100',  # Частота дискретизации
        '-ac', '2',  # Количество аудиоканалов
        audio_file  # Выходной файл
    ]

    update_message(f"Обработка файла")
    # Запуск команды
    subprocess.run(command, check=True)

    # Это будет захватывать весь вывод ячейки, если debug = False
    # Транскрибация и перевод текста
    update_message(f"Транскрибация и перевод текста")
    initial_prompt = "A lecture on medicine. Medical terms are used. Lecture for medical university students. Лекция по медицине. Используются медицинские термины. Лекция для студентов медицинского университета."
    # Возобновление перенаправления
    console_redirector.start_redirect()
    result_AUTOTRANSLATED = model.transcribe("source_audio_track.wav", verbose=True, initial_prompt=initial_prompt,
                                             task='translate', language='ru'
                                             )
    # Остановка перенаправления
    console_redirector.stop_redirect()
    update_message(f"Перевод закончен")

    # Сохранение результатов в формате txt и субтитров с таймкодами
    from whisper.utils import get_writer
    output_format = "all"
    output_writer = get_writer(output_format, project_output_persistance)
    output_writer(result_AUTOTRANSLATED, input_file_videoname_without_ext)

    srt_name = os.path.join(project_output_persistance, input_file_videoname_without_ext + '.srt')
    txt_name = os.path.join(project_output_persistance, input_file_videoname_without_ext + '.txt')
    vtt_name = os.path.join(project_output_persistance, input_file_videoname_without_ext + '.vtt')
    tsv_name = os.path.join(project_output_persistance, input_file_videoname_without_ext + '.tsv')
    json_name = os.path.join(project_output_persistance, input_file_videoname_without_ext + '.json')

    """**Синтез речи**"""

    update_message(f"Синтез речи")
    import os
    import torch
    from pydub import AudioSegment
    import re

    device = torch.device(this_device_processor_tourch)
    torch.set_num_threads(4)
    # local_tts_file = project_persistance_tts_modelfile
    # local_tts_file = tts_model_name
    local_tts_file = os.path.join(project_persistance_folder, tts_model_name)

    if not os.path.isfile(local_tts_file):
        torch.hub.download_url_to_file(tts_url,
                                       local_tts_file)

    model = torch.package.PackageImporter(local_tts_file).load_pickle("tts_models", "model")
    model.to(device)
    sample_rate = 48000

    # Функция для конвертации временного кода в миллисекунды
    def timecode_to_ms(tc):
        h, m, s, ms = map(int, re.split('[:,]', tc))
        return (h * 3600000) + (m * 60000) + (s * 1000) + ms

    # Функция для группировки текста по предложениям
    def group_sentences(subtitles, max_length=1000):
        sentences = []
        current_sentence = ""
        current_duration = 0
        last_end_time = 0

        for i, segment in enumerate(subtitles):
            lines = segment.split('\n')
            if len(lines) >= 3:
                start_tc, end_tc = lines[1].split(' --> ')
                start_time = timecode_to_ms(start_tc)
                end_time = timecode_to_ms(end_tc)
                text = ' '.join(lines[2:])
                segment_duration = end_time - start_time

                # Добавление паузы, если есть промежуток между сегментами
                if last_end_time and start_time - last_end_time > 0:
                    pause_duration = start_time - last_end_time
                    sentences.append((f"sil<[{pause_duration}]>", pause_duration))

                current_sentence += ' ' + text
                current_duration += segment_duration

                if (len(current_sentence) > max_length) or (text[-1] in ".!?"):
                    sentences.append((current_sentence, current_duration))
                    current_sentence = ''
                    current_duration = 0
                    segment_duration = 0

                last_end_time = end_time

        if current_duration:
            sentences.append((current_sentence, current_duration))

        return sentences

    # Чтение файла субтитров
    subtitles_file = srt_name
    with open(subtitles_file, "r", encoding='UTF-8') as file:
        subtitles = file.read().split('\n\n')

    import glob

    # Очистка папки с WAV файлами
    for wav_file in glob.glob(temp_waves_folder + '*.wav'):
        os.remove(wav_file)

    # Разбор файла субтитров на отдельные сегменты
    sentences = group_sentences(subtitles)
    print(sentences)
    one_per = 100 / len(sentences)
    for i, (text, source_duration) in enumerate(sentences):
        update_progress(int(one_per * i) + 1)
        current_wav_file = os.path.join(temp_waves_folder, f'{i}'.zfill(5) + '.wav')
        if not text:
            silence = AudioSegment.silent(duration=source_duration)
            silence.export(current_wav_file, format="wav")
            continue
        # Генерация аудиофайла
        print(i, text, source_duration)
        model.save_wav(text=text,
                       speaker='en_11',
                       sample_rate=sample_rate, audio_path=current_wav_file)

        # Обработка аудиофайла для соответствия длительности
        audio = AudioSegment.from_wav(current_wav_file)
        current_duration = len(audio)
        print(f"current_duration: {current_duration} < source_duration: {source_duration}")
        if current_duration > source_duration:
            # Ускорение аудио
            audio = audio.speedup(playback_speed=current_duration / source_duration)
        elif current_duration < source_duration:
            # Проверка, если длительность файла меньше source_duration более чем в 1.5 раза
            if current_duration * 1.2 < source_duration:
                # Увеличение длительности файла в 1.2 раза
                stretch_ratio = 1.2
                new_duration = int(current_duration * stretch_ratio)
                audio = audio._spawn(audio.raw_data, overrides={
                    "frame_rate": int(audio.frame_rate / stretch_ratio)
                }).set_frame_rate(audio.frame_rate)
                silence_duration = source_duration - new_duration
            else:
                silence_duration = source_duration - current_duration

            # Добавление тишины в конце
            silence = AudioSegment.silent(duration=silence_duration)
            audio += silence

        audio.export(current_wav_file, format="wav")

    update_message(f"Запись речи в файл")
    # Объединение всех WAV файлов в одну дорожку
    combined = AudioSegment.empty()
    for wav_file in sorted(glob.glob(os.path.join(temp_waves_folder, '*.wav'))):
        segment = AudioSegment.from_wav(wav_file)
        combined += segment
    # Сохранение результата в одном файле
    target_wave = os.path.join(project_output_persistance, input_file_videoname_without_ext + '.wav')
    combined.export(target_wave, format="wav")

    # Конвертация в MP3
    target_mp3 = os.path.join(project_output_persistance, input_file_videoname_without_ext + '.mp3')
    combined.export(target_mp3, format="mp3")
    print(f"Объединенный аудиофайл сохранен в формате MP3: {target_mp3}")

    """Объединяем дорожки в видео"""

    import os

    # Пути к файлам
    video_path = input_file  # Путь к исходному видеофайлу
    audio_path = os.path.join(project_output_persistance, target_mp3)  # Путь к аудиофайлу
    subtitles_path = srt_name  # Путь к файлу субтитров
    output_video_path = os.path.join(project_output_persistance, input_file_videoname_without_ext + '_final.mp4')

    # Команда для удаления исходной аудиодорожки, добавления новой аудиодорожки и субтитров

    # перекодирование

    # !ffmpeg -i "$video_path" -i "$audio_path" -c:v libx264 -crf 23 -preset veryslow -map 0:v -map 1:a -map 0:a -c:a aac -b:a 192k -vf "subtitles=$subtitles_path" "$output_video_path"
    # Команда ffmpeg
    # subtitles_path = subtitles_path.replace('\\', '/')
    command = [
        "ffmpeg",
        "-y",  # Перезапись без подтверждения
        "-i", f'{video_path}',
        "-i", f'{audio_path}',
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "veryslow",
        "-map", "0:v",
        "-map", "1:a",
        "-map", "0:a",
        "-c:a", "aac",
        "-b:a", "192k",
        "-vf", f'subtitles={"output/" + input_file_videoname_without_ext + ".srt"}',
        f'{output_video_path}'
    ]

    update_message(f"Сборка")
    # Запуск команды
    subprocess.run(command, check=True)

    print(f"Объединенное видео с новой аудиодорожкой и субтитрами сохранено: {output_video_path}")

    import shutil
    def move_and_cleanup_files(input_file, output_dir, files_to_move):
        # Получаем имя файла без пути и расширения
        input_file_videoname_without_ext = os.path.splitext(os.path.basename(input_file))[0]

        # Создаем целевую папку для перемещения
        target_folder = os.path.join(output_dir, f"{input_file_videoname_without_ext}_final")
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # Перемещаем файлы
        for file in files_to_move:
            if os.path.exists(file):
                shutil.move(file, os.path.join(target_folder, os.path.basename(file)))

        # Удаление всех файлов в директории output, но не папок
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    #
    output_dir = project_output_persistance
    files_to_move = [srt_name, txt_name, vtt_name, tsv_name, json_name, output_video_path]

    move_and_cleanup_files(output_video_path, output_dir, files_to_move)
    update_message("Работа над файлом закончена")
    update_progress(100)
    # Вызов функции по завершении работы скрипта
    # on_script_finish(project_output_persistance)  # Укажите путь к выходной папке


class ConsoleRedirector(object):
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)  # Автопрокрутка к последней строке

    def flush(self):
        self.stdout.flush()

    def start_redirect(self):
        sys.stdout = self

    def stop_redirect(self):
        sys.stdout = self.stdout


def on_script_finish(output_folder):
    # Функция для создания всплывающего окна
    def create_popup():
        popup = tk.Toplevel()
        popup.title("Обработка завершена")

        message = tk.Label(popup, text="Файл обработан", font=("Arial", 14))
        message.pack(pady=10)

        def open_folder():
            os.startfile(output_folder)
            popup.destroy()

        open_button = tk.Button(popup, text="Открыть результат", command=open_folder)
        open_button.pack(pady=10)

    # Запуск функции создания всплывающего окна в главном потоке
    root.after(0, create_popup)


def run_script():
    global running
    if not running:
        exit()

    try:
        update_progress(0)
        update_message('Подготовка...')
        text.delete('1.0', tk.END)
        disable_events()
        disable_widgets()
        main()
        # on_script_finish(project_output_persistance)
    except Exception as e:
        console_redirector.start_redirect()
        update_message('Произошла ошибка! Не обработано')
        print(f"Произошла ошибка: {e}")
    finally:
        pass
        # # Удаление всех файлов в директории output, но не папок
        # for file in os.listdir(project_output_persistance):
        #     file_path = os.path.join(project_output_persistance, file)
        #     if os.path.isfile(file_path):
        #         os.remove(file_path)
        #
        # # Удаление всех файлов в директории waves, но не папок
        # for file in os.listdir(temp_waves_folder):
        #     file_path = os.path.join(temp_waves_folder, file)
        #     if os.path.isfile(file_path):
        #         os.remove(file_path)
        # if os.path.exists(audio_file):
        #     os.remove(audio_file)

        console_redirector.stop_redirect()
        enable_events()
        enable_widgets()


def on_closing():
    global running
    running = False  # Установка флага в False для завершения потоков
    root.destroy()
    exit(0)


def disable_events():
    def do_nothing(event):
        return "break"  # Возвращение "break" предотвратит дальнейшую обработку события

    root.bind_all("<Button-1>", do_nothing)  # Отключаем клики мыши
    root.bind_all("<Key>", do_nothing)  # Отключаем нажатия клавиш


def enable_events():
    root.unbind_all("<Button-1>")
    root.unbind_all("<Key>")


def set_widget_state(widget, state):
    if 'state' in widget.keys() and widget != text:
        widget.configure(state=state)


def disable_widgets():
    for widget in root.winfo_children():
        set_widget_state(widget, 'disable')


def enable_widgets():
    for widget in root.winfo_children():
        set_widget_state(widget, 'normal')


# Создание скрытого корневого окна
root = Tk()
# root.withdraw()
# Настройка цвета (не гарантирует изменения в диалоговом окне)
root.configure(bg='lightblue')
root.title("Перевод видео")
# Переменная для индикатора прогресса
progress_var = tk.IntVar()

# Виджет индикатора прогресса
progress_bar = ttk.Progressbar(root, length=300, variable=progress_var, maximum=100)
progress_bar.pack(pady=10)

# Метка для сообщений
message_label = tk.Label(root, text="Подготовка...", bg='red', font=("Arial", 20))
message_label.pack(pady=10)

# Кнопка для запуска
start_button = tk.Button(root, text="Запустить",
                         command=lambda: threading.Thread(target=run_script, daemon=False).start())
start_button.pack(pady=10)

# Кнопка для открытия папки с результатами
open_folder_button = tk.Button(root, text="Открыть папку", command=open_output_folder)
open_folder_button.pack(pady=10)

# Текстовое поле для вывода
text = tk.Text(root, height=20, width=120)
text.pack(pady=10)

# Создание объекта перенаправления
console_redirector = ConsoleRedirector(text)

root.protocol("WM_DELETE_WINDOW", on_closing)  # Привязка функции к закрытию окна

# Запуск основного цикла
root.mainloop()
