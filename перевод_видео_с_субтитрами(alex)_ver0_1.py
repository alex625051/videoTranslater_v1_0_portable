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
import os
from tkinter import filedialog
from tkinter import Tk
import pkg_resources
import subprocess
import sys


def is_package_installed(package_name):
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False


def install_package(git_url):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", git_url])
    except subprocess.CalledProcessError as e:
        print(f"Не удалось установить пакет: {e}")


# Установка языков для транскрибации и перевода
#####################################
source_language = "Russian"
target_language = "English"
tts_model_name = 'v3_en.pt'
speaker = 'en_11'
tts_url = 'https://models.silero.ai/models/tts/en/v3_en.pt'
trans_variant = '2'  # '1':траскрибация на языке;  '2': транскрибация+перевод
DEBUG = True
project_persistance_folder = "./"
test_file = './immunitet-cast-3-allergiceskie-reakcii_(VIDEOMIN.NET).mp4'
#####################################
project_output_persistance = os.path.join(project_persistance_folder, 'output')
project_persistance_folder_tts = os.path.join(project_persistance_folder, 'tts_models')
project_persistance_tts_modelfile = os.path.join(project_persistance_folder_tts, tts_model_name)
os.makedirs(project_persistance_folder, exist_ok=True)
os.makedirs(project_output_persistance, exist_ok=True)
os.makedirs(project_persistance_folder_tts, exist_ok=True)

import os

# Путь к папке для хранения модели large_v3 на Google Диске
model_path = "./whisper3/large-v3"
os.makedirs(model_path, exist_ok=True)

# Блок для загрузки файла 
# Создание скрытого корневого окна
root = Tk()
root.withdraw()

# Запрос файла через диалоговое окно
if DEBUG and test_file and os.path.exists(test_file):
    input_file = test_file
else:
    input_file = filedialog.askopenfilename()

# Проверка, был ли выбран файл
if not input_file:
    print("Файл не был выбран.")
    exit()

# Вывод пути к выбранному файлу
print(f"Выбранный файл: {input_file}")

input_file_basename = os.path.basename(input_file)
input_file_videoname_without_ext = os.path.splitext(input_file_basename)[0]
input_file_videoname_only_ext = os.path.splitext(input_file_basename)[1]

# Commented out IPython magic to ensure Python compatibility.
#########################################
# Загрузка нейросети и процессы обработки 1
#########################################
package_name = "whisper"  # Имя пакета, которое обычно используется для импорта
git_url = "git+https://github.com/alex625051/whisper.git"

if not is_package_installed(package_name):
    print(f"Установка пакета '{package_name}'...")
    install_package(git_url)
else:
    print(f"Пакет '{package_name}' уже установлен.")

"""**Загрузка модели**"""

model_file = os.path.join(model_path, "large-v3.pt")
os.path.isfile(model_file)

# Commented out IPython magic to ensure Python compatibility.
#########################################
# Загрузка нейросети и процессы обработки 2
#########################################
import whisper

# Это будет захватывать весь вывод ячейки, если debug = False


# Проверка наличия модели large_v3

if not os.path.isfile(model_file):
    # Загрузка модели с OpenAI, если она отсутствует
    model = whisper.load_model("large-v3", download_root=model_path)
else:
    # Загрузка модели из кеша
    model = whisper.load_model(model_file)

# Commented out IPython magic to ensure Python compatibility.
import whisper

# Это будет захватывать весь вывод ячейки, если debug = False


# Извлечение аудиодорожки из видео
# !ffmpeg -i "$input_file" -vn -acodec pcm_s16le -ar 44100 -ac 2 source_audio_track.wav
# Команда ffmpeg для извлечения аудиодорожки
command = [
    'ffmpeg',
    '-i', f'{input_file}',  # Исходный файл
    '-vn',  # Отключение обработки видео
    '-acodec', 'pcm_s16le',  # Кодек аудио
    '-ar', '44100',  # Частота дискретизации
    '-ac', '2',  # Количество аудиоканалов
    'source_audio_track.wav'  # Выходной файл
]

# Запуск команды
subprocess.run(command, check=True)

"""**Вариант транскрибации №1 без перевода**"""

# Commented out IPython magic to ensure Python compatibility.
if int(trans_variant) == 1:
    import whisper

    # Это будет захватывать весь вывод ячейки, если debug = False
    # Транскрибация и перевод текста
    result = model.transcribe("source_audio_track.wav", verbose=True, initial_prompt='Лекция по медицине',
                              output_dir=project_output_persistance, language=source_language)

if int(trans_variant) == 1:
    # Сохранение результатов в формате txt и субтитров с таймкодами
    output_source_text_file = os.path.join(project_output_persistance, "transcription.txt")
    output_source_srt_file = os.path.join(project_output_persistance, "subtitles.srt")

    with open(output_source_text_file, "w") as f:
        f.write(result["text"])

    with open(output_source_srt_file, "w") as f:
        for i, segment in enumerate(result["segments"], 1):
            f.write(f"{i}\n")
            f.write(f"{segment['start']} --> {segment['end']}\n")
            f.write(f"{segment['text']}\n\n")

"""**Вариант транскрибации №2 с переводом**"""

# Commented out IPython magic to ensure Python compatibility.
if int(trans_variant) == 2:
    import whisper

    # Это будет захватывать весь вывод ячейки, если debug = False
    # Транскрибация и перевод текста
    initial_prompt = "A lecture on medicine. Medical terms are used. Lecture for medical university students. Лекция по медицине. Используются медицинские термины. Лекция для студентов медицинского университета."
    result_AUTOTRANSLATED = model.transcribe("source_audio_track.wav", verbose=True, initial_prompt=initial_prompt,
                                             task='translate',
                                             # output_dir=project_output_persistance,
                                             # output_format='all'
                                             )

# Commented out IPython magic to ensure Python compatibility.
if int(trans_variant) == 2:
    # Сохранение результатов в формате txt и субтитров с таймкодами
    output_source_text_file_AUTOTRANSLATED = os.path.join(project_output_persistance,
                                                          "transcription_in_target_lang.txt")
    output_source_srt_file_AUTOTRANSLATED = os.path.join(project_output_persistance, "subtitles_in_target_lang.srt")

    with open(output_source_text_file_AUTOTRANSLATED, "w") as f:
        f.write(result_AUTOTRANSLATED["text"])

    with open(output_source_srt_file_AUTOTRANSLATED, "w") as f:
        for i, segment in enumerate(result_AUTOTRANSLATED["segments"], 1):
            f.write(f"{i}\n")
            f.write(f"{segment['start']} --> {segment['end']}\n")
            f.write(f"{segment['text']}\n\n")

from whisper.utils import get_writer

srt_name = os.path.join(project_output_persistance, input_file_videoname_without_ext + '.srt')
txt_name = os.path.join(project_output_persistance, input_file_videoname_without_ext + '.txt')
vtt_name = os.path.join(project_output_persistance, input_file_videoname_without_ext + '.vtt')
tsv_name = os.path.join(project_output_persistance, input_file_videoname_without_ext + '.tsv')
json_name = os.path.join(project_output_persistance, input_file_videoname_without_ext + '.json')

"""**Перевод текста (если получена транскрибация без перевода)**"""

if int(trans_variant) == 1:
    pass

"""**Синтез речи**"""

# Commented out IPython magic to ensure Python compatibility.
required_packages = ["silero", "pydub", "torch", "librosa"]

for package in required_packages:
    if not is_package_installed(package):
        print(f"Установка пакета '{package}'...")
        install_package(package)
    else:
        print(f"Пакет '{package}' уже установлен.")

import os
import torch
import librosa
from pydub import AudioSegment
import re

device = torch.device('cpu')
torch.set_num_threads(4)
local_file = project_persistance_tts_modelfile

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file(tts_url,
                                   local_file)

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)
sample_rate = 48000

temp_waves_folder = './waves/'
os.makedirs(temp_waves_folder, exist_ok=True)


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
with open(subtitles_file, "r") as file:
    subtitles = file.read().split('\n\n')

import glob

# Очистка папки с WAV файлами
for wav_file in glob.glob(temp_waves_folder + '*.wav'):
    os.remove(wav_file)

# Разбор файла субтитров на отдельные сегменты
sentences = group_sentences(subtitles)
print(sentences)
for i, (text, source_duration) in enumerate(sentences):
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

Audio(os.path.join(temp_waves_folder, f'{2}'.zfill(5) + '.wav'), autoplay=True)

# Объединение всех WAV файлов в одну дорожку
combined = AudioSegment.empty()
for wav_file in sorted(glob.glob(temp_waves_folder + '*.wav')):
    segment = AudioSegment.from_wav(wav_file)
    combined += segment
# Сохранение результата в одном файле
target_wave = os.path.join(project_output_persistance, input_file_videoname_without_ext + '.wav')
combined.export(target_wave, format="wav")

# Конвертация в MP3
target_mp3 = os.path.join(project_output_persistance, input_file_videoname_without_ext + '.mp3')
combined.export(target_mp3, format="mp3")
print(f"Объединенный аудиофайл сохранен в формате MP3: {target_mp3}")

srt_name

"""Объединяем дорожки в видео"""

import os

# Пути к файлам
video_path = input_file  # Путь к исходному видеофайлу
audio_path = os.path.join(project_output_persistance, input_file_videoname_without_ext + '.mp3')  # Путь к аудиофайлу
subtitles_path = srt_name  # Путь к файлу субтитров
output_video_path = os.path.join(project_output_persistance, input_file_videoname_without_ext + '_final.mp4')

# Команда для удаления исходной аудиодорожки, добавления новой аудиодорожки и субтитров

# перекодирование

# !ffmpeg -i "$video_path" -i "$audio_path" -c:v libx264 -crf 23 -preset veryslow -map 0:v -map 1:a -map 0:a -c:a aac -b:a 192k -vf "subtitles=$subtitles_path" "$output_video_path"
# Команда ffmpeg
command = [
    "ffmpeg",
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
    "-vf", f'subtitles={subtitles_path}',
    f'{output_video_path}'
]

# Запуск команды
subprocess.run(command, check=True)

print(f"Объединенное видео с новой аудиодорожкой и субтитрами сохранено: {output_video_path}")
