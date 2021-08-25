import numpy as np
import pandas as pd
from numpy.random import choice as ch
from sklearn.neighbors import BallTree
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD as trsvd
from sklearn.pipeline import make_pipeline
from vosk import Model, KaldiRecognizer  # оффлайн-распознавание от Vosk
import speech_recognition  # распознавание пользовательской речи (Speech-To-Text)
import wave  # создание и чтение аудиофайлов формата wav
import json  # работа с json-файлами и json-строками
import os  # работа с файловой системой
import pyttsx3

engine = pyttsx3.init()    # Инициализировать голосовой движок.

voices = engine.getProperty('voices')

for voice in voices:    # голоса и параметры каждого
    print('------')
    print(f'Имя: {voice.name}')
    print(f'ID: {voice.id}')
    print(f'Язык(и): {voice.languages}')
    print(f'Пол: {voice.gender}')
    print(f'Возраст: {voice.age}')


# Заменить на любой русский из консоли!
# Заменить на любой русский из консоли!
# Заменить на любой русский из консоли!
ru_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\IVONA 2 Voice Tatyana22"
engine.setProperty('voice', ru_voice_id)
print("Запуск...")
def softmax(x):
    proba = np.exp(-x)
    return proba/sum(proba)
    
class NeighborSampler(BaseEstimator):
    def __init__(self, k=5, temperature=1.0):
        self.k = k
        self.temperature = temperature
    def fit(self, X, y):
        self.tree_ = BallTree(X)
        self.y_ = np.array(y)
    def predict(self, X, random_state=None):
        distances, indices = self.tree_.query(X, return_distance=True, k=self.k)
        result = []
        for distance, index in zip(distances, indices):
            result.append(ch(index, p=softmax(distance*self.temperature)))
        return self.y_[result]






dataset = pd.read_csv('dialogs_nita.tsv', sep="\t")
dataset.sample(3);

vector = TfidfVectorizer()
vector.fit(dataset.context_0)
matrix = vector.transform(dataset.context_0)

svd=trsvd(n_components=300)
svd.fit(matrix)
matrix_mini = svd.transform(matrix)

ns = NeighborSampler()
ns.fit(matrix_mini, dataset.reply)



def record_and_recognize_audio(*args: tuple):
    """
    Запись и распознавание аудио
    """
    with microphone:
        recognized_data = ""

        # регулирование уровня окружающего шума
        recognizer.adjust_for_ambient_noise(microphone, duration=2)

        try:
            print("Listening...")
            audio = recognizer.listen(microphone, 5, 5)

            with open("microphone-results.wav", "wb") as file:
                file.write(audio.get_wav_data())

        except speech_recognition.WaitTimeoutError:
            print("Can you check if your microphone is on, please?")
            return

        # использование online-распознавания через Google 
        try:
            print("Started recognition...")
            recognized_data = recognizer.recognize_google(audio, language="ru").lower()

        except speech_recognition.UnknownValueError:
            pass

        # в случае проблем с доступом в Интернет происходит попытка 
        # использовать offline-распознавание через Vosk
        except speech_recognition.RequestError:
            print("Trying to use offline recognition...")
            recognized_data = use_offline_recognition()

        return recognized_data


def use_offline_recognition():
    """
    Переключение на оффлайн-распознавание речи
    :return: распознанная фраза
    """
    recognized_data = ""
    try:
        # проверка наличия модели на нужном языке в каталоге приложения
        if not os.path.exists("models/vosk-model-small-ru-0.4"):
            print("Please download the model from:\n"
                  "https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
            exit(1)

        # анализ записанного в микрофон аудио (чтобы избежать повторов фразы)
        wave_audio_file = wave.open("microphone-results.wav", "rb")
        model = Model("models/vosk-model-small-ru-0.4")
        offline_recognizer = KaldiRecognizer(model, wave_audio_file.getframerate())

        data = wave_audio_file.readframes(wave_audio_file.getnframes())
        if len(data) > 0:
            if offline_recognizer.AcceptWaveform(data):
                recognized_data = offline_recognizer.Result()

                # получение данных распознанного текста из JSON-строки
                # (чтобы можно было выдать по ней ответ)
                recognized_data = json.loads(recognized_data)
                recognized_data = recognized_data["text"]
    except:
        print("Sorry, speech service is unavailable. Try again later")

    return recognized_data


if __name__ == "__main__":

    # инициализация инструментов распознавания и ввода речи
    recognizer = speech_recognition.Recognizer()
    microphone = speech_recognition.Microphone()

    while True:
        # старт записи речи с последующим выводом распознанной речи
        # и удалением записанного в микрофон аудио
        
        voice_input = record_and_recognize_audio()
        print(voice_input)
        os.remove("microphone-results.wav")
        pipe=make_pipeline(vector, svd, ns)
        answer=pipe.predict([voice_input])
        print(answer)
        engine.say(answer)

        engine.runAndWait()