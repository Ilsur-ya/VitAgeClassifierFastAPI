#etc
import pandas as pd
import os
from io import BytesIO
#BOT
import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from aiogram.filters import Text
from aiogram import filters
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.utils.keyboard import InlineKeyboardBuilder
logging.basicConfig(level=logging.INFO)
BOT_TOKEN = "6940547909:AAHJp47vasKor-YB9PMJokUVvtoQi-8X-6Q"
bot = Bot(token="6940547909:AAHJp47vasKor-YB9PMJokUVvtoQi-8X-6Q", parse_mode="HTML") #session=session
dp = Dispatcher()


#NEURO
import requests
from PIL import Image
import sentencepiece
import re
import validators
from io import BytesIO
import torch
#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from transformers import ViTFeatureExtractor, ViTForImageClassification
import uvicorn

from fastapi import FastAPI

from transformers import pipeline


#For nlp bert
model_name_1 = 'distilbert-base-cased-distilled-squad'
model_name_2 = "rsvp-ai/bertserini-bert-base-squad"
nlp = pipeline('question-answering', model=model_name_1, tokenizer=model_name_1)

#For FAQ
tqa = pipeline(task="table-question-answering", model="google/tapas-large-finetuned-wtq")
# data = {"Buildings": ["A", "B", "V", "G", "D"],
#         "Path": ["from street or from building V",
#                  "only from building A",
#                  "from street, from building D (floor №3) or from building A"]}
data = {"Buildings": ["A", "B", "V", "G", "D"],
        "Path": ["с улицы или с корпуса В",
                 "только с корпуса А",
                 "с улицы, с корпуса Д (этаж №3) или с корпуса А",
                 "со второго этажа корпуса Д",
                 "с улицы или с корпусов Д, А"]}

table = pd.DataFrame.from_dict(data)

#Language
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer_translate = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
model_translate = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")

#VIT
# Init model, transforms
model_vit = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
transforms_vit = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')

# Хэндлер на команду /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    username = message.from_user.username

    builder = ReplyKeyboardBuilder()
    builder.add(types.KeyboardButton(text='👤Определить возраст'))
    builder.add(types.KeyboardButton(text='🛒Вопросы'))
    builder.adjust(2)

    await message.answer(f'Привет, {username}. Какой у тебя вопрос?', reply_markup=builder.as_markup(resize_keyboard=True))

@dp.message()
async def handle_text(message: types.Message):
    if message.text == "👤Определить возраст":
        await message.answer(f'Чтобы определить возраст пришли ссылку на изображение :)')
    elif message.text == "🛒Вопросы":
        await message.answer(
            f'Этот бот может ответить на вопросы связанные с местонахождением корпусов КГЭУ. Напиши свой вопрос.')
    elif message.content_type == 'photo':
        age = await get_age(message.photo)
        await message.answer('Этому человеку примерно ' + age["Age"] + 'лет')
    elif validators.url(message.text): # Проверяем, является ли текст сообщения ссылкой
        age = await get_age(message.text)
        await message.reply('Этому человеку примерно ' + age["Age"] + 'лет')
    else:
        answer = await get_answer_from_table(message.text)
        await message.reply(f"Можно пройти {answer}")# В противном случае, продолжаем обработку текстового сообщения

@dp.message(Text("👤Определить возраст"))
async def text_catalog(message: types.Message):
    await message.answer(f'Чтобы определить возраст пришли ссылку на изображение :)')

@dp.message(Text("🛒Вопросы"))
async def text_catalog(message: types.Message):
    await message.answer(f'Этот бот может ответить на вопросы связанные с местонахождением корпусов КГЭУ. Напиши свой вопрос.')

async def get_age(photo):
    if validators.url(photo) is True:

        im = Image.open(requests.get(photo, stream=True, verify=False).raw)
    else:
        file_id = photo[-1].file_id
        file_info = await bot.get_file(file_id)

        # Создаем URL файла
        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info.file_path}"

        # Загружаем изображение в память
        img_data = requests.get(file_url).content
        im = Image.open(BytesIO(img_data))

    # Transform our image and pass it through the model
    inputs = transforms_vit(im, return_tensors='pt')
    output = model_vit(**inputs)

    # Predicted Class probabilities
    proba = output.logits.softmax(1)

    # Predicted Classes
    preds = proba.argmax(1)
    imgplot = plt.imshow(im)
    plt.show()
    # print(output.logits)
    a = {'0': "0-2", '1': "3-9", '2':  "10-19", '3': "20-29", '4': "30-39", '5': "40-49", '6': "50-59", '7': "60-69", '8': "больше 70"}
    number = re.findall(r'\d+', str(preds[0]))
    result = a[number[0]]
    return {"Age": result}

async def get_answer_from_table(question):
    inputs = tokenizer_translate(question, return_tensors="pt")
    outputs = model_translate.generate(**inputs)
    translated_text = tokenizer_translate.decode(outputs[0], skip_special_tokens=True)
    result = tqa(table=table, query=translated_text)['cells'][0]

    return result

async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()