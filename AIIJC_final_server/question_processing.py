import nltk

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

stopwords = stopwords.words("english")

from nltk.tokenize import RegexpTokenizer, word_tokenize, sent_tokenize

main_tokenizer = RegexpTokenizer(r'\w+', )
sec_tokenizer = RegexpTokenizer(r'\S+')


import joblib
import numpy as np
import urllib
import difflib
import WikiSearch.wikipedia.wikipedia as wikipedia
from bs4 import BeautifulSoup

model = joblib.load('/Data/AIIJC/aiijc_1578_goodFromTrain_pretrained.model')
model.args.max_seq_length = 512
model.args.silent = True

def normal_form(word):  # Получение нормальной формы слова
    word = word.lower()
    return word


def clean_html(html):  # Очистка html
    soup = BeautifulSoup(BeautifulSoup(html, "lxml").text)
    return str(soup.body)


def get_good_tokens(text):  # Выделение ключевых слов
    good_tokens = []

    for tokens in tokenizer(text)[1]:
        for token in tokens:
            token = normal_form(token)
            if token not in stopwords:
                good_tokens.append(token)
    return good_tokens


def tokenizer(text):  # Токенизация текста в обработанные и необработанные токены
    raw_tokens = sec_tokenizer.tokenize(text)
    clean_tokens = main_tokenizer.tokenize_sents(raw_tokens)

    nClean_tokens = []
    for i in range(len(clean_tokens)):
        nClean_tokens.append([])
        for m in range(len(clean_tokens[i])):
            if normal_form(clean_tokens[i][m]) != 's':
                nClean_tokens[i].append(normal_form(clean_tokens[i][m]))

    return (raw_tokens, nClean_tokens)


def similarity(s1, s2):  # Нахождение коэффициента схожести между двумя строками
    normalized1 = s1.lower()
    normalized2 = s2.lower()
    matcher = difflib.SequenceMatcher(None, normalized1, normalized2)
    return matcher.ratio()


def part_extractor(data, question, step,
                   part_length):  # Функция выделения релевантного фрагмента (Текст, вопрос, длинна фрагмента)
    good_tokens = get_good_tokens(question)

    tokens = tokenizer(data)

    for i in range(step - (len(tokens[0]) % step)):  # Увеличение количества токенов до кратного длины части
        tokens[0].append('')
        tokens[1].append('')

    match_counter = 0  # Счетчик точных совпадений токенов
    best_part = ''  # Лучшая часть
    max_match_qty = 0  # Максимальное количество совпавших токенов

    main_clrTokens = tokens[1]
    main_tokens = tokens[0]

    for i in range(0, len(tokens[0]) - 1, part_length):  # Нахождение наиболее релевантной части текста
        tokens = main_tokens[i:i + part_length - 1]
        clrTokens = main_clrTokens[i:i + part_length - 1]

        for good_token in good_tokens:
            if in_tokens(good_token, clrTokens):
                match_counter += 1

        if match_counter > max_match_qty:
            max_match_qty = match_counter
            best_part = tokens

        match_counter = 0

    fin = ''  # Восстановление текста
    for i in best_part:
        fin += (i + ' ')

    return fin


def in_tokens(token, text):
    for i in text:
        for m in i:
            if token == m:
                return True
    return False


def answering(question):
    text = question

    good_tokens = get_good_tokens(text)

    try:
        urls = wikipedia.search(text, results=2)
    except:
        link_1 = '-'
        link_2 = '-'

    try:
        link_1 = urls[0]
    except:
        link_1 = '-'

    try:
        link_2 = urls[1]
    except:
        link_2 = '-'

    # Загрузка статей википедии
    try:
        link_1 = link_1.replace('https://en.wikipedia.org/wiki/', '')  # Убераем начало ссылки
        link_1 = urllib.parse.unquote(link_1)  # Заменяем кривые символы на оригинал
        data_1 = wikipedia.page(link_1, auto_suggest=False).content  # Парсим страничку вики
        data_1 = data_1.replace('\n', ' ')
    except:
        pass
    try:
        link_2 = link_2.replace('https://en.wikipedia.org/wiki/', '')  # Убераем начало ссылки
        link_2 = urllib.parse.unquote(link_2)  # Заменяем кривые символы на оригинал
        data_2 = wikipedia.page(link_2, auto_suggest=False).content  # Парсим страничку вики
        data_2 = data_2.replace('\n', ' ')
    except:
        pass

    try:  # Поиск релевантного куска длиной 128 токенов с шагом 64 в самой релевантной статье
        context = part_extractor(data_1, question, 16, 64)
    except:
        pass

    try:  # Поиск релевантного куска длиной 64 токена с шагом 32 во второй по релевантности статье
        context += ' ' + part_extractor(data_2, question, 16, 32)
    except:
        pass

    try:
        predict = model.predict([{'context': context, 'qas': [{'id': 0, 'question': question}]}])[
            0]  # Предсказание ответа
    except:
        predict = [{'answer': ['']}]
        predict[0]['answer'][0] = 'empty'

    if predict[0]['answer'][0] == 'empty':
        try:
            context = part_extractor(data_1, question, 16, 64)
            predict = model.predict([{'context': context, 'qas': [{'id': 0, 'question': question}]}])[0]
        except:
            pass

    if predict[0]['answer'][0] == 'empty':
        try:
            context = part_extractor(data_2, question, 16, 64)
            predict = model.predict([{'context': context, 'qas': [{'id': 0, 'question': question}]}])[0]
        except:
            pass

    if predict[0]['answer'][0] == 'empty':
        try:
            context = part_extractor(data_1, question, 16, 128)
            predict = model.predict([{'context': context, 'qas': [{'id': 0, 'question': question}]}])[0]
        except:
            pass

    if predict[0]['answer'][0] == 'empty':
        try:
            context = part_extractor(data_2, question, 16, 128)
            predict = model.predict([{'context': context, 'qas': [{'id': 0, 'question': question}]}])[0]
        except:
            pass

    if predict[0]['answer'][0] == 'empty':
        try:
            context = part_extractor(data_1, question, 16, 256)
            predict = model.predict([{'context': context, 'qas': [{'id': 0, 'question': question}]}])[0]
        except:
            pass

    if predict[0]['answer'][0] == 'empty':
        try:
            context = part_extractor(data_2, question, 16, 256)
            predict = model.predict([{'context': context, 'qas': [{'id': 0, 'question': question}]}])[0]
        except:
            pass

    return predict[0]['answer'][0]


def qa_answering(question, text):
    try:  # Поиск релевантного куска длиной 128 токенов с шагом 64 в самой релевантной статье
        context = part_extractor(text, question, 16, 64)
    except:
        pass

    try:
        predict = model.predict([{'context': context, 'qas': [{'id': 0, 'question': question}]}])[0]  # Предсказание ответа
    except:
        predict = [{'answer': ['']}]
        predict[0]['answer'][0] = 'empty'

    if predict[0]['answer'][0] == 'empty':
        try:
            context = part_extractor(text, question, 16, 64)
            predict = model.predict([{'context': context, 'qas': [{'id': 0, 'question': question}]}])[0]
        except:
            pass

    if predict[0]['answer'][0] == 'empty':
        try:
            context = part_extractor(text, question, 16, 128)
            predict = model.predict([{'context': context, 'qas': [{'id': 0, 'question': question}]}])[0]
        except:
            pass

    if predict[0]['answer'][0] == 'empty':
        try:
            context = part_extractor(text, question, 16, 256)
            predict = model.predict([{'context': context, 'qas': [{'id': 0, 'question': question}]}])[0]
        except:
            pass

    return predict[0]['answer'][0]
