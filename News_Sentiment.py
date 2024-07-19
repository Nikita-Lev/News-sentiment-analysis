# coding: utf-8

import requests

from keras.models import  load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy2
import re
from datetime import datetime


max_seq_len = 13

def fetch_news(api_key, phrase, from_date, to_date):
    '''
    Поиск новостей при помощи newsapi
    '''
    url = f"https://newsapi.org/v2/everything?q={phrase}&from={from_date}&to={to_date}&sortBy=publishedAt&apiKey={api_key}"
    response = requests.get(url)
    return response.json()

def TextPreprocessing(text):
    '''
    Обработка текста:
        1) Удаление знаков препинания
        2) Токенизация
        3) Лемматизация
        4) Удаление стоп-слов
        
    Parameters
    ----------
    text : str 
        Строка для обработки
    
    Returns
    -------
    str
        Обработанный текст
    '''
    
    text = re.sub("[^a-zA-Zа-яА-Я1234567890]"," ", text.lower()) # Удаление знаков препинания и приведение к нижнему регистру
    
    # Лемматизация
    lemmatizer = pymorphy2.MorphAnalyzer()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.parse(token)[0].normal_form for token in tokens]

    # Удаление стоп-слов
    noise = stopwords.words("russian")
    tokens = [w for w in tokens if not w in noise]

    return ' '.join(tokens)

def GetSentiment(api_key, phrase, from_date, to_date):
    '''
    Получение тональности новостных заголовков
    
    Parameters
    ----------
    api_key : str 
        api_key from https://newsapi.org/
    phrase : str
        Строка, по которой будет произведен поиск новостей
    from_date : str in format YYYY-MM-DD
        Дата начала поиска новостей
    to_date : str in format YYYY-MM-DD
        Дата окончания поиска новостей
    
    Returns
    -------
    dict
        Словарь списков: {'Date' : [...], 'Time' : [...], 'Title' : [...], 'Sentiment' : [...]}
    '''
    news_data = fetch_news(api_key, phrase, from_date, to_date)
    
    news_sentiment = {'Date' : [], 'Time' : [], 'Title' : [], 'Sentiment' : []}
    
    if len(news_data['articles']) == 0:
        print('Новости отсутствуют. Попробуйте изменить период или фразу для поиска.')
        return news_sentiment 
    
    prepared_text = []
    for article in news_data['articles']:
        news_sentiment['Date'].append(datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d')) 
        news_sentiment['Time'].append(datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ').strftime('%H:%M'))
        news_sentiment['Title'].append(article['title'])
        
        prepared_text.append(TextPreprocessing(article['title']))
    
    # Load tokenizer
    with open("Tokenizer.txt", "r") as text_file:
        tokenizer = tokenizer_from_json(text_file.read())
    
    sequences = tokenizer.texts_to_sequences(prepared_text)
    data = pad_sequences(sequences, maxlen = max_seq_len)
    
    model = load_model('model.h5')
    sentiment = model.predict(data)
    
    news_sentiment['Sentiment'] = sentiment.reshape(-1).tolist()
        
    return news_sentiment

