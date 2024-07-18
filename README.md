# News-sentiment-analysis
Программа определяет тональность новостных заголовков, касающихся слова (словосочетания) phrase за период from_date – to_date:

```python
from NewsSentiment import GetSentiment

sentiment = GetSentiment(api_key, phrase, from_date, to_date)
```
Поиск новостей осуществляется при помощи newsapi. api_key можно получить на https://newsapi.org/

Лучшая модель — нейронная сеть со слоем эмбеддинга и LSTM, активация на выходе — tanh.
Тональность новостных заголовков лежит в отрезке от -1 до 1:
+ -1 — негативный заголовок
+ 0 — нейтральный
+ 1 — положительный

Использованные данные тональности новостей: https://github.com/WebOfRussia/financial-news-sentiment


Предобработка текста включает:
1) Удаление знаков препинания
2) Токенизация
3) Лемматизация (pymorphy2)
4) Удаление стоп-слов (nltk)

| Модель            | MAE           | $R^2$          |
| -------------     | ------------- | -------------  | 
| LSTM (Embedding)  | **0.33**      | **0.41**       | 
| ML models (TF-IDF)| Уточняется    | Уточняется     | 
| dostoevsky        | 0.79          | -2.15          | 
| Hugging Face      | 0.69          | -1.59          | 

