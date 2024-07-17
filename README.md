# News-sentiment-analysis

Предобработка текста:
1) Удаление знаков препинания
2) Токенизация
3) Лемматизация (pymorphy2)
4) Удаление стоп-слов (nltk)

| Модель            | MAE           | $R^2$          | MAPE
| -------------     | ------------- | -------------  | -------------
| LSTM (Embedding)  | 0.33          | 0.41           | 83%
| ML models (TF-IDF)| Уточняется    | Уточняется     | Уточняется
| dostoevsky        | 0.79          | -2.15          | 214%
| Hugging Face      | 0.69          | -1.59          | 193%
