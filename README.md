# CoreClassification
Leushin Alexander, Golunov Arsenii 

2023

## Описание
Веб-приложение для классификации фаций на основе описания

## Датасет
В работе использовался документ, содержащий детальное макроскопическое изучение керна. Этот документ был получен путем анализа образцов керна, взятых из различных геологических формаций.

Для каждого образца были проведены различные измерения и оценки, включая анализ текстуры, глубину, ГК, литотипы и другие характеристики. Данные в документе были представлены в виде таблицы, где каждая строка соответствовала отдельному образцу керна, а каждый столбец содержал информацию о характеристиках 

![image](https://user-images.githubusercontent.com/72046996/222974778-50764ff3-632a-49be-9055-54853661bc94.png)

После парсинга и балансировки, был получен датасет, содержащий 5 классов и 172 записи

| Класс  | Количество записей |
| ------------- | ------------- |
| дистальный прирусловой вал | 55  |
| проксимальный прирусловой вал | 37  |
| отложения межрусловых площадей  | 33  |
| питающий канал | 29  |
| фронтальная зона проксимальной части лопасти | 18  |

## Эксперименты
Для экспериментов, использовалась модель XGBoost, а также наивный байесовский классификатор и многослойный персептрон. Данные модели показывают наилучшие результаты на малом количестве данных. 

| Модель  | Параметры | Точность, % |
| ------------- | ------------- | ------------- |
| XGBoost | n_estimators = 64 max_depth = 2 | 42.18 |
| XGBoost | n_estimators = 128 max_depth = 4 | 51.43 |
| XGBoost | n_estimators = 256 max_depth = 8 | 51.43 |
| GaussianNB | - | 42.86 |
| MLPClassifier | solver='lbfgs', alpha=1e5, hidden_layer_sizes=(8,24) | 37.14 |
| MLPClassifier | solver='lbfgs', alpha=1e5, hidden_layer_sizes=(16,32) | 45.71 |
| MLPClassifier | solver='lbfgs', alpha=1e5, hidden_layer_sizes=(32,64) | 50.72 |

## Результат
Для удобного взаимодействия с моделью был разработан веб-сервис с помощью фреймворка Flask, использующий в своей работе обученную модель XGBoost.

![image](https://user-images.githubusercontent.com/72046996/222975170-a4602746-1575-42e9-88c8-80003f9dd272.png)