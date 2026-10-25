# Ответы по коду лабораторной работы 2

Материал относится к упрощённому [lab2_classification.py](lab2_classification.py) и [ноутбуку](lab2_classification.ipynb). Нумерация совпадает с первым блоком вопросов из задания. Дополнительные проверки можно выполнить в отдельной ячейке после обучения; они не обязательно уже напечатаны в основном листинге.

Положительный класс – **Cammeo (1)**, отрицательный – **Osmancik (0)**. Обучение: 2286 объектов, validation: 762, test: 762. Все итоговые модели обучены на одном train, без последующего объединения с validation.

| Кодовая ячейка | Блок в `.py` | Назначение |
| ---: | :--- | :--- |
| 1 | `imports` | Библиотеки, seed, каталоги |
| 2 | `data` | Загрузка и проверка данных |
| 3–4 | `distributions`, `correlations` | Анализ распределений и связей |
| 5 | `split` | Разбиение и CV |
| 6 | `metrics` | Общая функция метрик |
| 7 | `models` | Модели и сетки параметров |
| 8–9 | `tune`, `parameter_plots` | Подбор и его анализ |
| 10–12 | `neural_setup`, `neural_train`, `learning_plots` | Нейросетевые эксперименты |
| 13–16 | `selection`, `test`, `comparison_plots`, `confusion` | Выбор и итоговое сравнение |

## 1. Где загружается датасет? Как проверить размер, названия признаков и типы данных?

Блок `data`:

```python
df = pd.read_csv(BASE / 'data/rice.csv')
df.info()
```

CSV содержит строку заголовков; запятая – разделитель по умолчанию. `BASE` позволяет запускать программу из каталога лабораторной или корня проекта.

Для дополнительных проверок:

```python
print(df.shape)
print(df.columns.tolist())
print(df.dtypes)
```

Размер – **(3810, 8)**: семь числовых столбцов и текстовая цель `Class`. `info()` показывает также количество непустых значений. Исходные данные UCI и их CSV-копия лежат в `data`.

## 2. Где формируются X и y? Какие столбцы в них входят?

В том же блоке:

```python
X = df.drop(columns='Class')
y = df['Class'].map({'Osmancik': 0, 'Cammeo': 1})
```

`X` содержит **Area, Perimeter, Major_Axis_Length, Minor_Axis_Length, Eccentricity, Convex_Area, Extent**. Размер – **(3810, 7)**. `y` – 3810 числовых меток сорта. Цель исключена из входных признаков, поэтому модель не получает правильный ответ на вход.

## 3. Как проверяется количество объектов каждого класса? Каков результат?

```python
print(df['Class'].value_counts())
```

Результат:

| Сорт | Объектов | Доля |
| :--- | ---: | ---: |
| Osmancik | 2180 | 57,22% |
| Cammeo | 1630 | 42,78% |

Дисбаланс умеренный. Постоянный прогноз Osmancik дал бы примерно 57,22% Accuracy, поэтому нельзя оценивать классификатор только по факту превышения 50%. При разбиении используется стратификация.

## 4. Где ищутся и обрабатываются пропуски? Что будет при появлении пропуска?

```python
print('Пропуски:', df.isna().sum().sum())
df = df.dropna()
```

Первая сумма считает пропуски по столбцам, вторая – во всей таблице. `dropna()` удаляет строку, содержащую хотя бы один пропуск, до формирования X и y. Сейчас пропусков **0**, осталось **3810 строк**.

Если pandas распознает новое значение как `NaN`, соответствующая строка будет удалена и размеры выборок изменятся. Код не выполняет заполнение средним. Если пропуск записан нестандартным текстом, например `?`, его сначала нужно объявить через `na_values=['?']` при чтении; неизвестное текстовое значение автоматически не становится пропуском.

## 5. Как категории и метки переводятся в числа? Как восстановить название класса?

Категориальных **входных** признаков нет. Текстовая цель преобразуется явно:

```python
y = df['Class'].map({'Osmancik': 0, 'Cammeo': 1})
```

Обратное преобразование в дополнительной ячейке:

```python
class_names = {0: 'Osmancik', 1: 'Cammeo'}
labels = pd.Series(prediction).map(class_names)
print(labels.head())
```

Квантильное кодирование в MultinomialNB/ComplementNB относится к интервалам числовых признаков. Его не следует путать с кодированием исходных текстовых столбцов.

## 6. Где разделяются выборки? Как заданы test_size, random_state и stratify?

Блок `split`:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=SEED, stratify=y_train
)
```

Сначала выделяется 20% в test. Затем 25% от оставшихся 80% выделяется в validation: `0,25 × 0,8 = 0,2`. Итого **60/20/20**, то есть **2286/762/762** объектов.

`SEED=42` фиксирует перемешивание. `stratify` сохраняет пропорции сортов: train содержит 978 Cammeo и 1308 Osmancik, validation и test – по 326 и 436 соответственно. Во втором вызове переменные `X_train` и `y_train` заменяются окончательным train.

## 7. Где масштабируются признаки? На чём вызываются fit и transform?

Для классических методов scaler включён в Pipeline:

```python
models['LDA'] = Pipeline([
    ('scale', StandardScaler()),
    ('model', LinearDiscriminantAnalysis())
])
```

Такой конвейер находится в словаре `models`. При `search.fit(X_train, y_train)` scaler обучается внутри каждой обучающей части CV. Для итогового `best_estimator_` он настраивается на всём train, а при `predict` только преобразует validation/test.

Для нейросети операции видны напрямую:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype('float32')
X_val_scaled = scaler.transform(X_val).astype('float32')
X_test_scaled = scaler.transform(X_test).astype('float32')
```

`fit` узнаёт обучающие средние и масштабы, `transform` применяет их. Такой порядок исключает использование статистик validation/test при настройке. Дерево не требует масштабирования; для двух счётных вариантов Байеса вместо scaler используется дискретизация.

## 8. Где создаётся наивный Байес? Почему выбранный вариант подходит?

Основной вариант для числовых измерений – GaussianNB:

```python
models['GaussianNB'] = Pipeline([
    ('scale', StandardScaler()),
    ('model', GaussianNB())
])
```

Он предполагает нормальные условные распределения признаков внутри класса. Это допустимая модель для непрерывных измерений; выполнение её предположений не гарантируется и проверяется практическим качеством.

Также реализованы три варианта из задания:

- **MultinomialNB, ComplementNB:** `KBinsDiscretizer` разбивает значения на квантильные интервалы и формирует индикаторы интервалов. Непрерывные площади не подаются как частоты событий напрямую.
- **BernoulliNB:** `StandardScaler → Binarizer → BernoulliNB(binarize=None)`. На вход классификатору поступают готовые бинарные индикаторы.

Таким образом, требования к представлению данных учтены отдельно для каждой разновидности.

## 9. Где параметры Байеса? Как изменить один из них?

В `param_grids` заданы значения `model__var_smoothing` GaussianNB: **1e-12, 1e-10, 1e-9, 1e-8, 1e-6, 1e-4, 1e-2**.

`var_smoothing` добавляет стабилизирующую величину к дисперсиям и уменьшает чувствительность к очень малым оценкам дисперсии. В текущем поиске выбрано **0,0001**. Ручной пример изменения, не отдельный выполненный эксперимент отчёта:

```python
models['GaussianNB'].set_params(model__var_smoothing=1e-4)
```

Для MultinomialNB/ComplementNB меняется `alpha` – сглаживание счётчиков, а также число интервалов. Для BernoulliNB – `alpha` и порог бинаризации. `model__` указывает на параметр шага `model` внутри Pipeline.

## 10. Где создаётся дерево? Как заданы критерий, глубина и ограничения на объекты?

```python
models['Decision Tree'] = Pipeline([
    ('model', DecisionTreeClassifier(random_state=SEED))
])
```

По умолчанию критерий – `gini`, глубина не ограничена. В сетке проверяются:

- `criterion`: `gini`, `entropy`;
- `max_depth`: 2, 3, 5, 8, None;
- `min_samples_leaf`: 1, 5, 15.

Выбраны **entropy, max_depth=3, min_samples_leaf=15**. `min_samples_leaf` ограничивает число объектов в листе; `min_samples_split` – минимальное число объектов для попытки разделить узел. Последний параметр не подбирается и остаётся равным 2.

## 11. Как получить глубину, число листьев и важности признаков? Каков результат?

Дополнительная ячейка после обучения:

```python
tree = best_models['Decision Tree'].named_steps['model']
print(tree.get_depth())
print(tree.get_n_leaves())
print(pd.Series(tree.feature_importances_, index=X.columns))
```

Для сохранённого дерева получено: **глубина 3, листьев 8, всего узлов 15**.

| Признак | Важность |
| :--- | ---: |
| Major_Axis_Length | 0,988588 |
| Perimeter | 0,011412 |
| Area | 0 |
| Minor_Axis_Length | 0 |
| Eccentricity | 0 |
| Convex_Area | 0 |
| Extent | 0 |

Это нормированные уменьшения неоднородности при разбиениях конкретного дерева, а не причинная важность. Нулевое значение означает, что дерево не использовало признак; при сильной связи признаков информация одного могла быть представлена другим.

## 12. Где создаётся LDA? Что задано явно и что оставлено по умолчанию?

```python
models['LDA'] = Pipeline([
    ('scale', StandardScaler()),
    ('model', LinearDiscriminantAnalysis())
])
```

При создании параметры классификатора не меняются. Существенные значения по умолчанию: `solver='svd'`, `shrinkage=None`, `priors=None`, `tol=0.0001`. При `priors=None` априорные доли классов оцениваются на обучающих данных.

В сетке отдельно исследуются `svd` с двумя `tol` и `lsqr` с вариантами shrinkage. Выбран **solver='svd', tol=0.0001**, то есть базовая настройка. Shrinkage применяется только с совместимым решателем и в итоговом варианте не используется.

## 13. Где создаётся SVM? Как выбираются ядро, C и gamma?

```python
models['SVM'] = Pipeline([
    ('scale', StandardScaler()),
    ('model', SVC(random_state=SEED))
])
```

Базовые параметры: RBF-ядро, `C=1`, `gamma='scale'`. В сетке исследуются линейное и RBF-ядра, `C` от 0,1 до 100, а для RBF также несколько `gamma`.

Выбран **kernel='linear', C=100**. Для линейного ядра gamma не влияет на результат. C регулирует штраф за нарушения разделения: большое C сильнее штрафует ошибки на train. Для RBF большое gamma делает влияние отдельного объекта более локальным. SVM получает стандартизованные признаки.

## 14. Где создаётся kNN? Как заданы соседи, веса и расстояние?

```python
models['kNN'] = Pipeline([
    ('scale', StandardScaler()),
    ('model', KNeighborsClassifier())
])
```

Базовый вариант: 5 соседей, одинаковые веса, расстояние Минковского с `p=2`, то есть евклидово. Проверяются `n_neighbors=[3, 5, 11, 21, 41]`, `weights=['uniform', 'distance']`, `p=[1, 2]`.

Выбраны **41 сосед, uniform, p=2**. `p=1` соответствует манхэттенскому расстоянию, `p=2` – евклидову. При `distance` близкие соседи имеют больший вес. Масштабирование важно, иначе большие по числу значения площади доминировали бы в расстоянии.

## 15. Где fit и predict для всех пяти методов? Что передаётся и возвращается?

Все восемь реализаций пяти семейств проходят общий цикл `tune`:

```python
for name, model in models.items():
    search = GridSearchCV(model, param_grids[name], cv=cv,
                          scoring='roc_auc', error_score='raise')
    search.fit(X_train, y_train)
    best_models[name] = search.best_estimator_
    prediction = search.predict(X_val)
```

`fit` получает матрицу признаков и известные метки. GridSearchCV обучает копии конвейера на фолдах и затем лучший вариант на всём train. `fit` возвращает обученный объект, а не прогнозы.

В блоке `test`:

```python
for name, model in best_models.items():
    prediction = model.predict(X_test)
```

`predict` получает только признаки и возвращает массив из 762 меток 0/1. В этом блоке модели не переобучаются.

## 16. Где считается Accuracy? Какие массивы используются?

В общей функции:

```python
accuracy_score(y_true, y_pred)
```

На тесте она вызывается через `calculate_metrics(y_test, prediction, scores)`: `y_test` – истинные классы, `prediction` – прогнозы тех же строк в том же порядке.

Например, у выбранного SVM правильны **705 из 762** меток: Accuracy = **0,925197**, или **92,52%**. Здесь Accuracy – доля правильных ответов, не Precision положительного класса.

## 17. Как считаются Precision, Recall и F1? Какой average?

```python
precision_score(y_true, y_pred, zero_division=0)
recall_score(y_true, y_pred, zero_division=0)
f1_score(y_true, y_pred, zero_division=0)
```

`average` явно не передаётся, поэтому используется **average='binary'**, `pos_label=1`. Это соответствует бинарной цели и заранее выбранному Cammeo. `zero_division=0` задаёт ноль, если знаменатель метрики равен нулю.

При желании эти настройки можно явно написать в дополнительном примере:

```python
precision_score(y_test, prediction, average='binary', pos_label=1, zero_division=0)
```

Это не macro-усреднение по двум классам. На test SVM: Precision=**0,903904**, Recall=**0,923313**, F1=**0,913505**.

## 18. Как задаётся положительный класс и порядок классов?

Он зафиксирован кодированием:

```python
y = df['Class'].map({'Osmancik': 0, 'Cammeo': 1})
```

В бинарных метриках по умолчанию положителен 1. У обученных конвейеров `classes_` равно **[0, 1]**, поэтому `predict_proba(X)[:, 1]` относится к Cammeo. Проверка в отдельной ячейке:

```python
print(best_models['GaussianNB'].classes_)
```

Для нейросети вероятность ≥0,5 преобразуется в метку 1. В многоклассовой задаче нужно отдельно согласовать порядок `classes_`, столбцов вероятностей и меток метрик; в этой лабораторной такой задачи нет.

## 19. Где строится матрица ошибок? Что означает одна её ячейка?

Блок `confusion`:

```python
fig, axes = plt.subplots(3, 3, figsize=(13, 12))
for ax, (name, prediction) in zip(axes.flat, test_predictions.items()):
    ConfusionMatrixDisplay.from_predictions(
        y_test, prediction, labels=[0, 1], display_labels=['Osmancik', 'Cammeo'],
        cmap='Blues', colorbar=False, values_format='d', ax=ax,
        im_kw={'vmin': 0, 'vmax': y_test.value_counts().max()},
    )
    ax.set_title(name)
    ax.set_xlabel('Предсказанный класс')
    ax.set_ylabel('Истинный класс')
    ax.grid(False)
```

На рисунке расположены **девять матриц – для всех классификаторов и нейросети**. Цикл использует сохранённые прогнозы на одних 762 тестовых объектах. Порядок классов и цветовая шкала общие. Строки соответствуют истинным меткам, столбцы – прогнозам. Рассмотрим SVM, выбранный ранее по validation:

| Истинный / прогноз | Osmancik (0) | Cammeo (1) |
| :--- | ---: | ---: |
| Osmancik (0) | 404 | 32 |
| Cammeo (1) | 25 | 301 |

Например, **32** в верхней правой ячейке – зёрна Osmancik, ошибочно отнесённые к Cammeo. При положительном Cammeo это FP. Число 25 – FN, то есть пропущенные Cammeo.

## 20. Где считается ROC-AUC? Какие значения передаются?

```python
roc_auc_score(y_true, scores)
```

Функция находится внутри `calculate_metrics`. Для большинства классификаторов `scores` – вероятность класса 1, для SVM – значение решающей функции. **Готовые метки классов не используются**: они теряют информацию о ранжировании и сводят ROC к одной рабочей точке.

У SVM тестовый ROC-AUC = **0,977817**. Это качество ранжирования объектов, а не доля правильных меток при одном пороге.

## 21. Как получены оценки для ROC каждого классификатора? Что у SVM?

В блоках `tune` и `test`:

```python
if name == 'SVM':
    scores = model.decision_function(X_test)
else:
    scores = model.predict_proba(X_test)[:, 1]
```

У GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, дерева, LDA и kNN берётся вероятность Cammeo. У SVM – непрерывная оценка стороны и степени отделения от границы; это не вероятность. Для ROC её достаточно, поскольку меняется порог оценки.

У нейросети:

```python
scores = best_network.predict(X_test_scaled, verbose=0).ravel()
```

Sigmoid даёт значения от 0 до 1. `roc_curve(y_test, scores)` по этим оценкам строит FPR и TPR.

## 22. Где объединяются результаты? Как проверить одинаковый test?

Блок `test` создаёт общую таблицу:

```python
test_rows.append({'Модель': name, **calculate_metrics(y_test, prediction, scores)})
test_table = pd.DataFrame(test_rows)
test_table.to_csv(RESULTS / 'test_metrics.csv', index=False)
```

| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: |
| GaussianNB | 0,90157 | 0,89590 | 0,87117 | 0,88336 | 0,97062 |
| MultinomialNB | 0,89633 | 0,89457 | 0,85890 | 0,87637 | 0,96794 |
| ComplementNB | 0,89895 | 0,89274 | 0,86810 | 0,88025 | 0,96794 |
| BernoulliNB | 0,89764 | 0,86686 | 0,89877 | 0,88253 | 0,94616 |
| Decision Tree | 0,91601 | 0,89458 | 0,91104 | 0,90274 | 0,97257 |
| LDA | 0,93701 | 0,93711 | 0,91411 | 0,92547 | 0,97819 |
| SVM | 0,92520 | 0,90390 | 0,92331 | 0,91351 | 0,97782 |
| kNN | 0,91470 | 0,89666 | 0,90491 | 0,90076 | 0,97380 |
| TensorFlow MLP | 0,91470 | 0,89189 | 0,91104 | 0,90137 | 0,97597 |

Во всём цикле используются одни `X_test` и `y_test`; для нейросети преобразуются те же строки. Индексы сохранены в `split_assignment.csv`, а `test_predictions.csv` содержит `row_id`, истинные метки и прогнозы всех моделей. Проверить можно скриптом `verify_results.py`.

По validation выбран SVM. LDA имеет лучший наблюдаемый тестовый AUC и Accuracy, но это не повод менять правило выбора задним числом.

## 23. Какие пространства гиперпараметров исследованы?

В словаре `param_grids`:

| Метод | Параметры и значения |
| :--- | :--- |
| GaussianNB | `var_smoothing`: 1e-12, 1e-10, 1e-9, 1e-8, 1e-6, 1e-4, 1e-2 |
| MultinomialNB | `n_bins`: 3, 5, 8; `alpha`: 0,01; 0,1; 1; 10 |
| ComplementNB | `n_bins`: 3, 5, 8; `alpha`: 0,01; 0,1; 1; 10 |
| BernoulliNB | `threshold`: −0,5; 0; 0,5; `alpha`: 0,01; 0,1; 1; 10 |
| Decision Tree | `max_depth`: 2, 3, 5, 8, None; `min_samples_leaf`: 1, 5, 15; `criterion`: gini, entropy |
| LDA | svd: `tol` 1e-4, 1e-2; lsqr: `shrinkage` None, auto, 0,1; 0,5; 0,9 |
| SVM | linear: `C` 0,1; 1; 10; 100; rbf: те же C и `gamma` scale, 0,01; 0,1; 1 |
| kNN | `n_neighbors`: 3, 5, 11, 21, 41; `weights`: uniform, distance; `p`: 1, 2 |

Итого **120 сочетаний**. Для LDA и SVM используются списки словарей, чтобы не проверять неприменимые сочетания: например, shrinkage с svd. Полная сетка видна непосредственно в блоке `models`.

## 24. Где выполняется подбор параметров?

```python
search = GridSearchCV(
    model, param_grids[name], cv=cv,
    scoring='roc_auc', error_score='raise'
)
search.fit(X_train, y_train)
```

Это стандартный перебор всех сочетаний в GridSearchCV. Он максимизирует средний ROC-AUC по фолдам. `error_score='raise'` останавливает выполнение при ошибочной конфигурации, а не скрывает её за NaN.

По умолчанию `refit=True`, поэтому после поиска лучший Pipeline обучается на всех 2286 train-объектах. Validation и test в вызов `fit` не передаются.

## 25. Как устроена CV? Где заданы число разбиений и метрика выбора?

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
```

В GridSearchCV переданы `cv=cv` и `scoring='roc_auc'`. Для каждой конфигурации выполняются пять проверок: четыре части служат для обучения, пятая – для оценки. Стратификация сохраняет представительство сортов, перемешивание воспроизводимо.

ROC-AUC – основной критерий поиска. Остальные четыре метрики считаются на validation и test общей функцией. Преобразования внутри Pipeline обучаются заново в каждом фолде, поэтому проверочный фолд не участвует в настройке scaler или квантильных границ.

## 26. Как извлекаются лучшие параметры и модель? Где её проверяют на test?

```python
print(search.best_params_)
print(search.best_score_)
best_models[name] = search.best_estimator_
```

`best_params_` – выбранные настройки, `best_score_` – их средний CV ROC-AUC, `best_estimator_` – уже обученный Pipeline.

Например, для SVM выбраны **linear, C=100**, средний CV AUC – **0,979522**. Сначала его проверяют на validation; затем в блоке `test`:

```python
prediction = best_models['SVM'].predict(X_test)
scores = best_models['SVM'].decision_function(X_test)
```

Дополнительное обучение по test не выполняется.

## 27. Как формируется график влияния параметра?

В `parameter_plots` для текущего параметра:

```python
values = [str(params.get(parameter, '–')) for params in search.cv_results_['params']]
table = pd.DataFrame({'Значение': values, 'AUC': search.cv_results_['mean_test_score']})
curve = table.groupby('Значение', sort=False)['AUC'].max()
curve.plot(marker='o', ax=ax)
```

`cv_results_` содержит настройки и оценки всех сочетаний. Строки группируются по значению одного параметра, затем берётся **максимальный AUC по остальным настройкам**.

Поэтому график показывает лучший результат сетки при данном значении, а не изолированный причинный эффект единственного параметра. Например, улучшение дерева связано с ограничением глубины, но конкретная точка может также использовать другой критерий и размер листа. Для строгого исследования одного фактора остальные настройки надо зафиксировать.

## 28. Где создаётся нейросеть? Как используется Sequential?

Блок `neural_setup`, функция `create_network`:

```python
network = tf.keras.Sequential([tf.keras.Input(shape=(7,))])
for neurons in layers:
    network.add(tf.keras.layers.Dense(neurons, activation=activation))
network.add(tf.keras.layers.Dense(1, activation='sigmoid'))
```

Sequential объединяет слои в последовательную цепочку: выход одного поступает на вход следующего. Для рассматриваемой полносвязной сети этого достаточно; разветвлённой архитектуры нет. Список `layers` определяет скрытые слои.

## 29. Где входная размерность, слои и нейроны? Каков summary?

`Input(shape=(7,))` означает семь признаков одного объекта; размер батча сюда не входит. `layers=(16,)` создаёт один скрытый слой, `(16, 16)` – два.

В коде вызывается:

```python
create_network((16,), 'relu').summary()
```

Основные строки базовой архитектуры:

| Слой | Выход | Обучаемых параметров |
| :--- | :--- | ---: |
| Dense, 16 нейронов | (None, 16) | 7×16+16 = 128 |
| Dense, 1 нейрон | (None, 1) | 16×1+1 = 17 |
| Итого | – | **145** |

`None` – переменный размер батча. В модели два обучаемых Dense-слоя, из них один скрытый. Вход не имеет обучаемых весов. После обучения Keras может отдельно показывать состояния оптимизатора; они не меняют архитектуру сети.

## 30. Где активации? Почему один выход с sigmoid?

```python
tf.keras.layers.Dense(neurons, activation=activation)
tf.keras.layers.Dense(1, activation='sigmoid')
```

В скрытых слоях проверяются ReLU, tanh и sigmoid. Нелинейность нужна, иначе последовательность полносвязных слоёв свелась бы к одному аффинному преобразованию.

Два взаимоисключающих класса допускают **один выходной нейрон**: sigmoid даёт оценку вероятности Cammeo, а вероятность Osmancik равна `1−p`. Для K взаимоисключающих классов обычно применяют K выходов с softmax, но здесь такой архитектуры нет.

## 31. Где compile? Почему такие loss, optimizer и metrics?

```python
network.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

Метки имеют вид 0/1, выход – sigmoid, поэтому подходит бинарная кросс-энтропия. Adam обновляет веса по градиентам. Во время обучения Keras отслеживает Accuracy и loss, а при `validation_data` – также val_accuracy и val_loss.

Precision, Recall, F1 и ROC-AUC считаются отдельно общей функцией Scikit-Learn после получения прогнозов. Они не перечислены в `compile` и не являются автоматически появляющимися метриками History.

## 32. Где learning_rate, batch_size и epochs? Как изменить их?

В `experiments` заданы эпохи и батч. Например:

```python
('base', (16,), 'relu', 40, 32)
('epochs10', (16,), 'relu', 10, 32)
('batch64', (16,), 'relu', 40, 64)
```

Это отдельные элементы списка. Последние два поля передаются в `fit` как `epochs=epochs`, `batch_size=batch`.

Learning rate явно **не меняется**: строка `optimizer='adam'` использует начальное значение Adam **0,001**. Учебный пример явной настройки, не отдельный выполненный эксперимент отчёта:

```python
network.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='binary_crossentropy', metrics=['accuracy']
)
```

При train=2286 и batch=32 одна эпоха содержит 72 батча, последний неполный. Число эпох – проходы по train, не число отдельных обновлений весов.

## 33. Где fit нейросети? Какие данные и где история?

```python
history = network.fit(
    X_train_scaled, y_train.to_numpy(),
    validation_data=(X_val_scaled, y_val.to_numpy()),
    epochs=epochs, batch_size=batch,
    callbacks=[tensorboard], verbose=0,
)
```

Весам разрешено обучаться только по train. Validation используется для наблюдения и сравнения вариантов, но её градиенты не обновляют веса.

`history` – объект History. Словарь `history.history` содержит **loss, accuracy, val_loss, val_accuracy**. Он сохраняется в `histories[name]` и CSV каждого запуска в `results/neural_histories`. Каждая конфигурация проходит заданное число эпох полностью; EarlyStopping нет.

## 34. Где прогнозы нейросети на test? Как они превращаются в метки?

В блоке `test`:

```python
scores = best_network.predict(X_test_scaled, verbose=0).ravel()
prediction = (scores >= 0.5).astype(int)
test_rows.append({
    'Модель': 'TensorFlow MLP',
    **calculate_metrics(y_test, prediction, scores)
})
```

`predict` выдаёт матрицу **(762, 1)**, `ravel()` превращает её в вектор. Значение ≥0,5 даёт Cammeo, иначе Osmancik. ROC-AUC использует `scores`, остальные показатели – метки.

Выбрана сеть `epochs10`. На test: **Accuracy=0,914698**, **F1=0,901366**, **ROC-AUC=0,975974**. Это результаты данной сети, а не SVM, выбранного в общем сравнении.

## 35. Как сравниваются конфигурации сети? Где параметры и результаты?

Цикл перебирает список `experiments`:

```python
for name, layers, activation, epochs, batch in experiments:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    network = create_network(layers, activation)
```

После обучения в `nn_rows` записываются настройки и пять метрик validation. Таблица сохраняется как `neural_experiments.csv`. При улучшении AUC обновляются `best_network`, `best_run` и `best_nn_metrics`.

| Запуск | Скрытые слои | Активация | Эпохи | Батч | Validation AUC |
| :--- | :--- | :--- | ---: | ---: | ---: |
| base | 16 | relu | 40 | 32 | 0,979400 |
| epochs10 | 16 | relu | 10 | 32 | **0,980118** |
| epochs80 | 16 | relu | 80 | 32 | 0,979337 |
| batch16 | 16 | relu | 40 | 16 | 0,979196 |
| batch64 | 16 | relu | 40 | 64 | 0,979449 |
| width8 | 8 | relu | 40 | 32 | 0,979555 |
| width32 | 32 | relu | 40 | 32 | 0,979625 |
| depth2 | 16, 16 | relu | 40 | 32 | 0,979358 |
| depth3 | 16, 16, 16 | relu | 40 | 32 | 0,978612 |
| tanh | 16 | tanh | 40 | 32 | 0,979534 |
| sigmoid | 16 | sigmoid | 40 | 32 | 0,979998 |

В каждом дополнительном варианте относительно base меняется один фактор. Число параметров или эпох не гарантирует улучшения. Различия малы и получены при одном seed; статистическая устойчивость отдельно не доказана.

## 36. Как настроен TensorBoard? Как открыть два запуска и объяснить кривые?

Каталог LOGS создаётся в начале программы. Внутри цикла каждому эксперименту назначен подкаталог:

```python
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=str(LOGS / name), profile_batch=0
)
# В network.fit передаётся callbacks=[tensorboard].
```

Из корня проекта:

```bash
lab2/.venv/bin/tensorboard --logdir lab2/logs/tensorboard --port 6006
```

В PowerShell:

```powershell
.\lab2\.venv\Scripts\tensorboard.exe --logdir lab2/logs/tensorboard --port 6006
```

Откройте **http://localhost:6006**, выберите группу из `results/run_info.json` – для сохранённого расчёта `20261025-165807`. Сравните `epochs10/train`, `epochs10/validation`, `epochs80/train`, `epochs80/validation`. Для сравнения с CSV установите сглаживание в 0.

Теги `epoch_loss` и `epoch_accuracy` показаны отдельно для train и validation: это четыре нужные кривые. Числа последней эпохи:

| Запуск | Train loss | Val loss | Train accuracy | Val accuracy | Validation AUC |
| :--- | ---: | ---: | ---: | ---: | ---: |
| epochs10 | 0,185567 | 0,179985 | 0,928259 | 0,934383 | 0,980118 |
| epochs80 | 0,179278 | 0,181410 | 0,930446 | 0,937008 | 0,979337 |

При 80 эпохах train loss ниже, но итоговый validation loss выше, чем при 10. Минимум validation loss в длинном запуске был на **12-й эпохе**. По выбранному критерию AUC победил вариант 10 эпох: уменьшение обучающей ошибки само по себе не означает улучшения обобщения.

Val accuracy может быть выше train accuracy без утечки: выборки различаются, а обучающая метрика усредняется по батчам, пока веса меняются; валидация оценивается после эпохи. Сведения о событиях и команды позволяют выполнить демонстрацию на защите; отдельный запуск веб-интерфейса не является частью основного расчётного скрипта.

Основа ответов – текущий листинг, сохранённые модели и таблицы `results`. Стандартный способ работы callback описан в [документации TensorBoard](https://www.tensorflow.org/tensorboard/get_started).
