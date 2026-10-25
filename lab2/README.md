# Лабораторная работа 2: классификация и нейронные сети

- [Ноутбук с результатами и пояснениями](lab2_classification.ipynb).
- [Полный код](lab2_classification.py): две небольшие функции, последовательные шаги.
- [Зависимости](req.txt).

Данные: [Rice (Cammeo and Osmancik), UCI](https://archive.ics.uci.edu/dataset/545/rice%2Bcammeo%2Band%2Bosmancik). Программа читает готовый [data/rice.csv](data/rice.csv). Исходный ARFF, описание происхождения и условия цитирования также сохранены в `data`. Для расчётов интернет не нужен.

## Как читать код

Блоки идут сверху вниз и разделены комментариями `# %%`:

1. Импорт библиотек и seed.
2. Загрузка, проверка данных и графики.
3. Разделение train/validation/test.
4. Общая функция расчёта пяти метрик.
5. Словарь моделей и отдельный словарь сеток параметров.
6. Обычный цикл `GridSearchCV` по восьми классификаторам.
7. Функция создания нейросети и цикл по 11 понятным кортежам настроек.
8. Выбор по validation, прогнозы на test и сравнение.

В основном листинге нет проверки хешей, генерации отчёта, HParams или собственного конвейера TensorFlow `tf.data`. Используются стандартные `fit`, `predict`, `Pipeline` и callback TensorBoard. Подробные объяснения находятся в Markdown-ячейках ноутбука.

## Методика

Положительный класс – **Cammeo (1)**, отрицательный – **Osmancik (0)**. Accuracy учитывает оба класса, Precision/Recall/F1 относятся к Cammeo. ROC-AUC считается по непрерывным оценкам класса 1.

| Набор | Объектов | Назначение |
| :--- | ---: | :--- |
| Train | 2286 (60%) | Обучение и пятикратная CV |
| Validation | 762 (20%) | Выбор нейросети и общего подхода |
| Test | 762 (20%) | Итоговое сравнение |

Все итоговые модели обучены на одинаковом train. После выбора дополнительное переобучение не выполняется. Seed равен 42, разбиения стратифицированы. Scaler и квантильные границы обучаются внутри CV-конвейеров; scaler нейросети обучается только на train. Test не используется для подбора параметров.

Сохранены все четыре варианта наивного Байеса, Decision Tree, LDA, SVM и kNN. MultinomialNB/ComplementNB получают индикаторы квантильных интервалов, BernoulliNB – бинаризованные признаки. Подбор проверяет 120 классических конфигураций. У нейросети меняются эпохи, батч, ширина, глубина и активация.

## Запуск

Проверено на Python 3.12, Linux/WSL, CPU. Из корня проекта:

```bash
python3 -m venv lab2/.venv
lab2/.venv/bin/python -m pip install -r lab2/req.txt
lab2/.venv/bin/python lab2/lab2_classification.py
```

В PowerShell путь к интерпретатору – `.\lab2\.venv\Scripts\python.exe`.

В VS Code или Jupyter выберите ядро `lab2/.venv` и выполните **Restart Kernel and Run All Cells**. Для выполнения ноутбука из терминала:

```bash
lab2/.venv/bin/python lab2/run_notebook.py
```

Этот небольшой вспомогательный скрипт переносит код из `.py` в соответствующие ячейки и запускает чистое ядро. Пояснения редактируются непосредственно в ноутбуке. Если меняются данные или настройки, численные выводы также нужно обновить.

## TensorBoard

```bash
lab2/.venv/bin/tensorboard --logdir lab2/logs/tensorboard --port 6006
```

Откройте **http://localhost:6006**. Текущий каталог указан в [results/run_info.json](results/run_info.json). Во вкладке **Time Series / Scalars** выберите `epoch_loss` и `epoch_accuracy` отдельно у `train` и `validation`: это loss, accuracy, validation loss и validation accuracy. Каждый нейросетевой эксперимент имеет отдельный подкаталог.

## Результаты

- [classical_tuning.csv](results/classical_tuning.csv) – параметры и CV AUC до/после настройки.
- [classical_cv](results/classical_cv) – все сочетания параметров и оценки по фолдам.
- [neural_experiments.csv](results/neural_experiments.csv) – 11 конфигураций и пять метрик validation.
- [validation_metrics.csv](results/validation_metrics.csv) – общее сравнение на validation.
- [test_metrics.csv](results/test_metrics.csv) – пять итоговых метрик девяти моделей.
- [test_predictions.csv](results/test_predictions.csv) – сохранённые метки и оценки.
- [plots](results/plots) – все графики; [models](results/models) – обученные модели.

На [общей столбчатой диаграмме](results/plots/07_metrics.png) для каждой модели показаны пять метрик: Accuracy, Precision, Recall, F1-Score и AUC-ROC. AUC-ROC соответствует столбцу `ROC-AUC` в таблицах; фиолетовые столбцы показывают качество ранжирования по непрерывным оценкам.

[Матрицы ошибок](results/plots/08_confusion.png) показаны сеткой 3×3 для всех девяти моделей, включая нейросеть. Строки – истинные классы, столбцы – прогнозы; порядок классов Osmancik, Cammeo и цветовая шкала одинаковы для всех матриц.

На графиках настройки при каждом значении параметра показан максимум по остальным настройкам сетки. Это сравнение лучших сочетаний, а не изолированный эффект одного параметра.

Обученные модели в `results/models/` сохраняются локально и исключены из Git. После клонирования они создаются запуском основного скрипта или ноутбука. Логи TensorBoard, графики и таблицы результатов сохраняются в репозитории.

Проверка результатов без повторного обучения, когда модели уже созданы:

```bash
lab2/.venv/bin/python lab2/verify_results.py
```
