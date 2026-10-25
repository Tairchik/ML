# Лабораторные работы по машинному обучению

## Лабораторная № 2: классификация и нейронные сети

- [Отчёт с выполненными ячейками и графиками](lab2/lab2_classification.ipynb).
- [Полный листинг](lab2/lab2_classification.py), [зависимости](lab2/req.txt).
- [Запуск, методика сравнения и TensorBoard](lab2/README.md).
- [Итоговые метрики](lab2/results/test_metrics.csv) и [нейросетевые эксперименты](lab2/results/neural_experiments.csv).

Датасет: [Rice (Cammeo and Osmancik), UCI](https://archive.ics.uci.edu/dataset/545/rice%2Bcammeo%2Band%2Bosmancik). Сравниваются четыре варианта наивного Байеса, дерево решений, LDA, SVM, kNN и нейросеть TensorFlow. Параметры выбираются без тестовой выборки, настоящие логи TensorBoard сохранены в `lab2/logs/tensorboard`.

## Лабораторная № 1: регрессия и PCA

- [Отчет с выполненными ячейками и графиками](lab1/lab1_regression.ipynb).
- [Полный листинг кода](lab1/lab1_regression.py), также включен в приложение ноутбука.
- [Зависимости](lab1/req.txt).

Датасет [Computer Hardware, UCI](https://archive.ics.uci.edu/dataset/29/computer+hardware) сохранен локально в `lab1/machine.data`; описание полей — в `lab1/machine.names`. Для расчета метрик доступ к интернету не нужен.

## Запуск в Windows / PowerShell

Проверено на Python 3.12. Из корня проекта:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r lab1/req.txt
.\.venv\Scripts\python.exe -m jupyter lab lab1/lab1_regression.ipynb
```

В Jupyter выберите **Restart Kernel and Run All Cells**, чтобы выполнить отчет с чистого состояния. При открытии в VS Code выберите интерпретатор `.venv\Scripts\python.exe` как ядро ноутбука. Загружать данные можно при рабочей папке в корне проекта или в `lab1`.

Выполнение без открытия интерфейса:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=120 lab1/lab1_regression.ipynb
```

Можно запустить полный листинг отдельно:

```powershell
.\.venv\Scripts\python.exe lab1/lab1_regression.py
```

В этом случае графики открываются последовательно; закройте окно текущего графика, чтобы продолжить выполнение.

## Что исследуется

Линейная регрессия, Ridge и Lasso предсказывают `PRP` по шести числовым характеристикам. Используются разбиение 80/20, пятикратная кросс-валидация, подбор `alpha`, стандартизация внутри `Pipeline` и сравнение RMSE, R², MAPE до и после PCA. Порог PCA 90% оставляет пять компонент на полной тренировочной выборке.

После каждой вычислительной ячейки ноутбука приведено содержательное Markdown-пояснение. Если менять данные, параметры или разбиение, численные выводы в отчете и ответах к защите также потребуется обновить.
