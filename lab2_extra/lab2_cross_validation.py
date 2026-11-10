# %% imports
import os
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.metrics import accuracy_score, log_loss, ConfusionMatrixDisplay
from IPython.display import display, Markdown

SEED = 42
EPOCHS = 40
BATCH_SIZE = 32
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.experimental.enable_op_determinism()
tf.keras.utils.set_random_seed(SEED)
sns.set_theme(style='whitegrid')

BASE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
if not (BASE / 'data/rice.csv').exists():
    BASE = BASE / 'lab2_extra'
RESULTS = BASE / 'results'
PLOTS = RESULTS / 'plots'
MODELS = RESULTS / 'models'
HISTORIES = RESULTS / 'histories'
for folder in [RESULTS, PLOTS, MODELS, HISTORIES]:
    folder.mkdir(parents=True, exist_ok=True)
LOGS = BASE / 'logs/tensorboard' / datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%fZ')

# %% data
df = pd.read_csv(BASE / 'data/rice.csv')
df.info()
print('Пропуски по столбцам:\n', df.isna().sum())
print('Полные дубликаты:', df.duplicated().sum())
X = df.drop(columns='Class')
y = df['Class'].map({'Osmancik': 0, 'Cammeo': 1})
if X.isna().any().any() or y.isna().any():
    raise ValueError('В данных появились пропуски или неизвестный класс: нужна предобработка.')
class_table = df['Class'].value_counts().rename('count').to_frame()
class_table['share'] = class_table['count'] / len(df)
display(class_table)
display(X.describe().round(3))
class_table.to_csv(RESULTS / 'class_distribution.csv')

# %% distributions
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, column in zip(axes.flat, X.columns):
    sns.histplot(data=df, x=column, hue='Class', bins=25, element='step',
                 stat='density', common_norm=False, ax=ax)
sns.countplot(data=df, x='Class', hue='Class', legend=False, ax=axes.flat[-1])
plt.tight_layout()
plt.savefig(PLOTS / '01_distributions.png', dpi=130)
plt.show()

# %% split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(cv.split(X_train, y_train))
split_table = pd.DataFrame({'split': 'test', 'fold': 0}, index=X.index)
split_table.loc[X_train.index, 'split'] = 'train'
fold_rows = []
for fold, (train_index, val_index) in enumerate(folds, start=1):
    split_table.loc[X_train.iloc[val_index].index, 'fold'] = fold
    fold_rows.append({
        'fold': fold, 'train_size': len(train_index), 'val_size': len(val_index),
        'train_Cammeo_share': y_train.iloc[train_index].mean(),
        'val_Cammeo_share': y_train.iloc[val_index].mean(),
    })
split_table.to_csv(RESULTS / 'split_assignment.csv', index_label='row_id')
print('Train:', X_train.shape, 'Test:', X_test.shape)
display(pd.DataFrame(fold_rows))

# %% network
# Название архитектуры: скрытые слои и доля dropout после каждого скрытого слоя.
architectures = {
    'dense16': ((16,), 0.0),
    'dense32_16': ((32, 16), 0.0),
    'dropout32_16': ((32, 16), 0.25),
}


def create_network(architecture, learning_rate):
    layers, dropout = architectures[architecture]
    network = tf.keras.Sequential([tf.keras.Input(shape=(X.shape[1],))])
    for neurons in layers:
        network.add(tf.keras.layers.Dense(neurons, activation='relu'))
        if dropout > 0:
            network.add(tf.keras.layers.Dropout(dropout))
    network.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss='binary_crossentropy', metrics=['accuracy'])
    return network


def calculate_metrics(y_true, scores):
    return {'accuracy': accuracy_score(y_true, scores >= 0.5),
            'loss': log_loss(y_true, scores, labels=[0, 1])}


create_network('dense16', 0.001).summary()

# %% baseline
# Одинаковый train для baseline и финальной модели; test не влияет на подбор.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype('float32')
X_test_scaled = scaler.transform(X_test).astype('float32')
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(SEED)
baseline = create_network('dense16', 0.001)
started = perf_counter()
baseline_history = baseline.fit(
    X_train_scaled, y_train.to_numpy(), epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir=str(LOGS / 'baseline'),
                                             write_graph=False, profile_batch=0)], verbose=0,
)
baseline_seconds = perf_counter() - started
baseline_scores = baseline.predict(X_test_scaled, verbose=0).ravel()
baseline_metrics = calculate_metrics(y_test, baseline_scores)
pd.DataFrame(baseline_history.history).to_csv(HISTORIES / 'baseline.csv', index_label='epoch0')
pd.DataFrame([{'model': 'Baseline', **baseline_metrics, 'fit_seconds': baseline_seconds}]).to_csv(
    RESULTS / 'baseline_metrics.csv', index=False
)
baseline.save(MODELS / 'baseline.keras')
joblib.dump(scaler, MODELS / 'scaler.joblib')
print('Baseline: dense16, Adam(lr=0.001), 40 эпох, batch=32')
print('Test:', baseline_metrics, 'Время обучения:', round(baseline_seconds, 2), 'с')

# %% grid
param_grid = {
    'learning_rate': [0.0001, 0.001, 0.01],
    'architecture': list(architectures),
}
configurations = list(ParameterGrid(param_grid))
print('Конфигураций:', len(configurations))
print('Обучений в CV:', len(configurations) * len(folds))
display(pd.DataFrame(configurations))

# %% cross_validation
# Полный Grid Search: каждое сочетание параметров обучается на всех пяти фолдах.
cv_rows, histories = [], {}
oof = pd.DataFrame({'y_true': y_train, 'fold': split_table.loc[X_train.index, 'fold']})
for params in configurations:
    architecture = params['architecture']
    learning_rate = params['learning_rate']
    run = f'{architecture}_lr{learning_rate:g}'
    oof[run] = np.nan
    for fold, (train_index, val_index) in enumerate(folds, start=1):
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
        # Scaler заново обучается только на обучающей части текущего фолда.
        fold_scaler = StandardScaler()
        X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train).astype('float32')
        X_fold_val_scaled = fold_scaler.transform(X_fold_val).astype('float32')
        tf.keras.backend.clear_session()
        # Для разных конфигураций seed на одном фолде одинаков, между фолдами различен.
        tf.keras.utils.set_random_seed(SEED + fold)
        network = create_network(architecture, learning_rate)
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=str(LOGS / run / f'fold_{fold}'), write_graph=False, profile_batch=0,
        )
        started = perf_counter()
        history = network.fit(
            X_fold_train_scaled, y_fold_train.to_numpy(),
            validation_data=(X_fold_val_scaled, y_fold_val.to_numpy()),
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[tensorboard], verbose=0,
        )
        fit_seconds = perf_counter() - started
        train_scores = network.predict(X_fold_train_scaled, verbose=0).ravel()
        val_scores = network.predict(X_fold_val_scaled, verbose=0).ravel()
        train_metrics = calculate_metrics(y_fold_train, train_scores)
        val_metrics = calculate_metrics(y_fold_val, val_scores)
        oof.loc[X_fold_val.index, run] = val_scores
        history_table = pd.DataFrame(history.history)
        history_table.index = np.arange(1, EPOCHS + 1)
        history_table.index.name = 'epoch'
        histories[(run, fold)] = history_table
        history_table.to_csv(HISTORIES / f'{run}_fold{fold}.csv')
        cv_rows.append({
            'run': run, **params, 'fold': fold, 'parameters': network.count_params(),
            'train_size': len(train_index), 'val_size': len(val_index),
            'train_accuracy': train_metrics['accuracy'], 'train_loss': train_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'], 'val_loss': val_metrics['loss'],
            'fit_seconds': fit_seconds,
        })
        print(f'{run}, fold {fold}/5: accuracy={val_metrics["accuracy"]:.5f}, '
              f'loss={val_metrics["loss"]:.5f}, {fit_seconds:.1f} с', flush=True)
        # Промежуточное сохранение: уже рассчитанные фолды остаются на диске.
        pd.DataFrame(cv_rows).to_csv(RESULTS / 'cv_folds.csv', index=False)
        oof.to_csv(RESULTS / 'oof_predictions.csv', index_label='row_id')
cv_table = pd.DataFrame(cv_rows)

# %% comparison
summary = cv_table.groupby(['run', 'architecture', 'learning_rate'], as_index=False).agg(
    mean_accuracy=('val_accuracy', 'mean'), std_accuracy=('val_accuracy', 'std'),
    mean_loss=('val_loss', 'mean'), std_loss=('val_loss', 'std'),
    mean_train_accuracy=('train_accuracy', 'mean'), mean_train_loss=('train_loss', 'mean'),
    mean_fit_seconds=('fit_seconds', 'mean'), total_fit_seconds=('fit_seconds', 'sum'),
    parameters=('parameters', 'first'),
)
summary['accuracy_gap'] = summary['mean_train_accuracy'] - summary['mean_accuracy']
# При точном равенстве accuracy: меньше разброс, loss, число параметров, затем имя.
summary = summary.sort_values(
    ['mean_accuracy', 'std_accuracy', 'mean_loss', 'parameters', 'run'],
    ascending=[False, True, True, True, True],
).reset_index(drop=True)
summary.to_csv(RESULTS / 'cv_results.csv', index=False)
best = summary.iloc[0]
best_run = best['run']
display(summary.round(5))
print('Выбрано только по CV:', best_run)
print(f'CV accuracy = {best["mean_accuracy"]:.5f} ± {best["std_accuracy"]:.5f} (std)')

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
positions = np.arange(len(summary))
for ax, column, error, title in [
    (axes[0, 0], 'mean_accuracy', 'std_accuracy', 'CV accuracy: среднее ± std'),
    (axes[0, 1], 'mean_loss', 'std_loss', 'CV loss: среднее ± std'),
    (axes[1, 0], 'mean_fit_seconds', None, 'Среднее время fit на фолд, с'),
    (axes[1, 1], 'std_accuracy', None, 'Разброс accuracy между фолдами'),
]:
    ax.bar(positions, summary[column], yerr=summary[error] if error else None, capsize=4)
    ax.set_xticks(positions, summary['run'], rotation=65, ha='right')
    ax.set_title(title)
plt.tight_layout()
plt.savefig(PLOTS / '02_configurations.png', dpi=130)
plt.show()

# %% learning_curves
# Среднее по пяти фолдам и полоса ±1 std; для dropout train измеряется с активным dropout.
for metric in ['loss', 'accuracy']:
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)
    for ax, params in zip(axes.flat, configurations):
        run = f'{params["architecture"]}_lr{params["learning_rate"]:g}'
        for prefix, label, color in [('', 'Train', 'tab:blue'), ('val_', 'Validation', 'tab:orange')]:
            curves = np.array([histories[(run, fold)][prefix + metric] for fold in range(1, 6)])
            mean, std = curves.mean(axis=0), curves.std(axis=0, ddof=1)
            epochs = np.arange(1, EPOCHS + 1)
            ax.plot(epochs, mean, label=label, color=color)
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=color)
        ax.set(title=run, xlabel='Эпоха', ylabel=metric)
        ax.legend()
    plt.suptitle(f'{metric}: среднее и разброс на пяти фолдах')
    plt.tight_layout()
    plt.savefig(PLOTS / f'03_learning_{metric}.png', dpi=130)
    plt.show()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for fold in range(1, 6):
    history = histories[(best_run, fold)]
    for ax, metric in zip(axes, ['loss', 'accuracy']):
        ax.plot(history.index, history['val_' + metric], label=f'Fold {fold}')
        ax.set(title=f'Validation {metric}', xlabel='Эпоха')
        ax.legend()
plt.suptitle(f'Лучшая конфигурация: {best_run}')
plt.tight_layout()
plt.savefig(PLOTS / '04_best_folds.png', dpi=130)
plt.show()

# %% parameter_effects
heatmap = summary.pivot(index='architecture', columns='learning_rate', values='mean_accuracy')
plt.figure(figsize=(9, 4))
sns.heatmap(heatmap, annot=True, fmt='.5f', cmap='YlGnBu')
plt.title('Средняя CV accuracy: архитектура и скорость обучения')
plt.tight_layout()
plt.savefig(PLOTS / '05_parameter_effects.png', dpi=130)
plt.show()

effects = []
for factor in ['learning_rate', 'architecture']:
    means = summary.groupby(factor)['mean_accuracy'].mean()
    for value, mean in means.items():
        effects.append({'factor': factor, 'value': str(value), 'mean_accuracy': mean})
effects = pd.DataFrame(effects)
effects.to_csv(RESULTS / 'parameter_effects.csv', index=False)
display(effects)
lr_range = summary.groupby('learning_rate')['mean_accuracy'].mean().pipe(lambda s: s.max() - s.min())
architecture_range = summary.groupby('architecture')['mean_accuracy'].mean().pipe(lambda s: s.max() - s.min())
print(f'Размах средних по learning_rate: {100 * lr_range:.3f} п.п.')
print(f'Размах средних по архитектуре: {100 * architecture_range:.3f} п.п.')

# %% final_training
# Архитектура, lr и число эпох уже зафиксированы. Test в fit не передаётся.
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(SEED)
final_network = create_network(best['architecture'], float(best['learning_rate']))
started = perf_counter()
final_history = final_network.fit(
    X_train_scaled, y_train.to_numpy(), epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir=str(LOGS / 'final'),
                                             write_graph=False, profile_batch=0)], verbose=0,
)
final_seconds = perf_counter() - started
final_scores = final_network.predict(X_test_scaled, verbose=0).ravel()
final_metrics = calculate_metrics(y_test, final_scores)
final_network.save(MODELS / 'best.keras')
pd.DataFrame(final_history.history).to_csv(HISTORIES / 'final.csv', index_label='epoch0')
test_table = pd.DataFrame([
    {'model': 'Baseline', 'configuration': 'dense16_lr0.001', **baseline_metrics,
     'fit_seconds': baseline_seconds},
    {'model': 'Tuned MLP', 'configuration': best_run, **final_metrics,
     'fit_seconds': final_seconds},
])
test_table.to_csv(RESULTS / 'test_metrics.csv', index=False)
predictions = pd.DataFrame({
    'row_id': X_test.index, 'y_true': y_test.to_numpy(),
    'baseline_score': baseline_scores, 'baseline_prediction': (baseline_scores >= 0.5).astype(int),
    'tuned_score': final_scores, 'tuned_prediction': (final_scores >= 0.5).astype(int),
})
predictions.to_csv(RESULTS / 'test_predictions.csv', index=False)
display(test_table.round(5))
print(f'Изменение test accuracy: {100 * (final_metrics["accuracy"] - baseline_metrics["accuracy"]):+.3f} п.п.')

# %% final_plots
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, name, scores in zip(axes, ['Baseline', 'Tuned MLP'], [baseline_scores, final_scores]):
    ConfusionMatrixDisplay.from_predictions(y_test, scores >= 0.5,
        display_labels=['Osmancik', 'Cammeo'], cmap='Blues', colorbar=False, ax=ax)
    ax.set_title(name)
    ax.set_xlabel('Предсказанный класс')
    ax.set_ylabel('Истинный класс')
plt.tight_layout()
plt.savefig(PLOTS / '06_confusion.png', dpi=130)
plt.show()

# %% conclusion
baseline_cv = summary.loc[summary['run'] == 'dense16_lr0.001'].iloc[0]
delta = 100 * (final_metrics['accuracy'] - baseline_metrics['accuracy'])
stronger = 'скорость обучения' if lr_range > architecture_range else 'архитектура'
best_history = pd.concat([histories[(best_run, fold)] for fold in range(1, 6)]).groupby('epoch').mean()
minimum_loss_epoch = int(best_history['val_loss'].idxmin())
conclusion = f'''## Итоговые результаты

- Данные: {len(df)} объектов, {X.shape[1]} числовых признаков, два класса.
  Train — {len(X_train)}, test — {len(X_test)}. Проверены 9 конфигураций × 5 фолдов.
- Baseline: **dense16, lr=0.001**; test accuracy **{baseline_metrics['accuracy']:.5f}**,
  test loss **{baseline_metrics['loss']:.5f}**.
- Лучшая конфигурация по CV: **{best_run}**; средняя accuracy
  **{best['mean_accuracy']:.5f} ± {best['std_accuracy']:.5f}** (выборочное std, 5 фолдов),
  средний loss **{best['mean_loss']:.5f}**, среднее время fit **{best['mean_fit_seconds']:.2f} с/фолд**.
- Для исходной конфигурации CV accuracy равна **{baseline_cv['mean_accuracy']:.5f}**;
  прирост после подбора на CV — **{100 * (best['mean_accuracy'] - baseline_cv['mean_accuracy']):+.3f} п.п.**
- После обучения лучшей конфигурации на всём train: test accuracy
  **{final_metrics['accuracy']:.5f}**, test loss **{final_metrics['loss']:.5f}**.
  Изменение test accuracy относительно baseline: **{delta:+.3f} п.п.**
- В выбранной сетке сильнее повлияла **{stronger}**: размах средних accuracy
  по learning rate — **{100 * lr_range:.3f} п.п.**, по архитектуре — **{100 * architecture_range:.3f} п.п.**
  Это описательное сравнение усреднений по остальным настройкам; взаимодействия видны на тепловой карте.
- Разрыв между train и validation accuracy у победителя: **{100 * best['accuracy_gap']:.3f} п.п.**
  Минимум усреднённого validation loss достигнут на эпохе **{minimum_loss_epoch}**;
  оценивались веса последней, {EPOCHS}-й эпохи. Ранний минимум и последующий рост loss
  помогают заметить переобучение; низкое качество на обеих частях — недообучение.

Пять фолдов оценивают зависимость от разбиения, но их std не является доверительным интервалом.
CV-оценка выбранного победителя может быть оптимистичной из-за перебора параметров.
Итоговая проверка выполнена на отдельном test с тем же порогом 0.5.
Результаты одного seed и одного test не доказывают универсального преимущества настройки.
'''
display(Markdown(conclusion))
(RESULTS / 'conclusion.md').write_text(conclusion, encoding='utf-8')
run_info = {
    'seed': SEED, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'cv_folds': 5,
    'configurations': len(configurations), 'cv_fits': len(cv_table),
    'train_size': len(X_train), 'test_size': len(X_test),
    'best_run': best_run, 'best_architecture': best['architecture'],
    'best_learning_rate': float(best['learning_rate']),
    'selection': 'mean_accuracy desc, std_accuracy asc, mean_loss asc, parameters asc, run asc',
    'tensorboard_dir': str(LOGS.relative_to(BASE)),
    'tensorflow': tf.__version__,
}
(RESULTS / 'run_info.json').write_text(json.dumps(run_info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('Каталог TensorBoard:', LOGS)
