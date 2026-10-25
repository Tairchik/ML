# %% imports
import os
from pathlib import Path
from datetime import datetime
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, Binarizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay

SEED = 42
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.experimental.enable_op_determinism()
tf.keras.utils.set_random_seed(SEED)
sns.set_theme(style='whitegrid')

BASE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
if not (BASE / 'data/rice.csv').exists():
    BASE = BASE / 'lab2'
RESULTS = BASE / 'results'
PLOTS = RESULTS / 'plots'
MODELS = RESULTS / 'models'
for folder in [RESULTS, PLOTS, MODELS, RESULTS / 'classical_cv', RESULTS / 'neural_histories']:
    folder.mkdir(parents=True, exist_ok=True)
LOGS = BASE / 'logs/tensorboard' / datetime.now().strftime('%Y%m%d-%H%M%S')

# %% data
df = pd.read_csv(BASE / 'data/rice.csv')
df.info()
print('Пропуски:', df.isna().sum().sum())
print('Дубликаты:', df.duplicated().sum())
df = df.dropna()
X = df.drop(columns='Class')
y = df['Class'].map({'Osmancik': 0, 'Cammeo': 1})
print(df['Class'].value_counts())
print(X.describe().round(2))

# %% distributions
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, column in zip(axes.flat, X.columns):
    sns.histplot(data=df, x=column, hue='Class', bins=25, element='step',
                 stat='density', common_norm=False, ax=ax)
sns.countplot(data=df, x='Class', hue='Class', legend=False, ax=axes.flat[-1])
plt.tight_layout()
plt.savefig(PLOTS / '01_distributions.png')
plt.show()

# %% correlations
plt.figure(figsize=(9, 6))
sns.heatmap(X.corr(), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Корреляции признаков')
plt.tight_layout()
plt.savefig(PLOTS / '02_correlations.png')
plt.show()

# %% split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=SEED, stratify=y_train
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print('Train:', X_train.shape, 'Validation:', X_val.shape, 'Test:', X_test.shape)
splits = pd.Series(index=X.index, dtype='str', name='split')
for name, part in [('train', X_train), ('validation', X_val), ('test', X_test)]:
    splits.loc[part.index] = name
splits.to_csv(RESULTS / 'split_assignment.csv', index_label='row_id')

# %% metrics
def calculate_metrics(y_true, y_pred, scores):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, scores),
    }

# %% models
models = {
    'GaussianNB': Pipeline([('scale', StandardScaler()), ('model', GaussianNB())]),
    'MultinomialNB': Pipeline([
        ('bins', KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile', subsample=None)),
        ('model', MultinomialNB()),
    ]),
    'ComplementNB': Pipeline([
        ('bins', KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile', subsample=None)),
        ('model', ComplementNB()),
    ]),
    'BernoulliNB': Pipeline([
        ('scale', StandardScaler()), ('binary', Binarizer()),
        ('model', BernoulliNB(binarize=None)),
    ]),
    'Decision Tree': Pipeline([('model', DecisionTreeClassifier(random_state=SEED))]),
    'LDA': Pipeline([('scale', StandardScaler()), ('model', LinearDiscriminantAnalysis())]),
    'SVM': Pipeline([('scale', StandardScaler()), ('model', SVC(random_state=SEED))]),
    'kNN': Pipeline([('scale', StandardScaler()), ('model', KNeighborsClassifier())]),
}
param_grids = {
    'GaussianNB': {'model__var_smoothing': [1e-12, 1e-10, 1e-9, 1e-8, 1e-6, 1e-4, 1e-2]},
    'MultinomialNB': {'bins__n_bins': [3, 5, 8], 'model__alpha': [0.01, 0.1, 1, 10]},
    'ComplementNB': {'bins__n_bins': [3, 5, 8], 'model__alpha': [0.01, 0.1, 1, 10]},
    'BernoulliNB': {'binary__threshold': [-0.5, 0, 0.5], 'model__alpha': [0.01, 0.1, 1, 10]},
    'Decision Tree': {'model__max_depth': [2, 3, 5, 8, None],
                      'model__min_samples_leaf': [1, 5, 15], 'model__criterion': ['gini', 'entropy']},
    'LDA': [{'model__solver': ['svd'], 'model__tol': [1e-4, 1e-2]},
            {'model__solver': ['lsqr'], 'model__shrinkage': [None, 'auto', 0.1, 0.5, 0.9]}],
    'SVM': [{'model__kernel': ['rbf'], 'model__C': [0.1, 1, 10, 100],
             'model__gamma': ['scale', 0.01, 0.1, 1]},
            {'model__kernel': ['linear'], 'model__C': [0.1, 1, 10, 100]}],
    'kNN': {'model__n_neighbors': [3, 5, 11, 21, 41],
            'model__weights': ['uniform', 'distance'], 'model__p': [1, 2]},
}

# %% tune
best_models, searches = {}, {}
tuning_rows, validation_rows = [], []
for name, model in models.items():
    baseline_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc').mean()
    search = GridSearchCV(model, param_grids[name], cv=cv, scoring='roc_auc', error_score='raise')
    search.fit(X_train, y_train)
    best_models[name] = search.best_estimator_
    searches[name] = search
    table = pd.DataFrame(search.cv_results_)
    table.to_csv(RESULTS / 'classical_cv' / f'{name}.csv', index=False)
    tuning_rows.append({'Модель': name, 'CV до': baseline_auc, 'CV после': search.best_score_,
                        'CV std': table.loc[search.best_index_, 'std_test_score'],
                        'Параметры': str(search.best_params_)})
    prediction = search.predict(X_val)
    if name == 'SVM':
        scores = search.decision_function(X_val)
    else:
        scores = search.predict_proba(X_val)[:, 1]
    validation_rows.append({'Модель': name, **calculate_metrics(y_val, prediction, scores)})
    joblib.dump(search.best_estimator_, MODELS / f'{name}.joblib')
    print(name, search.best_params_, 'CV AUC:', round(search.best_score_, 5), flush=True)

tuning_table = pd.DataFrame(tuning_rows)
tuning_table.to_csv(RESULTS / 'classical_tuning.csv', index=False)
print(tuning_table.to_string(index=False))

# %% parameter_plots
parameters = ['model__var_smoothing', 'model__alpha', 'model__alpha', 'model__alpha',
              'model__max_depth', 'model__shrinkage', 'model__C', 'model__n_neighbors']
fig, axes = plt.subplots(2, 4, figsize=(17, 8))
for ax, (name, search), parameter in zip(axes.flat, searches.items(), parameters):
    values = [str(params.get(parameter, '–')) for params in search.cv_results_['params']]
    table = pd.DataFrame({'Значение': values, 'AUC': search.cv_results_['mean_test_score']})
    curve = table.groupby('Значение', sort=False)['AUC'].max()
    curve.plot(marker='o', ax=ax)
    ax.set_title(name)
    ax.set_xlabel(parameter.replace('model__', ''))
    ax.set_ylabel('Лучший CV AUC в сетке')
    ax.tick_params(axis='x', rotation=40)
plt.tight_layout()
plt.savefig(PLOTS / '03_parameter_effects.png')
plt.show()

# %% neural_setup
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype('float32')
X_val_scaled = scaler.transform(X_val).astype('float32')
X_test_scaled = scaler.transform(X_test).astype('float32')

def create_network(layers, activation):
    network = tf.keras.Sequential([tf.keras.Input(shape=(7,))])
    for neurons in layers:
        network.add(tf.keras.layers.Dense(neurons, activation=activation))
    network.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return network

# Название, нейроны в скрытых слоях, активация, эпохи, размер батча.
experiments = [
    ('base', (16,), 'relu', 40, 32),
    ('epochs10', (16,), 'relu', 10, 32),
    ('epochs80', (16,), 'relu', 80, 32),
    ('batch16', (16,), 'relu', 40, 16),
    ('batch64', (16,), 'relu', 40, 64),
    ('width8', (8,), 'relu', 40, 32),
    ('width32', (32,), 'relu', 40, 32),
    ('depth2', (16, 16), 'relu', 40, 32),
    ('depth3', (16, 16, 16), 'relu', 40, 32),
    ('tanh', (16,), 'tanh', 40, 32),
    ('sigmoid', (16,), 'sigmoid', 40, 32),
]
create_network((16,), 'relu').summary()

# %% neural_train
nn_rows, histories = [], {}
best_auc = -1
for name, layers, activation, epochs, batch in experiments:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    network = create_network(layers, activation)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=str(LOGS / name), profile_batch=0)
    history = network.fit(
        X_train_scaled, y_train.to_numpy(), validation_data=(X_val_scaled, y_val.to_numpy()),
        epochs=epochs, batch_size=batch, callbacks=[tensorboard], verbose=0,
    )
    scores = network.predict(X_val_scaled, verbose=0).ravel()
    metrics = calculate_metrics(y_val, (scores >= 0.5).astype(int), scores)
    nn_rows.append({'Run': name, 'Слои': str(layers), 'Активация': activation,
                    'Эпохи': epochs, 'Батч': batch, **metrics})
    histories[name] = history.history
    pd.DataFrame(history.history).to_csv(RESULTS / 'neural_histories' / f'{name}.csv', index=False)
    if metrics['ROC-AUC'] > best_auc:
        best_auc = metrics['ROC-AUC']
        best_network, best_run = network, name
        best_nn_metrics = metrics
    print(name, 'Validation AUC:', round(metrics['ROC-AUC'], 5), flush=True)

nn_table = pd.DataFrame(nn_rows)
nn_table.to_csv(RESULTS / 'neural_experiments.csv', index=False)
best_network.save(MODELS / 'tensorflow_mlp.keras')
joblib.dump(scaler, MODELS / 'tensorflow_scaler.joblib')
print(nn_table.round(5).to_string(index=False))
print('Лучшая конфигурация:', best_run)

# %% learning_plots
history = histories[best_run]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metric in zip(axes, ['loss', 'accuracy']):
    ax.plot(range(1, len(history[metric]) + 1), history[metric], label='Train')
    ax.plot(range(1, len(history[metric]) + 1), history['val_' + metric], label='Validation')
    ax.set(title=metric, xlabel='Эпоха')
    ax.legend()
plt.suptitle(f'Обучение нейросети: {best_run}')
plt.tight_layout()
plt.savefig(PLOTS / '04_learning_curves.png')
plt.show()

plt.figure(figsize=(11, 4))
plt.bar(nn_table['Run'], nn_table['ROC-AUC'])
plt.ylabel('Validation ROC-AUC')
plt.ylim(0, 1)
plt.xticks(rotation=35)
plt.tight_layout()
plt.savefig(PLOTS / '05_neural_experiments.png')
plt.show()

# %% selection
validation_rows.append({'Модель': 'TensorFlow MLP', **best_nn_metrics})
validation_table = pd.DataFrame(validation_rows)
best_method = validation_table.loc[validation_table['ROC-AUC'].idxmax(), 'Модель']
validation_table.to_csv(RESULTS / 'validation_metrics.csv', index=False)
print(validation_table.round(5).to_string(index=False))
print('Выбор по validation:', best_method)

# %% test
test_rows, test_scores, test_predictions = [], {}, {}
for name, model in best_models.items():
    prediction = model.predict(X_test)
    if name == 'SVM':
        scores = model.decision_function(X_test)
    else:
        scores = model.predict_proba(X_test)[:, 1]
    test_rows.append({'Модель': name, **calculate_metrics(y_test, prediction, scores)})
    test_scores[name], test_predictions[name] = scores, prediction

scores = best_network.predict(X_test_scaled, verbose=0).ravel()
prediction = (scores >= 0.5).astype(int)
test_rows.append({'Модель': 'TensorFlow MLP', **calculate_metrics(y_test, prediction, scores)})
test_scores['TensorFlow MLP'], test_predictions['TensorFlow MLP'] = scores, prediction
test_table = pd.DataFrame(test_rows)
test_table.to_csv(RESULTS / 'test_metrics.csv', index=False)
print(test_table.round(5).to_string(index=False))

predictions_table = pd.DataFrame({'row_id': y_test.index, 'y_true': y_test.to_numpy()})
for name in test_predictions:
    predictions_table[name + '__prediction'] = test_predictions[name]
    predictions_table[name + '__score'] = test_scores[name]
predictions_table.to_csv(RESULTS / 'test_predictions.csv', index=False)

# %% comparison_plots
plt.figure(figsize=(9, 7))
for name, scores in test_scores.items():
    fpr, tpr, thresholds = roc_curve(y_test, scores)
    plt.plot(fpr, tpr, label=f'{name}: AUC-ROC = {roc_auc_score(y_test, scores):.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые на test')
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS / '06_roc.png')
plt.show()

comparison = test_table.set_index('Модель')[['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']]
comparison = comparison.rename(columns={'F1': 'F1-Score', 'ROC-AUC': 'AUC-ROC'})
ax = comparison.plot(kind='bar', figsize=(15, 6), width=0.82,
                     color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'])
ax.set_title('Сравнение метрик качества классификаторов')
ax.set_xlabel('Метод')
ax.set_ylabel('Значение метрики')
ax.set_ylim(0, 1.05)
ax.legend(title='Метрика', loc='lower right')
ax.grid(axis='x', visible=False)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(PLOTS / '07_metrics.png', dpi=160)
plt.show()

# %% confusion
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
fig.suptitle('Матрицы ошибок классификаторов на тестовой выборке', fontsize=16)
plt.tight_layout()
plt.savefig(PLOTS / '08_confusion.png', dpi=150)
plt.show()

run_info = {'seed': SEED, 'train_size': len(X_train), 'validation_size': len(X_val),
            'test_size': len(X_test), 'best_method': best_method, 'best_nn_run': best_run,
            'tensorboard_dir': str(LOGS.relative_to(BASE)), 'nn_experiments': len(experiments)}
(RESULTS / 'run_info.json').write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding='utf-8')
print('Логи TensorBoard:', LOGS)
