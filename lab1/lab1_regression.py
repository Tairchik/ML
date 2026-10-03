# Полный листинг вычислительных ячеек lab1_regression.ipynb

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 180)

# Файл доступен при запуске из папки lab1 или из корня проекта
data_path = Path('machine.data')
if not data_path.exists():
    data_path = Path('lab1/machine.data')

columns = ['VendorName', 'ModelName', 'MYCT', 'MMIN', 'MMAX',
           'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']
raw_df = pd.read_csv(data_path, sep=',', header=None, names=columns)

print('Размер датасета:', raw_df.shape)
print(raw_df.head())
raw_df.info()

numeric_columns = ['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP']
numeric_df = raw_df[numeric_columns]

print(numeric_df.describe().round(2))
print('\nАсимметрия и эксцесс:')
print(pd.DataFrame({
    'Асимметрия': numeric_df.skew(),
    'Эксцесс': numeric_df.kurt()
}).round(2))

numeric_df.hist(bins=12, figsize=(12, 8), edgecolor='black')
plt.suptitle('Распределение признаков и целевой переменной', fontsize=16)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, column in zip(axes.flat, numeric_columns):
    sns.boxplot(y=numeric_df[column], ax=ax)
    ax.set_title(column)
axes.flat[-1].set_visible(False)
plt.suptitle('Диаграммы размаха признаков и PRP')
plt.tight_layout()
plt.show()

q1 = numeric_df.quantile(0.25)
q3 = numeric_df.quantile(0.75)
iqr = q3 - q1
outlier_mask = (numeric_df < q1 - 1.5 * iqr) | (numeric_df > q3 + 1.5 * iqr)
print('Число значений за границами 1.5 IQR:')
print(outlier_mask.sum())
print('Строк с хотя бы одним таким значением:', outlier_mask.any(axis=1).sum())

correlation_matrix = numeric_df.corr(method='pearson')
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f',
            vmin=-1, vmax=1)
plt.title('Матрица корреляций Пирсона')
plt.tight_layout()
plt.show()

# Константа нужна для вспомогательных регрессий с обычным свободным членом
X_vif = add_constant(numeric_df.drop(columns=['PRP']).dropna())
vif_data = pd.DataFrame({
    'Признак': X_vif.columns[1:],
    'VIF': [variance_inflation_factor(X_vif.values, i)
            for i in range(1, X_vif.shape[1])]
})
print(vif_data.round(3).to_string(index=False))

print('Пропуски в исходных данных:')
print(raw_df.isna().sum())
print('Полные дубликаты:', raw_df.duplicated().sum())
print('Повторы производителя и модели:',
      raw_df.duplicated(subset=['VendorName', 'ModelName']).sum())

df = raw_df.drop(columns=['VendorName', 'ModelName', 'ERP']).copy()
rows_before = len(df)
df = df.dropna()
print('Удалено строк с пропусками:', rows_before - len(df))
print('Осталось наблюдений:', len(df))
print('Дубликаты после исключения столбцов:', df.duplicated().sum())

X = df.drop(columns=['PRP'])
y = df['PRP']
print('Количество признаков:', X.shape[1])
print('Минимум PRP:', y.min())
print('Количество нулей в PRP:', (y == 0).sum())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print('Обучающая выборка:', X_train.shape)
print('Тестовая выборка:', X_test.shape)

# Эти матрицы используем для наглядного исследования PCA.
# При кросс-валидации scaler обучается отдельно внутри Pipeline.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print('Первые 5 строк после стандартизации:')
print(pd.DataFrame(X_train_scaled, columns=X.columns).head().round(3))

cv = KFold(n_splits=5, shuffle=True, random_state=42)
alpha_values = [0.01, 0.1, 1.0, 10.0, 100.0]

models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(max_iter=10000)
}

# Одна небольшая функция, чтобы одинаково считать метрики во всех сравнениях
def calculate_metrics(y_true, y_pred):
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R²': r2_score(y_true, y_pred),
        'MAPE (%)': mean_absolute_percentage_error(y_true, y_pred) * 100
    }

results_before = []
fitted_before = {}
searches_before = {}

for name, model in models.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    param_grid = {} if name == 'Linear Regression' else {'model__alpha': alpha_values}
    search = GridSearchCV(
        pipeline, param_grid, cv=cv,
        scoring='neg_root_mean_squared_error', refit=True
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    fitted_before[name] = best_model
    searches_before[name] = search

    y_pred_train = best_model.predict(X_train)
    y_pred = best_model.predict(X_test)
    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred)

    row = {
        'Модель': name,
        'alpha': search.best_params_.get('model__alpha', np.nan),
        'CV RMSE': -search.best_score_,
        'CV std': search.cv_results_['std_test_score'][search.best_index_]
    }
    for metric in train_metrics:
        row[metric + ' train'] = train_metrics[metric]
        row[metric + ' test'] = test_metrics[metric]
    results_before.append(row)

df_results_before = pd.DataFrame(results_before)
print(df_results_before.round(3).to_string(index=False))

for name in ['Ridge', 'Lasso']:
    cv_results = searches_before[name].cv_results_
    print('\nПодбор alpha:', name)
    print(pd.DataFrame({
        'alpha': [params['model__alpha'] for params in cv_results['params']],
        'CV RMSE': -cv_results['mean_test_score'],
        'CV std': cv_results['std_test_score']
    }).round(3).to_string(index=False))

coefficients = pd.DataFrame({
    name: pipeline.named_steps['model'].coef_
    for name, pipeline in fitted_before.items()
}, index=X.columns)
print('\nКоэффициенты при стандартизованных признаках:')
print(coefficients.round(3))
print('\nСвободные члены:')
print(pd.Series({name: pipeline.named_steps['model'].intercept_
                 for name, pipeline in fitted_before.items()}).round(3))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, pipeline) in zip(axes, fitted_before.items()):
    y_pred = pipeline.predict(X_test)
    residuals = y_test - y_pred
    ax.scatter(y_pred, residuals, alpha=0.7, color='navy')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax.set_title('Остатки: ' + name)
    ax.set_xlabel('Предсказанные значения')
    ax.set_ylabel('Остатки (y_test - y_pred)')
    ax.grid(True, linestyle=':', alpha=0.6)
plt.suptitle('Графики остатков моделей до PCA')
plt.tight_layout()
plt.show()

pca_full = PCA(svd_solver='full')
pca_full.fit(X_train_scaled)
explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
print(pd.DataFrame({
    'Компонента': np.arange(1, len(explained_variance) + 1),
    'Доля дисперсии': explained_variance,
    'Накопленная доля': cumulative_variance
}).round(4).to_string(index=False))

plt.figure(figsize=(9, 5))
plt.plot(range(1, 7), explained_variance, marker='o', label='Доля отдельной компоненты')
plt.plot(range(1, 7), cumulative_variance, marker='s', linestyle='--',
         label='Накопленная доля')
plt.axhline(0.90, color='black', linestyle=':', label='Порог 90%')
plt.title('График каменистой осыпи (Scree Plot)')
plt.xlabel('Номер главной компоненты')
plt.ylabel('Доля объясненной дисперсии')
plt.xticks(range(1, 7))
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

pca = PCA(n_components=0.90, svd_solver='full')
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print('Выбрано компонент:', pca.n_components_)
print('Сохранено дисперсии, %:', round(pca.explained_variance_ratio_.sum() * 100, 2))
print('Размер train до / после PCA:', X_train.shape, X_train_pca.shape)
print('Размер test до / после PCA:', X_test.shape, X_test_pca.shape)

pc_names = [f'PC{i + 1}' for i in range(pca.n_components_)]

# Веса исходных стандартизованных признаков в формулах компонент
component_weights = pd.DataFrame(pca.components_.T, index=X.columns, columns=pc_names)
print('Веса компонент:')
print(component_weights.round(3))

# Корреляционные нагрузки: связь исходного признака с компонентой
joint_data = np.column_stack([X_train_scaled, X_train_pca])
loadings = pd.DataFrame(
    np.corrcoef(joint_data, rowvar=False)[:X.shape[1], X.shape[1]:],
    index=X.columns, columns=pc_names
)
print('\nНагрузки (корреляции признаков с компонентами):')
print(loadings.round(3))

pca_df = pd.DataFrame(X_train_pca, columns=pc_names)
plt.figure(figsize=(7, 5))
sns.heatmap(pca_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Корреляции главных компонент на train')
plt.tight_layout()
plt.show()

X_pca_vif = add_constant(pca_df)
vif_pca = pd.DataFrame({
    'Компонента': pc_names,
    'VIF': [variance_inflation_factor(X_pca_vif.values, i)
            for i in range(1, X_pca_vif.shape[1])]
})
print(vif_pca.round(3).to_string(index=False))

results_pca = []
fitted_pca = {}
searches_pca = {}

for name, model in models.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.90, svd_solver='full')),
        ('model', model)
    ])
    param_grid = {} if name == 'Linear Regression' else {'model__alpha': alpha_values}
    search = GridSearchCV(
        pipeline, param_grid, cv=cv,
        scoring='neg_root_mean_squared_error', refit=True
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    fitted_pca[name] = best_model
    searches_pca[name] = search

    y_pred_train = best_model.predict(X_train)
    y_pred = best_model.predict(X_test)
    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred)

    row = {
        'Модель': name,
        'alpha': search.best_params_.get('model__alpha', np.nan),
        'CV RMSE': -search.best_score_,
        'CV std': search.cv_results_['std_test_score'][search.best_index_]
    }
    for metric in train_metrics:
        row[metric + ' train'] = train_metrics[metric]
        row[metric + ' test'] = test_metrics[metric]
    results_pca.append(row)

df_results_pca = pd.DataFrame(results_pca)
print(df_results_pca.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, pipeline) in zip(axes, fitted_pca.items()):
    y_pred_pca = pipeline.predict(X_test)
    residuals = y_test - y_pred_pca
    ax.scatter(y_pred_pca, residuals, alpha=0.7, color='darkgreen')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax.set_title('Остатки (PCA): ' + name)
    ax.set_xlabel('Предсказанные значения')
    ax.set_ylabel('Остатки (y_test - y_pred)')
    ax.grid(True, linestyle=':', alpha=0.6)
plt.suptitle('Графики остатков моделей после PCA')
plt.tight_layout()
plt.show()

metric_columns = ['Модель', 'RMSE test', 'R² test', 'MAPE (%) test']
df_comparison = pd.merge(
    df_results_before[metric_columns], df_results_pca[metric_columns],
    on='Модель', suffixes=(' до PCA', ' после PCA')
)
print('Сравнение на одних и тех же тестовых наблюдениях:')
print(df_comparison.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, metric in zip(axes, ['RMSE', 'R²', 'MAPE (%)']):
    comparison_plot = pd.DataFrame({
        'До PCA': df_results_before.set_index('Модель')[metric + ' test'],
        'После PCA': df_results_pca.set_index('Модель')[metric + ' test']
    })
    comparison_plot.plot.bar(ax=ax, rot=15)
    ax.set_title(metric + (' (выше — лучше)' if metric == 'R²' else ' (ниже — лучше)'))
    ax.set_xlabel('')
    ax.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# Выбор только по CV на train; test не участвует в выборе
all_results = pd.concat([
    df_results_before.assign(Признаки='Исходные'),
    df_results_pca.assign(Признаки='PCA')
], ignore_index=True)
best_row = all_results.loc[all_results['CV RMSE'].idxmin()]
print('Лучшая модель по CV RMSE:')
print(best_row[['Модель', 'Признаки', 'alpha', 'CV RMSE', 'CV std']])
