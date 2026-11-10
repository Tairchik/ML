"""Проверить артефакты эксперимента без повторного обучения нейросетей."""
import hashlib
import json
import os
import re
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

folder = Path(__file__).resolve().parent
results = folder / 'results'
info = json.loads((results / 'run_info.json').read_text(encoding='utf-8'))
data = pd.read_csv(folder / 'data/rice.csv')
splits = pd.read_csv(results / 'split_assignment.csv', index_col='row_id')
cv = pd.read_csv(results / 'cv_folds.csv')
summary = pd.read_csv(results / 'cv_results.csv')
oof = pd.read_csv(results / 'oof_predictions.csv', index_col='row_id')
predictions = pd.read_csv(results / 'test_predictions.csv')
test_metrics = pd.read_csv(results / 'test_metrics.csv').set_index('model')

# Точная копия датасета ЛР2, если основная работа доступна рядом.
original = folder.parent / 'lab2/data/rice.csv'
if original.exists():
    assert hashlib.sha256(original.read_bytes()).digest() == hashlib.sha256(
        (folder / 'data/rice.csv').read_bytes()).digest()
assert splits.index.is_unique and set(splits.index) == set(data.index)
assert splits['split'].value_counts().to_dict() == {'train': 3048, 'test': 762}
target = data['Class'].map({'Osmancik': 0, 'Cammeo': 1})
train_ids, test_ids = train_test_split(data.index, test_size=0.2, random_state=info['seed'], stratify=target)
assert set(train_ids) == set(splits.index[splits['split'] == 'train'])
assert set(test_ids) == set(predictions['row_id'])
assert set(train_ids).isdisjoint(test_ids)
assert set(oof.index) == set(train_ids) and not oof.isna().any().any()
np.testing.assert_array_equal(oof['y_true'], target.loc[oof.index])
np.testing.assert_array_equal(predictions['y_true'], target.loc[predictions['row_id']])

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=info['seed'])
for fold, (fit_index, val_index) in enumerate(folds.split(train_ids, target.loc[train_ids]), start=1):
    expected = set(train_ids[val_index])
    actual = set(splits.index[(splits['split'] == 'train') & (splits['fold'] == fold)])
    assert expected == actual and expected.isdisjoint(train_ids[fit_index])
    assert len(expected) in (609, 610)
assert (splits.loc[test_ids, 'fold'] == 0).all()
np.testing.assert_array_equal(oof['fold'], splits.loc[oof.index, 'fold'])

assert len(cv) == 45 and len(summary) == 9
assert set(cv['architecture']) == {'dense16', 'dense32_16', 'dropout32_16'}
assert set(cv['learning_rate']) == {0.0001, 0.001, 0.01}
assert not cv.duplicated(['architecture', 'learning_rate', 'fold']).any()
assert cv.groupby('run').size().eq(5).all()
assert (cv['fit_seconds'] > 0).all()
for row in cv.itertuples():
    held = oof[oof['fold'] == row.fold]
    assert len(held) == row.val_size and row.train_size + row.val_size == len(train_ids)
    scores = held[row.run].to_numpy(dtype='float32')
    np.testing.assert_allclose(accuracy_score(held['y_true'], scores >= 0.5), row.val_accuracy, atol=1e-12)
    # Восстанавливаем исходный float32: log_loss учитывает точность типа при clipping.
    np.testing.assert_allclose(log_loss(held['y_true'], scores), row.val_loss, atol=1e-12)
    history = pd.read_csv(results / 'histories' / f'{row.run}_fold{row.fold}.csv')
    assert history['epoch'].tolist() == list(range(1, info['epochs'] + 1))
    np.testing.assert_allclose(history.iloc[-1]['val_accuracy'], row.val_accuracy, atol=1e-6)
    for subset in ['train', 'validation']:
        events = EventAccumulator(
            str(folder / info['tensorboard_dir'] / row.run / f'fold_{row.fold}' / subset),
            size_guidance={'tensors': 0, 'scalars': 0},
        )
        events.Reload()
        tags = events.Tags()
        for tag in ['epoch_accuracy', 'epoch_loss']:
            records = events.Tensors(tag) if tag in tags['tensors'] else events.Scalars(tag)
            assert [record.step for record in records] == list(range(info['epochs']))

for row in summary.itertuples():
    part = cv[cv['run'] == row.run]
    for raw, mean, std in [('val_accuracy', 'mean_accuracy', 'std_accuracy'),
                           ('val_loss', 'mean_loss', 'std_loss')]:
        np.testing.assert_allclose(part[raw].mean(), getattr(row, mean), atol=1e-12)
        np.testing.assert_allclose(part[raw].std(ddof=1), getattr(row, std), atol=1e-12)
    np.testing.assert_allclose(part['fit_seconds'].mean(), row.mean_fit_seconds, atol=1e-10)
ranking = summary.sort_values(['mean_accuracy', 'std_accuracy', 'mean_loss', 'parameters', 'run'],
                              ascending=[False, True, True, True, True])
assert ranking.iloc[0]['run'] == info['best_run'] == summary.iloc[0]['run']

for name, prefix in [('Baseline', 'baseline'), ('Tuned MLP', 'tuned')]:
    scores = predictions[prefix + '_score'].to_numpy(dtype='float32')
    np.testing.assert_array_equal(scores >= 0.5, predictions[prefix + '_prediction'])
    np.testing.assert_allclose(accuracy_score(predictions['y_true'], scores >= 0.5), test_metrics.loc[name, 'accuracy'], atol=1e-12)
    np.testing.assert_allclose(log_loss(predictions['y_true'], scores), test_metrics.loc[name, 'loss'], atol=1e-12)

for name in ['baseline', 'final']:
    history = pd.read_csv(results / 'histories' / f'{name}.csv')
    assert len(history) == info['epochs']
    events = EventAccumulator(str(folder / info['tensorboard_dir'] / name / 'train'),
                              size_guidance={'tensors': 0, 'scalars': 0})
    events.Reload()
    for tag in ['epoch_accuracy', 'epoch_loss']:
        assert [record.step for record in events.Tensors(tag)] == list(range(info['epochs']))

notebook = nbformat.read(folder / 'lab2_cross_validation.ipynb', as_version=4)
nbformat.validate(notebook)
code = {}
for block in re.split(r'^# %% ', (folder / 'lab2_cross_validation.py').read_text(encoding='utf-8'), flags=re.M)[1:]:
    name, source = block.split('\n', 1)
    code[name] = source.strip()
code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
assert len(code_cells) == len(code)
assert [cell.execution_count for cell in code_cells] == list(range(1, len(code) + 1))
for cell in code_cells:
    assert cell.source == code[cell.id]
    assert not any(output.output_type == 'error' for output in cell.outputs)

if (results / 'models/best.keras').exists():
    import joblib
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    features = data.drop(columns='Class')
    scaler = joblib.load(results / 'models/scaler.joblib')
    assert scaler.n_samples_seen_ == len(train_ids)
    np.testing.assert_allclose(scaler.mean_, features.loc[train_ids].mean(), atol=1e-10)
    scaled = scaler.transform(features.loc[predictions['row_id']]).astype('float32')
    for path, prefix in [('baseline.keras', 'baseline'), ('best.keras', 'tuned')]:
        model = tf.keras.models.load_model(results / 'models' / path)
        scores = model.predict(scaled, verbose=0).ravel()
        np.testing.assert_allclose(scores, predictions[prefix + '_score'], atol=1e-6)

print(f'Проверено: {len(code_cells)} выполненных ячеек, 9 конфигураций, 45 фолдов, '
      'метрики OOF/test, выбор победителя и события TensorBoard.')
