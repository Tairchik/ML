"""Проверка сохранённых результатов без повторного обучения."""
import json
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

folder = Path(__file__).resolve().parent
results = folder / 'results'
info = json.loads((results / 'run_info.json').read_text(encoding='utf-8'))
predictions = pd.read_csv(results / 'test_predictions.csv')
metrics = pd.read_csv(results / 'test_metrics.csv').set_index('Модель')
splits = pd.read_csv(results / 'split_assignment.csv', index_col='row_id')['split']
assert splits.value_counts().to_dict() == {'train': 2286, 'validation': 762, 'test': 762}
assert len(predictions) == 762 and len(metrics) == 9
assert set(predictions['row_id']) == set(splits.index[splits == 'test'])

functions = {'Accuracy': accuracy_score, 'Precision': precision_score,
             'Recall': recall_score, 'F1': f1_score, 'ROC-AUC': roc_auc_score}
for name, row in metrics.iterrows():
    for metric, function in functions.items():
        column = name + ('__score' if metric == 'ROC-AUC' else '__prediction')
        np.testing.assert_allclose(function(predictions['y_true'], predictions[column]), row[metric], atol=1e-12)

data = pd.read_csv(folder / 'data/rice.csv')
features = data.drop(columns='Class')
test_x = features.loc[predictions['row_id']]
train_x = features.loc[splits.index[splits == 'train']]
for name in metrics.index:
    if name == 'TensorFlow MLP':
        continue
    model = joblib.load(results / 'models' / f'{name}.joblib')
    np.testing.assert_array_equal(model.predict(test_x), predictions[name + '__prediction'])
    if 'scale' in model.named_steps:
        assert model.named_steps['scale'].n_samples_seen_ == 2286
        np.testing.assert_allclose(model.named_steps['scale'].mean_, train_x.mean())

validation = pd.read_csv(results / 'validation_metrics.csv')
assert validation.loc[validation['ROC-AUC'].idxmax(), 'Модель'] == info['best_method']
neural = pd.read_csv(results / 'neural_experiments.csv')
assert len(neural) == 11
for _, row in neural.iterrows():
    history = pd.read_csv(results / 'neural_histories' / f'{row["Run"]}.csv')
    assert len(history) == row['Эпохи']
    for subset in ['train', 'validation']:
        events = EventAccumulator(str(folder / info['tensorboard_dir'] / row['Run'] / subset))
        events.Reload()
        tags = events.Tags()['tensors'] + events.Tags()['scalars']
        assert 'epoch_loss' in tags and 'epoch_accuracy' in tags

notebook = nbformat.read(folder / 'lab2_classification.ipynb', as_version=4)
nbformat.validate(notebook)
code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
assert [cell.execution_count for cell in code_cells] == list(range(1, 17))
for i, cell in enumerate(notebook.cells):
    if cell.cell_type == 'code':
        assert not any(output.output_type == 'error' for output in cell.outputs)
        assert notebook.cells[i + 1].cell_type == 'markdown'
print('Проверено: 16 ячеек, 9 моделей, 45 метрик, 11 запусков TensorBoard.')
