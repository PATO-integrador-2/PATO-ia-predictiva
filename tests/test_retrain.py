import json
import math
import os
import sys
import unittest

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from level2.lv2_vw1_train import PRIVATION_COLS, _prepare_data  # noqa: E402
from level2.lv2_vw1_predict import DepartmentPovertyPredictor  # noqa: E402

MODELS = ['lv1_vw1', 'lv1_vw2', 'lv1_vw3', 'lv1_vw4', 'lv1_vw5', 'lv1_vw6', 'lv2_vw1']


def _household_frame(households_per_group):
    rows = []
    for region, departamento in [('Caribe', 'Atlántico'), ('Pacífica', 'Cauca')]:
        for year in range(2018, 2026):
            for household in range(households_per_group):
                row = {'region': region, 'departamento': departamento, 'anio': year}
                for column in PRIVATION_COLS:
                    row[column] = float(household)
                row['ipm'] = 0.2 + 0.01 * household + 0.001 * (year - 2018)
                rows.append(row)
    return pd.DataFrame(rows)


class PrepareDataAggregationTest(unittest.TestCase):
    def test_one_row_per_department_and_year(self):
        data = _prepare_data(_household_frame(households_per_group=3))
        frame = data['frame']
        self.assertFalse(frame.duplicated(subset=['region', 'departamento', 'anio']).any())
        # 2 departamentos x 8 años, menos 3 años perdidos por los lags
        self.assertEqual(len(frame), 2 * 5)

    def test_ipm_is_mean_of_households(self):
        households = 3
        data = _prepare_data(_household_frame(households_per_group=households))
        frame = data['frame']
        row = frame[(frame['departamento'] == 'Cauca') & (frame['anio'] == 2021)].iloc[0]
        expected_ipm = np.mean([0.2 + 0.01 * h + 0.001 * 3 for h in range(households)])
        self.assertAlmostEqual(row['ipm'], expected_ipm, places=9)
        self.assertAlmostEqual(row[PRIVATION_COLS[0]], np.mean(range(households)), places=9)

    def test_temporal_split_uses_train_and_test_years(self):
        data = _prepare_data(_household_frame(households_per_group=2))
        frame = data['frame']
        self.assertEqual(sorted(frame.loc[frame['anio'] <= 2022, 'anio'].unique().tolist()), [2021, 2022])
        self.assertEqual(sorted(frame.loc[frame['anio'] >= 2023, 'anio'].unique().tolist()), [2023, 2024, 2025])


class RetrainedArtifactsTest(unittest.TestCase):
    def test_metrics_files_are_valid(self):
        for model in MODELS:
            with self.subTest(model=model):
                path = os.path.join(BASE_DIR, 'models', f'train_{model}', 'metrics.json')
                with open(path, encoding='utf-8') as file:
                    metrics = json.load(file)
                selected = metrics['selected_model']
                self.assertIn(selected, metrics['available_models'])
                self.assertIn('train_date', metrics)
                candidate = metrics['candidates'][selected]
                for name in ('mae', 'rmse'):
                    self.assertTrue(math.isfinite(candidate[name]), name)
                    self.assertGreaterEqual(candidate[name], 0)
                self.assertGreaterEqual(candidate['rmse'], candidate['mae'])
                self.assertLessEqual(candidate['r2'], 1.0)

    def test_model_artifacts_load(self):
        for model in MODELS:
            with self.subTest(model=model):
                path = os.path.join(BASE_DIR, 'models', f'train_{model}', 'model.pkl')
                self.assertTrue(os.path.exists(path))
                self.assertIsNotNone(joblib.load(path))


class DepartmentPredictorTest(unittest.TestCase):
    def setUp(self):
        self.predictor = DepartmentPovertyPredictor()

    def test_predict_household_returns_probability_like_value(self):
        row = {column: 0.5 for column in self.predictor.feature_cols}
        row.update({
            'year': 2026,
            'year_squared': 2026 ** 2,
            'years_since_2010': 16,
            'region': str(self.predictor.le_region.classes_[0]),
            'departamento': str(self.predictor.le_departamento.classes_[0]),
        })
        value = self.predictor.predict_household(row)
        self.assertTrue(math.isfinite(value))
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)


if __name__ == '__main__':
    unittest.main()
