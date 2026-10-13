import os
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from level2.lv2_vw1_predict import DepartmentPovertyPredictor, PRIVATION_COLS  # noqa: E402


def _fake_history(ipm_values):
    return pd.DataFrame({
        'anio': list(range(2018, 2018 + len(ipm_values))),
        'ipm': ipm_values,
        **{column: [0.5] * len(ipm_values) for column in PRIVATION_COLS},
    })


class TrendUsesHistoryTest(unittest.TestCase):
    """Los rezagos de IPM deben salir del historial, no quedar en 0."""

    @classmethod
    def setUpClass(cls):
        with mock.patch('level2.lv2_vw1_predict.DepartmentPovertyPredictor.__init__', return_value=None):
            cls.predictor = DepartmentPovertyPredictor()

    def test_household_trend_passes_real_lags_and_rolls_forward(self):
        history = [0.20, 0.21, 0.22, 0.23, 0.24]
        rows = []

        def capture(row):
            rows.append(dict(row))
            return 0.25

        self.predictor._get_historical_data = mock.Mock(return_value=_fake_history(history))
        self.predictor.predict_household = mock.Mock(side_effect=capture)

        out = self.predictor.predict_household_trend('Caribe', 'Atlántico', {}, 2026, 3)

        self.assertEqual([r['predicted_ipm'] for r in out], [0.25, 0.25, 0.25])
        self.assertEqual(rows[0]['ipm_lag1'], 0.24)
        self.assertEqual(rows[0]['ipm_lag2'], 0.23)
        self.assertEqual(rows[0]['ipm_lag3'], 0.22)
        # En el segundo año el rezago 1 es la predicción del año anterior
        self.assertEqual(rows[1]['ipm_lag1'], 0.25)
        self.assertEqual(rows[1]['ipm_lag2'], 0.24)
        self.assertNotEqual(rows[0]['ipm_lag1'], 0)

    def test_household_trend_sets_year_features(self):
        rows = []
        self.predictor._get_historical_data = mock.Mock(return_value=_fake_history([0.2] * 5))
        self.predictor.predict_household = mock.Mock(side_effect=lambda row: rows.append(dict(row)) or 0.2)

        self.predictor.predict_household_trend('Caribe', 'Atlántico', {}, 2026, 2)

        self.assertEqual([r['year'] for r in rows], [2026, 2027])
        self.assertEqual([r['years_since_2010'] for r in rows], [16, 17])
        self.assertEqual(rows[0]['departamento'], 'Atlántico')
        self.assertEqual(rows[0]['region'], 'Caribe')

    def test_household_data_cannot_override_region(self):
        rows = []
        self.predictor._get_historical_data = mock.Mock(return_value=_fake_history([0.2] * 5))
        self.predictor.predict_household = mock.Mock(side_effect=lambda row: rows.append(dict(row)) or 0.2)

        self.predictor.predict_household_trend('Caribe', 'Atlántico', {'region': 'Otra'}, 2026, 1)

        # Los argumentos explícitos mandan sobre household_data
        self.assertEqual(rows[0]['region'], 'Caribe')


class ErrorsPropagateTest(unittest.TestCase):
    """Un departamento sin datos no debe devolver éxito con lista vacía."""

    @classmethod
    def setUpClass(cls):
        with mock.patch('level2.lv2_vw1_predict.DepartmentPovertyPredictor.__init__', return_value=None):
            cls.predictor = DepartmentPovertyPredictor()

    def test_department_trend_raises_when_history_fails(self):
        self.predictor._get_historical_data = mock.Mock(side_effect=ValueError('Datos insuficientes'))
        with self.assertRaises(ValueError):
            self.predictor.predict_department_trend('Caribe', ['Atlántico'], 2026, 3)

    def test_department_trend_does_not_skip_failing_department(self):
        def history(region, departamento):
            if departamento == 'Malo':
                raise ValueError('Datos insuficientes para Caribe - Malo')
            return _fake_history([0.2] * 5)

        self.predictor._get_historical_data = mock.Mock(side_effect=history)
        self.predictor.predict_household = mock.Mock(return_value=0.2)
        with self.assertRaises(ValueError):
            self.predictor.predict_department_trend('Caribe', ['Atlántico', 'Malo'], 2026, 2)

    def test_household_trend_raises_when_history_fails(self):
        self.predictor._get_historical_data = mock.Mock(side_effect=ValueError('sin historial'))
        with self.assertRaises(ValueError):
            self.predictor.predict_household_trend('Caribe', 'Desconocido', {}, 2026, 2)


@unittest.skipUnless(os.getenv('RUN_DB_TESTS') == '1', 'Pruebas contra la base: definir RUN_DB_TESTS=1')
class DatabaseIntegrationTest(unittest.TestCase):
    """Usa la base real. Solo lectura. Se activa con RUN_DB_TESTS=1."""

    @classmethod
    def setUpClass(cls):
        cls.predictor = DepartmentPovertyPredictor()

    def test_department_trend_returns_rows_with_real_history(self):
        df = self.predictor.predict_department_trend('Caribe', ['Atlántico'], 2026, 3)
        self.assertEqual(len(df), 3)
        self.assertTrue(df['ipm_predicho'].between(0, 1).all())
        self.assertEqual(int(df['anio_real'].iloc[0]), 2025)

    def test_household_trend_with_real_history(self):
        zeros = {column: 0 for column in self.predictor.feature_cols if column.startswith(('privacion', 'personas'))}
        out = self.predictor.predict_household_trend('Caribe', 'Atlántico', zeros, 2026, 3)
        self.assertEqual(len(out), 3)
        for item in out:
            self.assertTrue(0 <= item['predicted_ipm'] <= 1)

    def test_unknown_department_raises(self):
        with self.assertRaises(ValueError):
            self.predictor.predict_department_trend('Caribe', ['NoExiste'], 2026, 2)


if __name__ == '__main__':
    unittest.main()
