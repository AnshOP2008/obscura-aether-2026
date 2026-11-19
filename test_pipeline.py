import unittest
import numpy as np
import pandas as pd
from pipeline import metric, validate_submission, TARGETS


class ContractTests(unittest.TestCase):
    def test_missing_label_is_masked_before_arithmetic(self):
        y = np.array([[1, 0, 0, 0, 0], [np.nan, 0, 0, 0, 0.]])
        p = np.array([[3, 0, 0, 0, 0], [900, 0, 0, 0, 0.]])
        self.assertAlmostEqual(metric(y, p)['overall'], .4)

    def test_column_rmse_is_not_pooled(self):
        y = np.zeros((2, 5))
        p = y.copy()
        p[:, 0] = 5
        self.assertEqual(metric(y, p)['overall'], 1)

    def test_invalid_submission_rejected(self):
        template = pd.DataFrame({'sample_id': ['001', '002'], **{t: [0., 0.] for t in TARGETS}})
        validate_submission(template, template[['sample_id']], template)
        bad = template.copy()
        bad.loc[0, TARGETS[0]] = np.nan
        with self.assertRaises(ValueError):
            validate_submission(bad, template[['sample_id']], template)


if __name__ == '__main__':
    unittest.main()
