import unittest

from ellipse_fit_evaluation import evaluate_ellipse_fit


class TestEvaluateEllipseFit(unittest.TestCase):
    def test_perfect_fit_1(self):
        fit_ellipse = {'center': (626.76, 494.98), 'axes': (387.96, 381.45), 'angle': 170}
        score = evaluate_ellipse_fit('2018-02-15 17.26.47.474000.tiff', fit_ellipse,
                                     csv_filepath='./tests/ground_truths_develop.csv')
        self.assertEqual(score, 1.0)

    def test_perfect_fit_2(self):
        fit_ellipse = {'center': (635.86, 521.4), 'axes': (168.05, 165.09), 'angle': 164}
        score = evaluate_ellipse_fit('2018-02-15 17.27.27.162000.tiff', fit_ellipse,
                                     csv_filepath='./tests/ground_truths_develop.csv')
        self.assertEqual(score, 1.0)

    def test_perfect_fit_3(self):
        fit_ellipse = {'center': (847.58, 751.44), 'axes': (33.93, 30.67), 'angle': 18}
        score = evaluate_ellipse_fit('2018-02-15 17.27.54.680000.tiff', fit_ellipse,
                                     csv_filepath='./tests/ground_truths_develop.csv')
        self.assertEqual(score, 1.0)

    def test_partial_fit_1(self):
        fit_ellipse = {'center': (630.0, 500.0), 'axes': (390.0, 380.0), 'angle': 170}
        score = evaluate_ellipse_fit('2018-02-15 17.26.47.474000.tiff', fit_ellipse,
                                     csv_filepath='./tests/ground_truths_develop.csv')
        self.assertAlmostEqual(score, 0.9909615)

    def test_partial_fit_2(self):
        fit_ellipse = {'center': (600.0, 500.0), 'axes': (100.0, 100.0), 'angle': 200}
        score = evaluate_ellipse_fit('2018-02-15 17.27.27.162000.tiff', fit_ellipse,
                                     csv_filepath='./tests/ground_truths_develop.csv')
        self.assertAlmostEqual(score, 0.3624268)

    def test_partial_fit_3(self):
        fit_ellipse = {'center': (900.0, 800.0), 'axes': (100.0, 100.0), 'angle': 0}
        score = evaluate_ellipse_fit('2018-02-15 17.27.54.680000.tiff', fit_ellipse,
                                     csv_filepath='./tests/ground_truths_develop.csv')
        self.assertAlmostEqual(score, 0.1081905)

    def test_invalid_fit(self):
        fit_ellipse = {'center': (200, 100), 'axes': (50, 25), 'angle': 0}
        score = evaluate_ellipse_fit('2018-02-15 17.26.47.474000.tiff', fit_ellipse,
                                     csv_filepath='./tests/ground_truths_develop.csv')
        self.assertEqual(score, 0.0)

    def test_nonempty_fit_with_empty_gt(self):
        fit_ellipse = {'center': (200, 100), 'axes': (50, 25), 'angle': 0}
        score = evaluate_ellipse_fit('2018-02-15 17.36.17.793000.tiff', fit_ellipse,
                                     csv_filepath='./tests/ground_truths_develop.csv')
        self.assertEqual(score, 0.0)

    def test_empty_fit_with_empty_gt(self):
        fit_ellipse = None
        score = evaluate_ellipse_fit('2018-02-15 17.36.17.793000.tiff', fit_ellipse,
                                     csv_filepath='./tests/ground_truths_develop.csv')
        self.assertEqual(score, 1.0)

    def test_empty_fit_with_nonempty_gt(self):
        fit_ellipse = None
        score = evaluate_ellipse_fit('2018-02-15 17.26.47.474000.tiff', fit_ellipse,
                                     csv_filepath='./tests/ground_truths_develop.csv')
        self.assertEqual(score, 0.0)

    def test_invalid_tiff_filename(self):
        with self.assertRaises(ValueError):
            fit_ellipse = {'center': (200, 100), 'axes': (50, 25), 'angle': 0}
            score = evaluate_ellipse_fit('nonexisting.tiff', fit_ellipse,
                                         csv_filepath='./tests/ground_truths_develop.csv')

    def test_invalid_csv_filename(self):
        with self.assertRaises(FileNotFoundError):
            fit_ellipse = {'center': (200, 100), 'axes': (50, 25), 'angle': 0}
            score = evaluate_ellipse_fit('2018-02-15 17.26.47.474000.tiff', fit_ellipse,
                                         csv_filepath='./tests/nonexisting.csv')

if __name__ == "__main__":
    unittest.main()