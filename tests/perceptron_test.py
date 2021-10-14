
import unittest

# テストしたい関数のあるモジュールをimport
import practice.perceptron as perceptron


class Perceptron(unittest.TestCase):
    def test_AND(self):
        patterns = [
            (0, 0, 0),
            (0, 1, 0),
            (1, 0, 0),
            (1, 1, 1),
        ]

        for x, y, result in patterns:
            with self.subTest(x=x, y=y, result=result):
                self.assertEqual(perceptron.AND(x, y), result)

    def test_NAND(self):
        patterns = [
            (0, 0, 1),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
        ]

        for x, y, result in patterns:
            with self.subTest(x=x, y=y, result=result):
                self.assertEqual(perceptron.NAND(x, y), result)

    def test_OR(self):
        patterns = [
            (0, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 1),
        ]

        for x, y, result in patterns:
            with self.subTest(x=x, y=y, result=result):
                self.assertEqual(perceptron.OR(x, y), result)


if __name__ == '__main__':
    unittest.main()
