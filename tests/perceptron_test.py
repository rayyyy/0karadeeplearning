
import unittest

# テストしたい関数のあるモジュールをimport
import practice.perceptron as perceptron


# クラス名はなんでも良いが、unittest.TestCaseの継承は必須
class Perceptron(unittest.TestCase):
    # unittestでは関数名がtest〜で始まる関数をテストコードとして扱う。
    # test〜としておかないとテストが実行されないので注意。
    def test_AND(self):
        patterns = [
            (0, 0, 0),
            (0, 1, 0),
            (1, 0, 0),
            (1, 1, 1),
        ]

        for x, y, result in patterns:
            # subTest()の引数には失敗時に出力したい内容を指定。パラメータ全てを入れておくのが無難。
            self.assertEqual(perceptron.AND(x, y), result)


if __name__ == '__main__':
    unittest.main()
