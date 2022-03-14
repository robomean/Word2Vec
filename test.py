import unittest

from Word2Vec import Word2Vec


class Test(unittest.TestCase):

    def test_window_edge(self):
        self.assertEqual(test_w2v.window(0),
                         [test_w2v.word_to_num['learning'], test_w2v.word_to_num['is']])
        self.assertEqual(test_w2v.window(len(test_w2v.words) - 1),
                         [test_w2v.word_to_num['the'], test_w2v.word_to_num['needed']])

    def test_window_middle(self):
        self.assertEqual(test_w2v.window(2),
                         [test_w2v.word_to_num['machine'], test_w2v.word_to_num['learning'],
                          test_w2v.word_to_num['the'], test_w2v.word_to_num['study']])

    def test_dicts(self):
        self.assertEqual(len(test_w2v.word_to_num), 65)
        self.assertEqual(test_w2v.word_to_num[test_w2v.num_to_word[0]], 0)

    def test_probs(self):
        probs_sum = 0
        for value in test_w2v.word_prob_dict.values():
            probs_sum += value

        self.assertAlmostEqual(probs_sum, 1)
        self.assertLessEqual(test_w2v.word_prob_dict['study'], test_w2v.word_prob_dict['the'])

    def test_samples(self):
        self.assertEqual(len(test_w2v.get_data_to_train(test_w2v.words)[0][0]),
                         len(test_w2v.get_data_to_train(test_w2v.words)[1][0]) // test_w2v.neg_per_pos)
        self.assertIn(test_w2v.word_to_num[test_w2v.words[1]],
                      test_w2v.get_data_to_train(test_w2v.words)[0][test_w2v.word_to_num['machine']])
        self.assertNotIn(test_w2v.word_to_num[test_w2v.words[0]],
                         test_w2v.get_data_to_train(test_w2v.words)[0][test_w2v.word_to_num['machine']])
        self.assertNotIn(test_w2v.word_to_num[test_w2v.words[1]],
                         test_w2v.get_data_to_train(test_w2v.words)[1][test_w2v.word_to_num['machine']])

        for i in range(65):
            for pos in test_w2v.get_data_to_train(test_w2v.words)[0][i]:
                self.assertNotIn(pos, test_w2v.get_data_to_train(test_w2v.words)[1][i])

        for i in range(65):
            for pos in test_w2v.get_data_to_train(test_w2v.words)[2][i]:
                self.assertNotIn(pos, test_w2v.get_data_to_train(test_w2v.words)[3][i])

    # Embeddings themselves were tested on https://ods.ai/tracks/nlp-course-spring-22


if __name__ == '__main__':
    with open('test_data.txt', 'r') as file:
        test_data = file.read().rstrip()
    test_w2v = Word2Vec(test_data)
    unittest.main()
