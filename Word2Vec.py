import numpy as np
import random


class Word2Vec:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def init_2d_lists(size, *args):
        """Makes each empty list given in args filled
           with #size empty lists"""
        for arg in args:
            for i in range(size):
                arg.append([])
        return args

    def window(self, position):
        """Returns all words from around given word
           at a distance of no more than #window_size"""
        window = []
        for i in range(position - self.window_size, position + self.window_size + 1):
            if i < 0 or i >= len(self.words) or i == position:
                continue
            window.append(self.word_to_num[self.words[i]])

        return window

    @staticmethod
    def __build_dicts(words_str):
        """Takes list of words and returns dicts to convert
           from num to word and backwards"""
        word_to_num = {}
        num_to_word = ["" for _ in range(len(words_str))]

        for num, word in enumerate(words_str):
            if word not in word_to_num:
                num_to_word[num] = word
                word_to_num[word] = num

        return num_to_word, word_to_num

    def __build_prob_dict(self, word_freq):
        """Returns list of smoothed with parameter #alpha
           probabilities"""
        prob_sum = (word_freq[1] ** self.alpha).sum()
        word_prob = (word_freq[1] ** self.alpha) / prob_sum

        word_prob_dict = {}
        for num, word in enumerate(word_freq[0]):
            word_prob_dict[word] = word_prob[num]

        return word_prob_dict

    def __build_pos_data(self, words, text_size):
        """Returns list of word pairs standing close to
           each other (#window_size) and inversed list"""
        pos_samples, inv_pos_samples = self.init_2d_lists(self.vocab_size, [], [])

        for position in range(text_size):
            window_nums = self.window(position)
            for word_num in window_nums:
                pos_samples[self.word_to_num[words[position]]].append(word_num)
                inv_pos_samples[word_num].append(self.word_to_num[words[position]])

        return pos_samples, inv_pos_samples

    def __sample_negative(self, position, pos_samples):
        """Randomly choose not standing close (#window_size)
           words according to their uniform probabilities"""
        cur_pos_words = pos_samples[self.word_to_num[self.words[position]]]
        cur_pos_words = [self.num_to_word[num] for num in cur_pos_words]
        words_to_choose = list(self.vocab - set(cur_pos_words))
        word_probs = [self.word_prob_dict[word] for word in words_to_choose]
        negative_words = random.choices(words_to_choose, weights=word_probs, k=len(cur_pos_words) * self.neg_per_pos)
        negative_nums = [self.word_to_num[word] for word in negative_words]
        return negative_nums

    def __build_neg_data(self, words, pos_samples, text_size):
        """Return (#neg_per_pos * #number_of_pos) negative
           (not standing close) examples for each word"""
        neg_samples, inv_neg_samples = self.init_2d_lists(self.vocab_size, [], [])

        for position in range(text_size):
            if not neg_samples[self.word_to_num[words[position]]]:
                for word_num in self.__sample_negative(position, pos_samples):
                    neg_samples[self.word_to_num[words[position]]].append(word_num)
                    inv_neg_samples[word_num].append(self.word_to_num[words[position]])

        return neg_samples, inv_neg_samples

    def get_data_to_train(self, words):
        text_size = len(words)

        pos_samples, inv_pos_samples = self.__build_pos_data(words, text_size)
        neg_samples, inv_neg_samples = self.__build_neg_data(words, pos_samples, text_size)

        return pos_samples, neg_samples, inv_pos_samples, inv_neg_samples

    def __update_c_pos(self, w, c, inv_pos_samples):
        """For each word choose random positive example in
           inversed positive examples and update it's 'c'"""
        for word_num in range(self.vocab_size):
            possible_words = inv_pos_samples[word_num]
            random_word = random.choices(possible_words, k=1)[0]
            c[word_num] = c[word_num] - self.eta * (self.sigmoid(c[word_num] * w[random_word]) - 1) * w[random_word]

    def __update_c_neg(self, w, c, inv_neg_samples):
        """For each word choose random negative example in
           inversed negative examples and update it's 'c'"""
        for word_num in range(self.vocab_size):
            possible_words = inv_neg_samples[word_num]
            random_word = random.choices(possible_words, k=1)[0]
            c[word_num] = c[word_num] - self.eta * self.sigmoid(c[word_num] * w[random_word]) * w[random_word]

    def __update_w(self, w, c, pos_samples, neg_samples):
        """For each word choose random positive example and
           #neg_per_pos negative examples and update it's 'w'"""
        for word_num in range(self.vocab_size):
            positive_num = random.choices(pos_samples[word_num], k=1)[0]
            negative_num = random.choices(neg_samples[word_num], k=self.neg_per_pos)
            negative_sum = 0
            for neg in negative_num:
                negative_sum += self.sigmoid(c[neg] * w[word_num]) * c[neg]
            w[word_num] = w[word_num] - self.eta * ((self.sigmoid(c[positive_num] * w[word_num]) - 1) *
                                                    c[positive_num] + negative_sum)

    def __build_w2v_dict(self, w, c):
        w2v = w + c
        w2v_dict = {}
        for word in self.word_to_num:
            w2v_dict[word] = w2v[self.word_to_num[word]]

        return w2v_dict

    def __build_embedding(self, data):
        self.words = data.split(' ')
        word_freq = np.unique(self.words, return_counts=True)
        self.vocab = set(word_freq[0])
        self.vocab_size = len(word_freq[0])

        self.num_to_word, self.word_to_num = self.__build_dicts(word_freq[0])
        self.word_prob_dict = self.__build_prob_dict(word_freq)

        pos_samples, neg_samples, inv_pos_samples, inv_neg_samples = self.get_data_to_train(self.words)

        w = np.random.random((self.vocab_size, self.embed_size))
        c = np.random.random((self.vocab_size, self.embed_size))
        # updates based on stochastic gradient descent
        for iteration in range(self.iter_num):
            self.__update_c_pos(w, c, inv_pos_samples)
            self.__update_c_neg(w, c, inv_neg_samples)
            self.__update_w(w, c, pos_samples, neg_samples)

        w2v_dict = self.__build_w2v_dict(w, c)
        return w2v_dict

    def __init__(self, data, alpha=0.75, window_size=2, neg_per_pos=2, embed_size=100, eta=0.01,
                 iter_num=750):
        self.alpha = alpha
        self.window_size = window_size
        self.neg_per_pos = neg_per_pos
        self.embed_size = embed_size
        self.eta = eta
        self.iter_num = iter_num
        self.embedding = self.__build_embedding(data)
