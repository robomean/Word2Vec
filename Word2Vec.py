class Word2Vec:
    
    def __build_dicts(self, words):
        word_to_num = {}
        num_to_word = ["" for _ in range(len(words))]
        
        for num, word in enumerate(words):
            if word not in word_to_num:
                num_to_word[num] = word
                word_to_num[word] = num
                
        return num_to_word, word_to_num
    
    def __build_prob_dict(self, word_freq):
        prob_sum = (word_freq[1] ** self.alpha).sum()
        word_prob = (word_freq[1] ** self.alpha) / prob_sum
        
        word_prob_dict = {}
        for num, word in enumerate(word_freq[0]):
            word_prob_dict[word] = word_prob[num]
    
        return word_prob_dict
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def init_2d_lists(size, *args):
        for arg in args:
            for i in range(size):
                arg.append([])
        return args
    
    @staticmethod
    def window(pos, window_size):
        window = []
        for i in range(pos - window_size, pos + window_size + 1):
            if i < 0 or i >= len(words) or i == pos:
                continue
            window.append(word_to_num[words[i]])

        return window
    
    def __build_pos_data(self, words, text_size):
        pos_samples, inv_pos_samples = self.init_2d_lists(self.vocab_size, [], [])
        
        for position in range(text_size):
            window_nums = self.window(position, self.window_size)
            for word_num in window_nums:
                pos_samples[word_to_num[words[position]]].append(word_num)
                inv_pos_samples[word_num].append(word_to_num[words[position]])
        
        return pos_samples, inv_pos_samples
    
    def __sample_negative(self, position, pos_samples):
        cur_pos_words = pos_samples[word_to_num[words[pos]]]
        words_to_choose = list(self.vocab - set(cur_pos_words))
        word_probs = [self.word_prob_dict[word] for word in words_to_choose]
        negative_words = random.choices(words_to_choose, weights=word_probs, k=len(cur_pos_words) * self.neg_per_pos)
        negative_nums = [word_to_num[word] for word in negative_words]
        return negative_nums
    
    def __build_neg_data(self, words, pos_samples, text_size):
        neg_samples, inv_neg_samples = self.init_2d_lists(self.vocab_size, [], [])
        
        for position in range(len(words)):
            if len(neg_samples[word_to_num[words[position]]]) == 0:
                for word_num in self.__sample_negative(position, pos_samples):
                    neg_samples[word_to_num[words[position]]].append(word_num)
                    inv_neg_samples[word_num].append(word_to_num[words[position]])
        
        return neg_samples, inv_neg_samples
    
    def __get_data_to_train(self, words):
        text_size = len(words)
        
        pos_samples, inv_pos_samples = self.__build_pos_data(words, text_size)
        neg_samples, inv_neg_samples = self.__build_neg_data(words, pos_samples, text_size)
        
        return pos_samples, neg_samples, inv_pos_samples, inv_neg_samples
    
    def __build_w2v_dict(self, w, c):
        word2vec = w + c
        w2v_dict = {}
        for word in self.word_to_num:
            w2v_dict[word] = word2vec[self.word_to_num[word]]
    
    def __build_embedding(self, data):
        words = data.split(' ')
        word_freq = np.unique(words, return_counts=True)
        self.vocab = set(word_freq[0])
        
        self.vocab_size = len(self.vocab)

        self.num_to_word, self.word_to_num = self.__build_dicts(word_freq[0])

        self.word_prob_dict = self.__build_prob_dict(word_freq)
        
        pos_samples, neg_samples, inv_pos_samples, inv_neg_samples = self.__get_data_to_train(words)

        w = np.random.random((self.vocab_size, self.embed_size))
        c = np.random.random((self.vocab_size, self.embed_size))

        for iteration in range(self.iter_num):
            for word_num in range(self.vocab_size):
                possible_words = inv_pos_samples[word_num]
                print(random.choices(possible_words, k = 1))
                random_word = random.choices(possible_words, k = 1)[0]
                c[word_num] = c[word_num] - self.eta * (sigmoid(c[word_num] * w[random_word]) - 1) * w[random_word]

            for word_num in range(self.vocab_size):
                possible_words = inv_neg_samples[word_num]
                print(possible_words)
                random_word = random.choices(possible_words, k = 1)[0]
                c[word_num] = c[word_num] - self.eta * sigmoid(c[word_num] * w[random_word]) * w[random_word]

            for word_num in range(self.vocab_size):
                positive_nums = pos_samples[word_num]
                negative_nums = neg_samples[word_num]
                positive_num = random.choices(positive_nums, k = 1)[0]
                negative_num = random.choices(negative_nums, k = 2)
                negative_sum = 0
                for neg in negative_num:
                    negative_sum += sigmoid(c[neg] * w[word_num]) * c[neg]
                w[word_num] = w[word_num] - self.eta * ((sigmoid(c[positive_num] * w[word_num]) - 1) * c[positive_num] + negative_sum) 

        w2v_dict = self.__build_w2v_dict(w, c)

        return w2v_dict 
    
    def __init__(self, data, alpha=0.75, window_size = 2, neg_per_pos = 2, embed_size = 100, eta = 0.01, iter_num = 1000):
        self.alpha = alpha
        self.window_size = window_size
        self.neg_per_pos = neg_per_pos
        self.embed_size = embed_size
        self.eta = eta
        self.iter_num = iter_num
        self.embedding = self.__build_embedding(data)
