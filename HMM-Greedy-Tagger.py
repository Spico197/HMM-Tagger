import numpy as np


def load_data(datapath):  
    sentence = list()
    sentences = list()  
    tags = list()
    tag = list()
    with open(datapath,'r',encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) > 0:
                sentence.append(line.split()[1])
                tag.append(line.split()[3])
            else:
                sentences.append(sentence)
                tags.append(tag)
                tag = []
                sentence = []     # 注意sentence = [] 和 sentence.clear()的区别 ,
                                  # sentence = []是新开辟内存，原来的sentence在sentences中的值不会改变，
                                  # 而sentence.clear()直接把原来的sentence清空，使得snetences中的内容也清空了
    return sentences, tags


class HMM(object):
    def __init__(self, sentences, tags):
        words = set()
        tags_ = set()
        for sentence, labels in zip(sentences, tags):
            for word, tag in zip(sentence, labels):
                words.add(word)
                tags_.add(tag)
        words_list = sorted(list(words))
        tags_list = sorted(list(tags_))
        tags_list.append('<START>')
        words_list.append("<UNK>")     #加入未知词
        self.word2id = {word:index for index, word in enumerate(words_list)} 
        self.tag2id = {tag:index for index, tag in enumerate(tags_list)}
        self.id2tag = {self.tag2id[tag]: tag for tag in self.tag2id.keys()}
        self.vocab_length = len(self.word2id)   #词的个数,包括未知词
        self.tag_num = len(self.tag2id)         #词性个数
        self.transition_mat = np.zeros((self.tag_num + 1,self.tag_num))
        self.emission_mat = np.zeros((self.tag_num,self.vocab_length))

        for sentence, labels in zip(sentences, tags):
            pre = 0
            for word, tag in zip(sentence, labels):
                self.emission_mat[self.tag2id.get(tag)][self.word2id.get(word)] += 1
                self.transition_mat[pre][self.tag2id.get(tag)] += 1
                pre = self.tag2id.get(tag)

        self.transition_mat = (self.transition_mat + alpha) / (np.sum(self.transition_mat, axis=1, keepdims=True) + alpha*(self.tag_num))
        self.emission_mat = (self.emission_mat + alpha) / (np.sum(self.emission_mat, axis=1, keepdims=True) + alpha*self.vocab_length)


    def predict(self, sentences):
        transition = np.log(self.transition_mat)
        emission = np.log(self.emission_mat)
        
        result_tags = []
        for sentence in sentences:
            result_tag = []
            word_index = [self.word2id.get(word, self.word2id["<UNK>"]) for word in sentence]
            sentence_length = len(sentence)  #句子长度

            max_prob = transition[0, :].T + emission[:, word_index[0]]
            max_tag = max_prob.argmax()
            result_tag.append(self.id2tag[max_tag])
            result_tags.append(result_tag)
            for ind in range(1, sentence_length, 1):
                max_prob = transition[max_tag, :].T + emission[:, word_index[ind]] + max_prob
                max_tag = max_prob.argmax()
                result_tag.append(self.id2tag[max_tag])
                result_tags.append(result_tag)
        return result_tags

    def evaluate(self, preds, labels):
        correct_words = 0
        preds_ = []
        labels_ = []
        for pred, label in zip(preds, labels):
            preds_ += pred
            labels_ += label
        for pred, tag in zip(preds_, labels_):
            if pred == tag:
                correct_words += 1

        print('准确率：%f' % (correct_words / len(preds_)))


if __name__ == "__main__":
    train_data_file = 'big-data/train.conll'
    dev_data_file = 'big-data/dev.conll'
    test_data_file = 'big-data/test.conll'
    alpha = 0.01

    import time
    sentences, tags = load_data(train_data_file)
    start_time = time.time()
    hmm = HMM(sentences, tags)
    end_time = time.time() - start_time
    print('training time:', end_time)
    
    print("\n训练集：")
    start_time = time.time()
    preds = hmm.predict(sentences)
    end_time = time.time() - start_time
    print('predicting time on training set:', end_time)
    hmm.evaluate(preds, tags)

    print("\n开发集：")
    sentences, tags = load_data(dev_data_file)
    start_time = time.time()
    preds = hmm.predict(sentences)
    end_time = time.time() - start_time
    print('predicting time on dev set:', end_time)
    hmm.evaluate(preds, tags)

    print("\n测试集：")
    sentences, tags = load_data(test_data_file)
    start_time = time.time()
    preds = hmm.predict(sentences)
    end_time = time.time() - start_time
    print('predicting time on training set:', end_time)
    hmm.evaluate(preds, tags)
