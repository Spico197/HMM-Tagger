# import os
# import pickle as pkl
# import numpy as np
# from pprint import pprint

# TRAIN_DATA = 'data/train.conll'
# DEV_DATA = 'data/dev.conll'
# # TEST_DATA = 'data/test.conll'


# class HMMModel(object):
#     def __init__(self):
#         self.states = []
#         self.words = []
#         self.state2id = {}
#         self.word2id = {}
#         self.transition_prob = []
#         self.emission_prob = []

#     def fit(self, data, labels, alpha=0.01):
#         # initialization
#         l_t = []
#         for label in labels:
#             l_t += label
#         self.states = list(set(l_t))
#         self.states.append('<START>')
#         self.states.append('<END>')
#         w_t = []
#         for sent in data:
#             w_t += sent
#         self.words = list(set(w_t))
#         self.words.append('<UNK>')
        
#         self.transition_prob = np.zeros((len(self.states) - 1, len(self.states) - 1))
#         self.emission_prob = np.zeros((len(self.states) - 2, len(self.words)))

#         self.word2id = {self.words[ind]: ind for ind in range(len(self.words))}
#         self.state2id = {self.states[ind]: ind for ind in range(len(self.states))}
#         self.id2state = {self.state2id[state]:state for state in self.states}
#         # fitting before smoothing
#         for sentence, label in zip(data, labels):
#             former_state = '<START>'
#             for word, lab in zip(sentence, label):
#                 self.emission_prob[self.state2id[lab], self.word2id[word]] += 1
#                 self.transition_prob[self.state2id[former_state], self.state2id[lab]] += 1
#                 former_state = lab
#             self.transition_prob[self.state2id[former_state], -1] += 1 # <END> + 1
        
#         # smoothing
#         self.emission_prob  = (self.emission_prob + alpha) / (self.emission_prob.sum(axis=1, keepdims=True) + alpha*len(self.words))
#         self.transition_prob = (self.transition_prob + alpha) / (self.transition_prob.sum(axis=1, keepdims=True) + alpha*(len(self.states) - 1))        
        
#     def predict(self, data):
#         paths = []
#         cnt = 1
#         for sentence in data:
#             word_ind = [self.word2id.get(word, self.word2id['<UNK>']) for word in sentence]
#             max_prob = np.zeros((len(word_ind), len(self.states) - 2))
#             path = np.zeros((len(word_ind), len(self.states) - 2), dtype=int)
#             transition_mat = np.log(self.transition_prob)
#             emission_mat = np.log(self.emission_prob)

#             # initialization
#             path[0] = -1
#             max_prob[0] = transition_mat[-1, :-1] + emission_mat[:, word_ind[0]]

#             # dynamic programming
#             for ind in range(1, len(sentence)):
#                 probs = transition_mat[:-1, :-1] + emission_mat[:, word_ind[ind]] + max_prob[ind - 1].T
#                 max_prob[ind] = np.max(probs, axis=0)
#                 path[ind] = np.argmax(probs, axis=0)
                
#             max_prob[-1] += transition_mat[:-1, -1]
#             step = np.argmax(max_prob[-1])
#             result_path = [step]

#             # backoff
#             for i in range(len(sentence) - 1, 0, -1):
#                 step = path[i][step]
#                 result_path.insert(0, step)
#             result_path = [self.id2state[x] for x in result_path]
#             paths.append(result_path)
#             if cnt % 100 == 0:
#                 print('Process: {}/{}\r'.format(cnt, len(data)), end='')
#             cnt += 1
            
#         return paths

#     def evaluate(self, ground_truth, predict):
#         acc = 0.0
        
#         if len(ground_truth) != len(predict):
#             raise ValueError('Comparison Dimensions Must Agree!')
#         for l1, l2 in zip(ground_truth, predict):
#             if l1 == l2:
#                 acc += 1.0
#         acc /= len(predict)

#         return acc

#     def save(self, path):
#         with open(path, 'wb') as f:
#             pkl.dump(self, f)
    
#     def load(self, path):
#         with open(path, 'rb') as f:
#             model = pkl.load(f)
#         self.state_set = model.state_set
#         self.transition_prob = model.transition_prob
#         self.emission_prob = model.emission_prob


# def load_data(data_path):
#     sentences = []
#     labels = []
#     words = []
#     label_tmp = []
#     with open(TRAIN_DATA, 'r', encoding='utf-8') as f:
#         for line in f:
#             if len(line.strip()) > 1:
#                 words.append(line.strip().split()[1])
#                 label_tmp.append(line.strip().split()[3])
#             else:
#                 sentences.append(words)
#                 labels.append(label_tmp)
#                 words = []
#                 label_tmp = []
#     return sentences, labels


# if __name__ == "__main__":
#     train_sentences, train_labels = load_data(TRAIN_DATA)
#     dev_sentences, dev_labels = load_data(DEV_DATA)
#     # test_sentences, test_labels = load_data(TEST_DATA)

#     model = HMMModel()
#     print('--------------- Train ----------------')
#     model.fit(train_sentences, train_labels, alpha=0.01)
#     print('--------------- Predict --------------')
#     preds = model.predict(['I love you .'.split()])
#     print(preds)
#     # preds = model.predict(dev_sentences)
#     # print('--------------- Test -----------------')
#     # # flatten
#     # test_ls = []
#     # pred_ls = []
#     # for ls1, ls2 in zip(dev_labels, preds):
#     #     test_ls += ls1
#     #     pred_ls += ls2
#     # acc = model.evaluate(test_ls, pred_ls)
#     # print('--------------- Report ---------------')
#     # print('Accuracy: {:.3f} %'.format(acc*100))
#     # print('--------------- End ------------------')


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
        tags_list.append("<START>")   #加入开始词性
        tags_list.append("<STOP>")    #加入结束词性
        words_list.append("<UNK>")     #加入未知词
        self.word2id = {word:index for index, word in enumerate(words_list)} 
        self.tag2id = {tag:index for index, tag in enumerate(tags_list)}
        self.id2tag = {self.tag2id[tag]: tag for tag in self.tag2id.keys()}
        self.vocab_length = len(self.word2id)     #词的个数,包括未知词
        self.tag_num = len(self.tag2id)      #词性个数,包括开始词性和结束词性
        self.transition_mat = np.zeros((self.tag_num-1,self.tag_num-1))       #最后一行表示从开始词性转移到各词性,最后一列表示转移到结束词性
        self.emission_mat = np.zeros((self.tag_num-2,self.vocab_length))            #最后一列表示发射到未知词

        for sentence, labels in zip(sentences, tags):
            pre = -2
            for word, tag in zip(sentence, labels):
                self.emission_mat[self.tag2id.get(tag)][self.word2id.get(word)] += 1
                self.transition_mat[pre][self.tag2id.get(tag)] += 1
                pre = self.tag2id.get(tag)
            self.transition_mat[pre][-1] += 1

        self.transition_mat = (self.transition_mat + alpha) / (np.sum(self.transition_mat, axis=1, keepdims=True) + alpha*(self.tag_num - 1))
        self.emission_mat = (self.emission_mat + alpha) / (np.sum(self.emission_mat, axis=1, keepdims=True) + alpha*self.vocab_length)


    def predict(self, sentences):
        paths = []
        cnt = 1
        for sentence in sentences:
            word_index = [self.word2id.get(word, self.word2id["<UNK>"]) for word in sentence]
            observeNum = len(sentence)  #句子长度
            tagNum = self.tag_num - 2   #词性数
            max_p = np.zeros((observeNum, tagNum))  #第一行用于初始化,max_p[i][j]表示从开始到第i个观测对应第j个词性的概率最大值
            path = np.zeros((observeNum, tagNum), dtype=int)    #第一行用于初始化,path[i][j]表示从开始到第i个观测对应第j个词性概率最大时i-1个观测的词性索引值

            transport_matrix = np.log(self.transition_mat)  #对数处理后，点乘运算变为加法运算
            launch_matrix = np.log(self.emission_mat)

            path[0] = -1
            max_p[0] = transport_matrix[-1,:-1] + launch_matrix[:,word_index[0]]
            for i in range(1, observeNum):
                probs = transport_matrix[:-1,:-1] + max_p[i-1].reshape(-1,1) + launch_matrix[:,word_index[i]]   #!这一步是关键
                max_p[i] = np.max(probs, axis=0)
                path[i] = np.argmax(probs, axis=0)

            max_p[-1] += transport_matrix[:-1,-1]

            step = np.argmax(max_p[-1])
            gold_path = [step]
            # backoff
            for i in range(observeNum-1, 0, -1):
                step = path[i][step]
                gold_path.insert(0, step)
            paths.append([self.id2tag[x] for x in gold_path])
            if cnt % 100 == 0:
                print('Process: {}/{}\r'.format(cnt, len(sentences)), end='')
            cnt += 1
        return paths

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
