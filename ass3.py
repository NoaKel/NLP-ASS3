import sys
import math
import numpy as np
from collections import Counter

MIN_WORD_FREQUENCY = 100
MIN_FEATURE_FREQUENCY = 20
MIN_MUTUAL_FREQUENCY = 3
target_words = ["car", "bus", "hospital", "hotel", "gun", "bomb", "horse", "fox", "table", "bowl", "guitar", "piano"]

class distributional_semantic:
    def __init__(self, filePath):
        """
        parses file
        :param filePath: file path
        """
        self.sentences = []
        with open(filePath) as file:
            cur_sentence = []
            for line in file.readlines():
                ls = line.split()
                if len(line) == 1:
                    self.sentences.append(cur_sentence)
                    cur_sentence = []
                else:
                    cur_sentence.append((ls[2], ls[3], ls[6]))

        self.vocab = set([item for sublist in self.sentences for item, _, _ in sublist])
        self.W2I = {word: i for i, word in enumerate(self.vocab)}
        self.I2W = {i: word for i, word in enumerate(self.vocab)}
        self.lemma_count = Counter([self.W2I[item] for sublist in self.sentences for item, _, _ in sublist])
        self.target_words_idx = [self.W2I[word] for word in target_words]
        print "num of words:" , len(Counter(el for el in self.lemma_count.elements() if self.lemma_count[el] >= MIN_WORD_FREQUENCY))
    def same_sentence_co(self):
        """
        type 1 co-occur in sentence
        :return: word attr counts
        """
        c_t = {}
        attr_map = {}
        for sentence in self.sentences:
            for i in range(0, len(sentence)):
                word_idx = self.W2I[sentence[i][0]]
                if self.lemma_count[word_idx] >= MIN_WORD_FREQUENCY:
                    if word_idx not in c_t:
                        c_t[word_idx] = {}
                    for j in range(0,len(sentence)):
                        if i != j:
                            attr_idx = self.W2I[sentence[j][0]]
                            if self.lemma_count[attr_idx] >= MIN_FEATURE_FREQUENCY:
                                if attr_idx not in c_t[word_idx]:
                                    c_t[word_idx][attr_idx] = 0
                                c_t[word_idx][attr_idx] += 1
                                if attr_idx not in attr_map:
                                    attr_map[attr_idx] = set()
                                attr_map[attr_idx].add(word_idx)
        print "sentence-co feature num:", len(attr_map)
        return c_t, attr_map

    def window_co(self):
        """
        type 2 co-occur in window
        :return: word attr counts
        """
        functional_words = set(['IN', 'PRP', 'PRP$', 'MD', 'POS', 'WDT', 'PDT', 'DT', 'CC', 'RP', 'TO', ',', '.', '(', ')', ';'])
        c_t = {}
        attr_map = {}

        for sentence in self.sentences:
            no_fw_sentence = []
            for word, tag, _ in sentence:
                if tag not in functional_words:
                    no_fw_sentence.append(word)
            for i in range(0, len(no_fw_sentence)):
                word_idx = self.W2I[no_fw_sentence[i]]
                if self.lemma_count[word_idx] >= MIN_WORD_FREQUENCY:
                    if word_idx not in c_t:
                        c_t[word_idx] = {}
                    for j in {-2,-1,1,2}:
                        if (i + j >= 0) and (i + j < len(no_fw_sentence)):
                            attr_idx = self.W2I[no_fw_sentence[i+j]]
                            if self.lemma_count[attr_idx] >= MIN_FEATURE_FREQUENCY:
                                if attr_idx not in c_t[word_idx]:
                                    c_t[word_idx][attr_idx] = 0
                                c_t[word_idx][attr_idx] += 1
                                if attr_idx not in attr_map:
                                    attr_map[attr_idx] = set()
                                attr_map[attr_idx].add(word_idx)
        print "window-co feature num:" , len(attr_map)
        return c_t, attr_map

    def dependency_co(self):
        """
        type 3 - dependency co-occur
        :return: word attr counts
        """
        c_t = {}
        attr_map = {}
        for sentence in self.sentences:
            for word, word_tag, word_head in sentence:
                word_idx = self.W2I[word]
                if self.lemma_count[word_idx] >= MIN_WORD_FREQUENCY:
                    if word_idx not in c_t:
                        c_t[word_idx] = {}
                    if sentence[int(word_head)-1][1] == "IN": # preposition
                        p_node = int(sentence[int(word_head)-1][2]) -1
                        attr_idx = self.W2I[sentence[p_node][0]]
                        att_tag = sentence[p_node][1]
                        direction = "C"
                        op_direction = "P"
                    else: # all other types
                        attr_idx = self.W2I[sentence[int(word_head)-1][0]]
                        att_tag = sentence[int(word_head)-1][1]
                        direction = "P"
                        op_direction = "C"
                    # insert relation
                    if self.lemma_count[attr_idx] >= MIN_FEATURE_FREQUENCY:
                        if (attr_idx, direction, att_tag) not in c_t[word_idx]:
                            c_t[word_idx][(attr_idx, direction, att_tag)] = 0
                        c_t[word_idx][(attr_idx, direction, att_tag)] += 1
                        if (attr_idx, direction, att_tag) not in attr_map:
                            attr_map[(attr_idx, direction, att_tag)] = set()
                        attr_map[(attr_idx, direction, att_tag)].add(word_idx)
                    # insert opposite relation
                    if self.lemma_count[attr_idx] >= MIN_WORD_FREQUENCY:
                        if attr_idx not in c_t:
                            c_t[attr_idx] = {}
                        if self.lemma_count[attr_idx] >= MIN_FEATURE_FREQUENCY:
                            if (word_idx, op_direction, word_tag) not in c_t[attr_idx]:
                                c_t[attr_idx][(word_idx, op_direction, word_tag)] = 0
                            c_t[attr_idx][(word_idx, op_direction, word_tag)] += 1
                            if (word_idx, op_direction, word_tag) not in attr_map:
                                attr_map[(word_idx, op_direction, word_tag)] = set()
                            attr_map[(word_idx, op_direction, word_tag)].add(attr_idx)
        print "dependency feature num:" , len(attr_map)
        return c_t , attr_map

    def similarity(self, pmi, attr_map):
        DT = {}
        count_features = {}

        for word in self.target_words_idx:
            DT[word], count_features[word] = cosine(word, pmi, attr_map)

        for word in DT:
            k = 20
            sorted_attributes = sorted([attr for attr in DT[word] if count_features[word][attr] > MIN_MUTUAL_FREQUENCY],
            key=lambda attr: DT[word][attr], reverse=True)
            sorted_pmi = sorted([attr for attr in pmi[word]], key=lambda attr: pmi[word][attr], reverse=True)
            print "target word: %s" % self.I2W[word]
            print ', '.join("%s" % (self.I2W[attr[0]]) if type(attr)==tuple else self.I2W[attr] for attr in sorted_pmi[:k])
            print ', '.join("%s" % (self.I2W[attr[0]]) if type(attr)==tuple else self.I2W[attr] for attr in sorted_attributes[:k])

    def calc_all(self):
        ### type 1 - whole sentence co-occur
        print " ***** type 1: sentence co-occur ***** "
        c_t1, attr_map1 = ds.same_sentence_co()
        pmi1 = pmi(c_t1)
        self.similarity(pmi1, attr_map1)

        ### type 2 - window co-occur
        print " ***** type 2: word window ***** "
        c_t2, attr_map2 = ds.window_co()
        pmi2 = pmi(c_t2)
        self.similarity(pmi2, attr_map2)

        ### type 3 - dependency
        print " ***** type 3: dependency ***** "
        c_t3, attr_map3 = ds.dependency_co()
        pmi3 = pmi(c_t3)
        self.similarity(pmi3, attr_map3)

def pmi(c_t):
    """
    calculates negative sampling smoothing pmi for each word, attr pair
    :param c_t: word attr counts
    :return: pmi
    """
    p_u_att = {}
    p_u = {}
    p_att = {}
    pmi = {}
    sum_all_pairs = sum(sum(c.values()) for c in c_t.values())
    for word in c_t:
        if word not in p_u_att:
            p_u_att[word] = {}
        if word not in p_u:
            p_u[word] = 0
        for attr in c_t[word]:
            if attr not in p_att:
                p_att[attr] = 0
            p_u_att[word][attr] = float(c_t[word][attr])
            p_u[word] += float(c_t[word][attr])
            p_att[attr] += float(c_t[word][attr])

    for word in c_t:
        if word not in pmi:
            pmi[word] = {}
        for attr in c_t[word]:
            pmi[word][attr] = max(math.log(((p_u_att[word][attr] * sum_all_pairs) / ((p_u[word]**0.75) * p_att[attr])) + 1, 2),0)

    return pmi

def size(vector):
    """
    vector size
    :param vector: vector
    :return: vector size
    """
    total = 0.0
    for word in vector:
        total += np.sqrt(vector[word] * vector[word])
    return total

def cosine(u, pmi, attr_map):
    """
    coisne similarity with efficient dot product calculation
    :param u: word
    :param pmi: pmi vector
    :param attr_map: maps all attr to possible words
    :return: DT
    """
    DT = {}
    count_features = {}
    for attr in pmi[u]:
        for v in attr_map[attr]:
            if v == u:
                continue
            if v not in count_features:
                count_features[v] = 0
            if v not in DT:
                DT[v] = 0
            DT[v] += pmi[u][attr] * pmi[v][attr]
            count_features[v] += 1
    for v in DT:
        DT[v] /= (size(pmi[u]) * (size(pmi[v])))

    return DT, count_features


def most_similar(DT, k):
    """
    return k most similar words
    :param DT: cosine similarity
    :param k: num of top similar words
    :return: k most similar words
    """
    words = sorted([attr for attr in DT], key=lambda attr: DT[attr], reverse=True)
    return words[:k]

if __name__ == '__main__':
    ds = distributional_semantic(sys.argv[1])
    ds.calc_all()
