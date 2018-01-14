import sys
import numpy as np
target_words = ["car", "bus", "hospital", "hotel", "gun", "bomb", "horse", "fox", "table", "bowl", "guitar", "piano"]

def most_similar(W, C, W2I, word, vocab, k):
    print "target word: %s" % word
    emb = W[W2I[word]]
    sim = C.dot(emb)
    most_similar_idx = (-sim).argsort()
    sim_words = vocab[most_similar_idx]
    print ', '.join(sim_words[1:k+1])

def calc_all(word_file_path, context_file_path):
    print "second order similarity:"
    with open(word_file_path) as vector_file:
        a = len(vector_file.readline().split(' '))
    W = np.loadtxt(word_file_path, delimiter=' ', usecols=range(1, a))
    vocab = np.loadtxt(word_file_path, delimiter=' ', usecols=[0], dtype=str)
    W2I = {word: i for i, word in enumerate(vocab)}
    k = 20
    for word in target_words:
        most_similar(W, W, W2I, word, vocab, k)

    print "first order similarity:"
    with open(context_file_path) as s_file:
        b = len(s_file.readline().split(' '))
    C = np.loadtxt(context_file_path, delimiter=' ', usecols=range(1, b))
    attr = np.loadtxt(context_file_path, delimiter=' ', usecols=[0], dtype=str)
    q = 10
    for word2 in target_words:
        most_similar(W, C, W2I, word2, attr, q)

if __name__ == "__main__":
    calc_all(sys.argv[1],sys.argv[2])
