#!/usr/bin/python3
# -*- coding:utf-8 -*-
#   ____    ____    ______________
#  |    |  |    |  |              |
#  |    |  |    |  |_____    _____|
#  |    |__|    |       |    |
#  |     __     |       |    |
#  |    |  |    |       |    |
#  |    |  |    |       |    |
#  |____|  |____|       |____|
#
# fileName:word2vec_tool 
# project: Fish_learning
# author: theo_hui
# e-mail:Theo_hui@163.com
# purpose: {文件作用描述｝
# creatData:2019/5/16
import multiprocessing
import numpy as np

from gensim.models import Word2Vec


def embedding_sentences(sentences, embedding_size=128, window=5, min_count=5, file_to_load=None, file_to_save=None):
    if file_to_load is not None:
        w2vModel = Word2Vec.load(file_to_load)
    else:
        print("w2vec")
        w2vModel = Word2Vec(sentences, size=embedding_size, window=window, min_count=min_count,
                            workers=multiprocessing.cpu_count())


        if file_to_save is not None:
            w2vModel.save(file_to_save)
    all_vectors = []
    embeddingDim = w2vModel.vector_size
    embeddingUnknown = [0 for i in range(embeddingDim)]

    for sentence in sentences:
        print("processing "+str(sentence))

        this_vector = []
        for word in sentence:
            if word in w2vModel.wv.vocab:
                this_vector.append(w2vModel[word])
            else:
                this_vector.append(embeddingUnknown)

        if np.array(this_vector).shape != (10,10):
            print(np.array(this_vector).shape)
        all_vectors.append(this_vector)

    return all_vectors