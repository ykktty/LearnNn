import torch
import jieba
import torch.nn as nn

# 1.获取数据 进行分词 获取词表
def build_vocab():
    #去重后的所有词，每行文本分词结果
    unique_words , all_words = [] ,[]
    for line in open('./datasets/jaychou_lyrics.txt','r',encoding='utf-8'):
        words = jieba.lcut(line)
        all_words.append( words)
        for word in words:
            if word not in unique_words:
                unique_words.append(word)

    # print(len(unique_words))
    # print(unique_words)
    # print(all_words)

    word2id = {wrod:i for i,wrod in enumerate(unique_words)}
    corpus_idx = []
    for words in all_words:
        tmp = []
        for word in words:
            tmp.append(word2id[word])
        tmp.append(word2id[' '])
        corpus_idx.extend(tmp)
        #corpus_idx.append(tmp)
    #去重后的所有词 词表 词个数 索引表述歌词表
    return unique_words,word2id,len(unique_words),corpus_idx






if  __name__ == '__main__':
    unique_words,word2id,vocab_size,corpus_idx = build_vocab()
    print(f'unique_words: {unique_words}, word2id: {word2id}, vocab_size: {vocab_size}, corpus_idx: {corpus_idx}')