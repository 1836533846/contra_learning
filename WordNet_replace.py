import random
# 使用下载的wordNet数据
from nltk.corpus import wordnet as wn
from nltk import pos_tag

def synonym_replacement(sentence, n=1):
    words = sentence.split()
    pos_tagged_words = pos_tag(words)

    for _ in range(n):
        # 随机选择一个单词
        word_index = random.randint(0, len(words) - 1)
        word = words[word_index]
        pos_tagged_word = pos_tagged_words[word_index]

        # 获取该单词的同义词
        synonyms = get_synonyms(word, pos_tagged_word[1])

        # 如果找到同义词，则用同义词替换原始单词
        if len(synonyms) > 0:
            synonym = random.choice(synonyms)
            words[word_index] = synonym
            pos_tagged_words = pos_tag(words)

    # 将替换后的单词列表重新组合成句子
    new_sentence = " ".join(words)
    return new_sentence

def get_synonyms(word, pos_tag):
    # 将POS tag转换为WordNet可识别的形式
    wordnet_tag = convert_pos_tag(pos_tag)
    if wordnet_tag is None:
        return []

    # 获取同义词集
    synsets = wn.synsets(word, pos=wordnet_tag)
    synonyms = []

    # 获取所有同义词
    for synset in synsets:
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())

    # 去除原始单词
    if word in synonyms:
        synonyms.remove(word)

    return list(set(synonyms))

def convert_pos_tag(pos_tag):
    if pos_tag.startswith('N'):
        return wn.NOUN
    elif pos_tag.startswith('V'):
        return wn.VERB
    elif pos_tag.startswith('R'):
        return wn.ADV
    elif pos_tag.startswith('J'):
        return wn.ADJ
    else:
        return None

# 示例
sentence = "The quick brown fox jumps over the lazy dog."
new_sentence = synonym_replacement(sentence)
print("Original sentence:", sentence)
print("New sentence:", new_sentence)
