import pandas as pd
from nltk.corpus import wordnet
from sklearn.feature_extraction import text
stop_words = set(text.ENGLISH_STOP_WORDS)
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import random
import nlpaug.flow as nafc
from nlpaug.util import Action
import csv
import os

from nlpaug.util.file.download import DownloadUtil
DownloadUtil.download_word2vec(dest_dir='./nlp_model/')  # Download word2vec model
DownloadUtil.download_glove(model_name='glove.6B', dest_dir='./nlp_model/')  # Download GloVe model
DownloadUtil.download_fasttext(model_name='wiki-news-300d-1M', dest_dir='./nlp_model/')  # Download fasttext model


def replace_synonym(text):
    aug = naw.SynonymAug(aug_src='wordnet')
    augmented_text = aug.augment(text)
    return augmented_text


def replace_word2vec(text):
    aug = naw.WordEmbsAug(
        model_type='word2vec', model_path=model_dir + 'GoogleNews-vectors-negative300.bin',
        action="substitute")
    augmented_text = aug.augment(text)
    return augmented_text


def back_translate(text):
    back_translation_aug = naw.BackTranslationAug(
        from_model_name='transformer.wmt19.en-de',
        to_model_name='transformer.wmt19.de-en', device='cuda'
    )
    augmented_text = back_translation_aug.augment(text)
    return augmented_text


def bert_insert(text):
    aug = naw.ContextualWordEmbsAug(
        model_path='bert-base-uncased', action="insert")
    augmented_text = aug.augment(text)
    return augmented_text


def random_aug1(sentence):
    augmentation = ['synonym_replace', 'word2vec_replace', 'back_translate', 'bert_insert']
    choose_aug = random.choice(augmentation)
    return aug_sentence(sentence, choose_aug)


def aug_sentence(sentence, choose_aug):
    aug = ""
    if choose_aug == "synonym_replace":
        aug = replace_synonym(sentence)
    elif choose_aug == "word2vec_replace":
        aug = replace_word2vec(sentence)
    elif choose_aug == "back_translate":
        aug = back_translate(sentence)
    elif choose_aug == "bert_insert":
        aug = bert_insert(sentence)
    return aug


if __name__ == '__main__':
    model_dir = "./nlp_model/"
    os.makedirs("./dau/snips_harder", exist_ok=True)
    data = pd.read_csv("../new_intent.csv")
    to_be_aug_data = data[:500]
    # selected_data = to_be_aug_data.sample(n=1000)

    selected_data = to_be_aug_data.reset_index(drop=True)
    print(len(selected_data))
    # print(selected_data)

    aug_data = pd.DataFrame(columns=('text', 'intent'))
    idx = 0

    aug_text_li = []
    aug_intent_li = []
    ori_text_li = []
    ori_intent_li = []

    for i in range(len(selected_data)):
        ori_text = selected_data.text[i]
        ori_intent = selected_data.intent[i]

        aug_text_li.append(random_aug1(ori_text))
        aug_intent_li.append(ori_intent)

        ori_text_li.append(ori_text)
        ori_intent_li.append(ori_intent)

        print("processed:", i)

    for text, intent in zip(ori_text_li, ori_intent_li):
        aug = {'text': text, 'intent': intent}
        aug_data.loc[idx] = aug
        idx = idx + 1

    for text, intent in zip(aug_text_li, aug_intent_li):
        aug = {'text': text, 'intent': intent}
        aug_data.loc[idx] = aug
        idx = idx + 1

    aug_data.to_csv("./dau/snips_harder/snips_toselect.csv")
