# -*- coding: utf-8 -*-


from nlp_tasks.utils import tokenizers


def to_english_like_sentence(sentence: str, tokenizer = tokenizers.JiebaTokenizer()):
    """

    :param sentence:
    :return:
    """
    return ' '.join(tokenizer(sentence))
