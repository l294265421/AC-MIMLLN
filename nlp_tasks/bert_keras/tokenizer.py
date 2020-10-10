# -*- coding: utf-8 -*-


from keras_bert import Tokenizer


class TokenizerReturningSpace(Tokenizer):
    """

    """
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')
            else:
                R.append('[UNK]')
        return R


class EnglishTokenizer(Tokenizer):
    """

    """
    pass
