# -*- coding: utf-8 -*-


from nlp_tasks.utils import my_corenlp
from nlp_tasks.utils import corenlp_factory


class CorenlpParser:
    def __init__(self, nlp: my_corenlp.StanfordCoreNLP):
        self.nlp = nlp

    def build_parse_child_dict(self, postags, arcs):
        """

        """
        format_parse_list = [[] for _ in range(len(postags))]
        child_dict_list = [{} for _ in range(len(postags))]
        for element in arcs:
            relation, start_index, end_index = element
            start_index -= 1
            end_index -= 1
            current_word = postags[end_index][0]
            current_word_pos = postags[end_index][1]
            head_word = postags[start_index][0] if start_index != -1 else 'ROOT'
            head_word_pos = postags[start_index][1] if start_index != -1 else ''
            format_parse_list[end_index] = [relation, current_word, end_index, current_word_pos,
                                            head_word, start_index, head_word_pos]
            if start_index == -1:
                continue
            if relation not in child_dict_list[start_index]:
                child_dict_list[start_index][relation] = []
            child_dict_list[start_index][relation].append(end_index)
        return child_dict_list, format_parse_list

    def parser_main(self, sentence):
        """

        :param sentence:
        :return:
        """
        words = self.nlp.word_tokenize(sentence)
        postags = self.nlp.pos_tag(sentence)
        arcs = self.nlp.dependency_parse(sentence)
        child_dict_list, format_parse_list = self.build_parse_child_dict(postags, arcs)
        return words, postags, arcs, child_dict_list, format_parse_list
