# -*- coding: utf-8 -*-

import os
import re
import collections
import json
import traceback
from typing import List

from nlp_tasks.common import common_path
from nlp_tasks.utils import file_utils


class KeyInstance:
    """

    """

    def __init__(self, text: str, from_index: int, to_index: int, polarity: str, category: str):
        self.text = text
        self.from_index = from_index
        self.to_index = to_index
        self.polarity = polarity
        self.category = category
        self.metadata = {}

    def __str__(self):
        return self.text


class Sentence:
    """

    """

    def __init__(self, text: str, categories: List[str], key_instances: List[KeyInstance]):
        self.text = text
        self.categories = categories
        self.key_instances = key_instances


class Mil:
    """

    """

    def __init__(self, configuration: dict=None):
        self.configuration = configuration

    def _get_category(self, category: str):
        return category

    def _load_samples_by_filepath(self, filepath):
        with open(filepath, encoding='utf-8') as input_file:
            json_dict = json.load(input_file)
            samples = []
            for e in json_dict:
                text = e['text']
                categories = e['categories']
                key_instance_dicts = e['key_instances']
                key_instances: List[KeyInstance] = []
                for key_instance_dict in key_instance_dicts:
                    key_instance_text = key_instance_dict['key_instance']
                    from_index = int(key_instance_dict['from'])
                    to_index = int(key_instance_dict['to'])
                    polarity = key_instance_dict['polarity']
                    category = self._get_category(key_instance_dict['category'])
                    key_instance = KeyInstance(key_instance_text, from_index, to_index, polarity, category)
                    key_instances.append(key_instance)
                sentence = Sentence(text, categories, key_instances)
                samples.append(sentence)
        return samples

    def load_samples(self) -> List[Sentence]:
        pass


class SemEval2014Task4RESTMil(Mil):
    """
    SemEval-2014-Task-4-REST test
    """

    def load_samples(self):
        base_dir = common_path.get_task_data_dir('absa', is_original=True)

        filepath = os.path.join(base_dir, 'SemEval-2014-Task-4-REST', 'SemEval-2014-Task-4-REST-mil',
                                'SemEval-2014-Task-4-REST-mil.json')
        samples = super()._load_samples_by_filepath(filepath)
        return samples

    def _get_category(self, category: str):
        return 'anecdotes/miscellaneous' if category == 'anecdotes_miscellaneous' else category


class SemEval2014Task4RESTHardMil(Mil):
    """
    SemEval-2014-Task-4-REST-hard test
    """

    def load_samples(self):
        base_dir = common_path.get_task_data_dir('absa', is_original=True)

        filepath = os.path.join(base_dir, 'SemEval-2014-Task-4-REST', 'SemEval-2014-Task-4-REST-mil',
                                'SemEval-2014-Task-4-REST-mil.json')
        samples = super()._load_samples_by_filepath(filepath)
        filepath_hard = os.path.join(base_dir, 'SemEval-2014-Task-4-REST', 'SemEval-2014-Task-4-REST-mil',
                        'SemEval-2014-Task-4-REST-hard-mil.txt')
        hard_sentences = set(file_utils.read_all_lines(filepath_hard))
        result = [sample for sample in samples if sample.text.lower() in hard_sentences]
        return result

    def _get_category(self, category: str):
        return 'anecdotes/miscellaneous' if category == 'anecdotes_miscellaneous' else category


class MAMSACSAMil(Mil):
    """
    MAMSACSA
    """

    def load_samples(self):
        base_dir = common_path.get_task_data_dir('absa', is_original=True)

        filepath = os.path.join(base_dir, 'MAMS-for-ABSA', 'MAMS-ACSA-mil',
                                'AMS-ACSA-mil.json')
        samples = super()._load_samples_by_filepath(filepath)
        return samples

    def _get_category(self, category: str):
        return 'miscellaneous' if category == 'anecdotes_miscellaneous' else category


if __name__ == '__main__':
    mil = MAMSACSAMil()
    samples = mil.load_samples()
    print()
