# -*- coding: utf-8 -*-


from nlp_tasks.utils import file_utils


def get_data_from_file(pos_filepath, neg_filepath):
    """

    :param pos_filepath:
    :param neg_filepath:
    :return:
    """
    neg = file_utils.read_all_lines(neg_filepath)
    pos = file_utils.read_all_lines(pos_filepath)
    data = []
    for d in neg:
        data.append((d, 0))
    for d in pos:
        data.append((d, 1))
    return data


def get_pair_data_from_file(filepath):
    """
    :param filepath:
    :return:
    """
    lines = file_utils.read_all_lines(filepath)
    samples = [line.split('\t') for line in lines]
    samples = [(sample[0], sample[1], int(sample[2])) for sample in samples]
    return samples