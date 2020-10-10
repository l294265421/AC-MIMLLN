import os

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

original_data_dir = project_dir + '/original_data/'

common_data_dir = project_dir + '/data/'
common_code_dir = project_dir + '/nlp_tasks/'


def get_task_data_dir(task_name: str, is_original=False):
    """
    """
    if not is_original:
        return '%s%s/data/' % (common_data_dir, task_name)
    else:
        return '%s%s/' % (original_data_dir, task_name)
