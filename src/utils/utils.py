import json
import os
import pickle
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import Figure

from config import PROJECT_NAME


def __get_project_dir() -> Path:
    '''
    @return: The path of the project
    '''
    current_path = Path.cwd()
    while current_path.name != PROJECT_NAME:
        current_path = current_path.parent
    return current_path


PROJECT_DIR = __get_project_dir()


def validate_path(file_path: Union[str, Path]):
    for folder in list(Path(file_path).parents)[::-1]:
        try:
            os.stat(folder)
        except:
            os.mkdir(folder)


def loop_through_iterable(iterable, func_for_ins):
    if type(iterable) == dict:
        outputs = dict()
        for k, v in iterable.items():
            outputs[k] = loop_through_iterable(v, func_for_ins)
        return outputs
    elif type(iterable) == list:
        outputs = list()
        for ele in iterable:
            outputs.append(loop_through_iterable(ele, func_for_ins))
        return outputs
    elif type(iterable) == set:
        outputs = set()
        for ele in iterable:
            outputs.add(loop_through_iterable(ele, func_for_ins))
        return outputs
    else:
        return func_for_ins(iterable)


def is_picklable(obj):
    try:
        pickle.dumps(obj)
    except pickle.PicklingError:
        return False
    return True


def save_data(data, file_path: Union[str, Path], encoding: str = "utf-8"):
    '''
    Saves data to file
    '''

    if type(file_path) == str:
        file_path = PROJECT_DIR / file_path

    validate_path(file_path)

    if file_path.suffix == '.pickle':
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
    elif file_path.suffix == '.json':
        with open(file_path, 'wb') as file:
            json.dump(data, file)
    elif type(data) == Figure:
        data.save_fig(file_path)
    else:
        with open(file_path, 'w+', encoding=encoding) as file:
            file.write(data)


def read_data(file_path: Union[str, Path, list, dict, set], encoding: str = "utf-8"):
    """
    Saves data to file_path
    @param file_path:
    @param encoding:
    @return:
    """

    if type(file_path) in [list, dict, set]:
        return loop_through_iterable(file_path, read_data)

    if type(file_path) == str:
        file_path = PROJECT_DIR / file_path

    validate_path(file_path)

    data = None

    if not os.path.exists(file_path):
        warnings.warn(f'   File not found: {file_path}', Warning)
        return None

    if file_path.suffix == '.pickle':
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
    elif file_path.suffix == '.json':
        with open(file_path, 'rb') as file:
            data = json.load(file)
    elif file_path.suffix == '.jpg' or file_path.suffix == '.png':
        data = plt.imread(file_path, format='PNG')
    else:
        with open(file_path, 'r+', encoding=encoding) as file:
            data = file.read()
    return data


def multiprocess(l, f, n_jobs=-1, as_series: bool = False):
    with Pool(n_jobs) as p:
        output = p.map(f, l)
    if as_series:
        return pd.Series(output)
    else:
        return output
