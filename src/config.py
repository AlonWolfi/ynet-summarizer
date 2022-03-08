from numpy.random import seed as set_random_seed

set_random_seed(42)

PROJECT_NAME = 'ynet-summerizer'

DEBUG = False

import os
from pathlib import Path

# import os
# PROJECT_DIR = Path(os.path.abspath(os.curdir))
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / 'data'
TMP_DIR = PROJECT_DIR / 'tmp'
SRC_DIR = PROJECT_DIR / 'src'
RESULTS_DIR = PROJECT_DIR / 'results'
NATS_DIR = os.path.join('..', 'nats_results')
