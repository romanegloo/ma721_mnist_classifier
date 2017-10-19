import os

cur_dir = os.path.dirname(os.path.realpath(__file__))
DEFAULTS = {
    'DATA_DIR': os.path.join(cur_dir, '../data'),
    'MODEL_DIR': os.path.join(cur_dir, '../model')
}

from . import models
from . import data
