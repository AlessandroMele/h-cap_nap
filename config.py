from os.path import join, dirname
from os import getcwd, listdir, makedirs, cpu_count
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

DIR_PATH = dirname(join(getcwd(), __file__))
INPUT_PATH = join(DIR_PATH, 'dataset')
LOG_NAME = sorted([item.split('_processed.g')[0] for item in listdir(INPUT_PATH)
                   if item.endswith('_processed.g')])
MAX_WORKERS = cpu_count() - 3

EPOCHS = 150
N_TRIALS = 20
EARLY_STOP = 15
