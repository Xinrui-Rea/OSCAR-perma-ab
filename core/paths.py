import os
from pathlib import Path

##====================
##====================

## path base folder
path_base = Path(__file__).resolve().parent.parent

## path for input data
path_data =  os.path.join(path_base, 'input_data/')
