import os
import numpy as np


subjects = [ f.name for f in os.scandir('/root/autodl-tmp/data/BraTS2021_ValidationData') if f.is_dir() ]
with open('/root/autodl-tmp/data/official_valid.txt', 'w') as f:
    for item in subjects:
        f.write("%s\n" % item)
