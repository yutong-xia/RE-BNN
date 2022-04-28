import yaml
import sys
import os
import numpy as np


proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
conf_fp = os.path.join(proj_dir, 'config.yaml')
with open(conf_fp) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


# nodename = os.uname().nodename
# nodename = 'schwarzhorn.d2.comp.nus.edu.sg'
# file_dir = config['filepath'][nodename]

# file_dir = 'ssh://yutong@schwarzhorn.d2.comp.nus.edu.sg:/home/yutong/'

def main():
    pass


if __name__ == '__main__':
    main()
