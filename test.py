import fivemin
import os, glob

path = os.path.dirname(__file__)

test_files = glob.glob(path + '/test*.csv')

for filename in test_files:
    print filename
    fivemin.test(filename=filename)