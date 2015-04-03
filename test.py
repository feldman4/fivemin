import fivemin
import os
import glob

path = os.path.dirname(__file__)

test_files = glob.glob(path + '/test*.csv')

fivemin.test()

# for filename in test_files:
#     print filename
#     fivemin.test(filename=filename)