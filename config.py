__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


from collections import OrderedDict


import torch as T


def get_device_setting():
    return T.device('cuda') if T.cuda.is_available() else T.device('cpu')

data_config = OrderedDict({
    'bookcorpus_path': './all.txt'
#    'bookcorpus_path': '/Users/judepark/Documents/toy_projects/bookcorpus/books_large_p1.txt'
})

