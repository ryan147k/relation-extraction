from .dataset import Sentence
from .dataset import sentence2data

# import os
# with open(os.path.join(os.path.dirname(__file__), 'train2/train_data.txt'), 'r', encoding='utf-8') as f:
#     i = 1
#     line = f.readline()
#     while(line):
#         z = line.split('\t')
#         if len(z) != 4:
#             print(i)
#             pass
#         line = f.readline()
#         i += 1