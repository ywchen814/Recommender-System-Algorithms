import os
print(__name__)
dataset = 'Beibei'
if not os.path.exists('result'):
    os.mkdir(f'result')
if not os.path.exists('result/{dataset}'):
    os.mkdir(f'result/{dataset}')
