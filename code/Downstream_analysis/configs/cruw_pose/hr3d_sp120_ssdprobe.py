import os

_base_path = os.path.join(os.path.dirname(__file__), 'hr3d_sp120.py')
with open(_base_path, 'r') as _f:
    exec(compile(_f.read(), _base_path, 'exec'))

data['train']['label_file'] = 'Train_sp120_ssdprobe.json'
data['val']['label_file'] = 'Train_sp120_ssdprobe.json'
data['test']['label_file'] = 'Train_sp120_ssdprobe.json'

total_epochs = 1
work_dir = './work_dirs/hr3d_sp120_ssdprobe'
