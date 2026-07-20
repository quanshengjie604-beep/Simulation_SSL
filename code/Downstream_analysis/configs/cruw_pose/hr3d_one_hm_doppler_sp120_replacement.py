import os

_base_path = os.path.join(os.path.dirname(__file__), 'hr3d_one_hm_doppler_sp120.py')
with open(_base_path, 'r') as _f:
    exec(compile(_f.read(), _base_path, 'exec'))

# Replacement experiments use the same architecture/splits as the SP120 one-HM
# Doppler baseline, but train for 20 epochs and evaluate the test split every
# epoch. Validation is intentionally disabled for these runs.
DATASET['DIR']['RDR_CACHE_ROOT'] = os.environ.get(
    'RTPOSE_CACHE_ROOT',
    '/ssdtemp/users/quansj/rtpose_cache_local',
)

if 'RTPOSE_WORKERS_PER_GPU' in os.environ:
    data['workers_per_gpu'] = int(os.environ['RTPOSE_WORKERS_PER_GPU'])
if 'RTPOSE_PREFETCH_FACTOR' in os.environ:
    data['prefetch_factor'] = int(os.environ['RTPOSE_PREFETCH_FACTOR'])
if 'RTPOSE_TEST_WORKERS_PER_GPU' in os.environ:
    test_workers_per_gpu = int(os.environ['RTPOSE_TEST_WORKERS_PER_GPU'])

total_epochs = int(os.environ.get('RTPOSE_TOTAL_EPOCHS', '20'))
eval_epoch_interval = 1
val_eval_epoch_interval = 0
test_eval_epoch_interval = 1
checkpoint_config['interval'] = 1

work_dir = os.environ.get(
    'RTPOSE_WORK_DIR',
    './work_dirs/hr3d_one_hm_doppler_sp120_replacement',
)
