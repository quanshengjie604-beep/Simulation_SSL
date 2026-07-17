import os

_base_path = os.path.join(os.path.dirname(__file__), 'hr3d.py')
with open(_base_path, 'r') as _f:
    exec(compile(_f.read(), _base_path, 'exec'))

BATCH_SIZE = 6
data['samples_per_gpu'] = BATCH_SIZE
data['workers_per_gpu'] = 8
data['prefetch_factor'] = 2
test_samples_per_gpu = 16
test_workers_per_gpu = 4
data['train']['label_file'] = 'Train_sp120_train_minus_val6.json'
data['val']['label_file'] = 'Val_sp120_by_motion.json'
data['test']['label_file'] = 'Test_sp120_by_motion6.json'

DATASET['DIR']['RDR_CACHE_DIR'] = 'npy_DZYX_mag_roi_f16_norm'
DATASET['DIR']['RDR_CACHE_ROOT'] = '/ssdtemp/users/quansj/rtpose_cache_local'

log_config['interval'] = 200
log_config['hooks'] = [dict(type="TextLoggerHook")]
checkpoint_config['interval'] = 1
write_iter_loss_csv = False

enable_amp = True
cudnn_benchmark = True
disable_cudnn = False
find_unused_parameters = True
eval_epoch_interval = 1
cuda_device = '0'
work_dir = './work_dirs/hr3d_sp120'
