import os

_base_path = os.path.join(os.path.dirname(__file__), 'hr3d_one_hm_doppler.py')
with open(_base_path, 'r') as _f:
    exec(compile(_f.read(), _base_path, 'exec'))

# Local RT-Pose single-person 120-sequence experiment.
# Keep the HRRadarPose/one-heatmap Doppler architecture and training recipe from
# hr3d_one_hm_doppler.py, but point it to the generated local tensors/splits.
BATCH_SIZE = 8
data['samples_per_gpu'] = BATCH_SIZE
data['workers_per_gpu'] = 8
data['prefetch_factor'] = 2
test_samples_per_gpu = 16
test_workers_per_gpu = 4

DATASET['DIR']['ROOT_DIR'] = str(REPO_ROOT / 'datasets')
DATASET['DIR']['META_FILE'] = 'filemeta.json'
DATASET['DIR']['KEYPOINT_META'] = 'Keypoints_meta.txt'
DATASET['DIR']['SEQUENCES_DIR'] = 'GT_sequences'
DATASET['DIR']['RADAR'] = 'radar'
DATASET['DIR']['RDR_CUBE_DIR'] = 'npy_DZYX_complex'
DATASET['DIR']['RDR_CACHE_DIR'] = 'npy_DZYX_mag_roi_f16_norm'
DATASET['DIR']['RDR_CACHE_ROOT'] = '/ssdtemp/users/quansj/rtpose_cache_local'
DATASET['DZYX']['DOPPLER_SLICE'] = [16, 48]
DATASET['DZYX']['LOAD_FLOAT32'] = True

data['train']['label_file'] = 'Train_sp120_train_minus_val6.json'
data['val']['label_file'] = 'Val_sp120_by_motion.json'
data['test']['label_file'] = 'Test_sp120_by_motion6.json'

# The paper baseline uses one-cycle Adam and 100 epochs in this config.  Keep
# those settings, while reducing noisy logging and saving/evaluating each epoch.
log_config['interval'] = 200
log_config['hooks'] = [dict(type="TextLoggerHook")]
checkpoint_config['interval'] = 3
write_iter_loss_csv = False
eval_epoch_interval = 1
val_eval_epoch_interval = 3
test_eval_epoch_interval = 3
cudnn_benchmark = True
disable_cudnn = False
find_unused_parameters = False
work_dir = './work_dirs/hr3d_one_hm_doppler_sp120'
