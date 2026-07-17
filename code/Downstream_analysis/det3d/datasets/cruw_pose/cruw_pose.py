import os
import os.path as osp
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
from glob import glob
from tqdm import tqdm
import numpy as np
from det3d.datasets.registry import DATASETS
from det3d.datasets.pipelines import Compose
from munch import DefaultMunch
import collections
import json
from collections import defaultdict
from eval_util import *

@DATASETS.register_module
class CRUW_POSE_Dataset(Dataset):
    def __init__(self, cfg, label_file, class_names=None, pipeline=None, split='train'):
        super().__init__()
        cfg = DefaultMunch.fromDict(cfg)
        self.cfg = cfg
        self.split = split
        self.class_names = class_names
        self.cfg.update(class_names=class_names)
        self.enable_lidar, self.enable_radar = False, False 
        for sensor in cfg.DATASET.ENABLE_SENSOR:
            if sensor == 'LIDAR':
                self.enable_lidar = True
            elif sensor == 'RADAR':
                self.enable_radar = True

        if self.enable_radar:
            self.rdr_path_name = 'npy_DZYX_complex' if 'd' in self.cfg.DATASET.RDR_TYPE else 'npy'
            if 'zyx_real' in self.cfg.DATASET.RDR_TYPE:
                # Default ROI for CB (When generating CB from matlab applying interpolation)
                self.arr_z_cb = np.arange(-5.8, 5.8, 11.6/32)
                self.arr_y_cb = np.arange(-10.05, 10.05, 20.1/128)
                self.arr_x_cb = np.arange(0, 11.6, 11.6/256)
                self.is_consider_roi_rdr_cb = cfg.DATASET.RDR_CUBE.IS_CONSIDER_ROI
                if self.is_consider_roi_rdr_cb:
                    self.consider_roi_cube(cfg.DATASET.ROI[cfg.DATASET.LABEL['ROI_TYPE']])
                # self.rdr_to_real = True if 'd' in self.cfg.DATASET.RDR_TYPE else False
                self.rdr_to_real = False
            self.rad_normalize_values = cfg.DATASET.DZYX.NORMALIZING_VALUE if 'd' in self.cfg.DATASET.RDR_TYPE else cfg.DATASET.RDR_CUBE.NORMALIZING_VALUE
        if self.enable_lidar:
            self.read_calib('lidar')
        self.read_meta()
        self.label_file = label_file if os.path.isabs(label_file) else os.path.join(self.cfg.DATASET.DIR.ROOT_DIR, label_file)
        self.load_samples()
        if pipeline is None:
            self.pipeline = None
        else:
            self.pipeline = Compose(pipeline)
        self._set_group_flag()

    def _set_group_flag(self):
        self.flag = np.ones(len(self), dtype=np.uint8)

    def read_meta(self):
        seq_id_to_name = {}
        seq_id_to_activity = {}
        meta_path = os.path.join(self.cfg.DATASET.DIR.ROOT_DIR, self.cfg.DATASET.DIR.META_FILE)
        if meta_path.endswith('.json'):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            for seq_id, seq_meta in meta.items():
                if isinstance(seq_meta, dict):
                    seq_name = seq_meta.get('seq_name') or seq_meta.get('seq_name'.upper()) or seq_meta.get('name')
                    seq_id_to_name[seq_id] = seq_name or seq_id
                    seq_id_to_activity[seq_id] = seq_meta.get('Activity', 'UNKNOWN')
                else:
                    seq_id_to_name[seq_id] = str(seq_meta)
                    seq_id_to_activity[seq_id] = 'UNKNOWN'
        else:
            with open(meta_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                seq_id, seq_name = line.split(',', 1)
                if seq_id.lower() == 'seq_id':
                    continue
                seq_id_to_name[seq_id] = seq_name
                seq_id_to_activity[seq_id] = 'UNKNOWN'
        self.seq_id_to_name = seq_id_to_name
        self.seq_id_to_activity = seq_id_to_activity

    def read_calib(self, sensor_type):
        with open(os.path.join(self.cfg.DATASET.DIR.ROOT_DIR, self.cfg.DATASET.DIR.CALIB), 'r') as f:
            calib = json.load(f)
        if sensor_type == 'lidar':
            self.P_L2R = np.array(calib['radar']['extrinsic']).reshape(4, 4)
        elif sensor_type == 'left_cam':
            self.P_L2LC = np.array(calib['left_cam']['extrinsic']).reshape(4, 4)
            self.LC_I = np.array(calib['left_cam']['intrinsic']).reshape(3, 4)

    def load_samples(self):
        with open(self.label_file, 'r') as f:
            samples_by_seq = json.load(f)
        samples = []
        for seq, seq_frames in samples_by_seq.items():
            # TODO: remove the below line in the future
            if self.seq_id_to_name[seq] in ['2023_0718_1642', '2023_0726_1602', '2023_0726_1619', '2023_0726_1620']:
                continue
            for frame, frame_objs in seq_frames.items():
                sample = {}
                sample['seq'] = seq
                for obj in frame_objs:
                    sample['rdr_frame'] = obj['Radar_frameID']
                    sample['frame'] = frame
                    sample['poses'] = [obj['pose']]
                    samples.append(sample)
        self.samples = samples

        # workaround for infferencing a directory
        # samples = []
        # for frame in sorted(os.listdir('/mnt/ssd3/cruw_pose/2024_0218_1209/DZYX_npy_f16')):
        #     sample = {}
        #     sample['seq'] = '2024_0218_1209'
        #     sample['frame'] = frame.split('.')[0]
        #     sample['rdr_frame'] = frame.split('.')[0]
        #     sample['poses'] = []
        #     samples.append(sample)
        # self.samples = samples

    def consider_roi_polar(self, roi_polar, is_reflect_to_cfg=True):
        self.list_roi_idx = []
        deg2rad = np.pi/180.
        rad2deg = 180./np.pi
        for k, v in roi_polar.items():
            if v is not None:
                min_max = v if k == 'r' else (np.array(v) * deg2rad).tolist() 
                arr_roi, idx_min, idx_max = self.get_arr_in_roi(getattr(self, f'arr_{k}'), min_max)
                setattr(self, f'arr_{k}', arr_roi)
                self.list_roi_idx.append(idx_min)
                self.list_roi_idx.append(idx_max)
                if is_reflect_to_cfg:
                    v_new = [arr_roi[0], arr_roi[-1]]
                    v_new =  v_new if k == 'r' else (np.array(v_new) * rad2deg).tolist()
                    self.cfg.DATASET.DEAR.ROI[k] = v_new


    def consider_roi_cube(self, roi_cart):
        # to get indices
        self.list_roi_idx_cb = [0, len(self.arr_z_cb)-1, \
            0, len(self.arr_y_cb)-1, 0, len(self.arr_x_cb)-1]
        idx_attr = 0
        for k, v in roi_cart.items():
            if v is not None:
                min_max = np.array(v).tolist()
                # print(min_max)
                arr_roi, idx_min, idx_max = self.get_arr_in_roi(getattr(self, f'arr_{k}_cb'), min_max)
                setattr(self, f'arr_{k}_cb', arr_roi)
                self.list_roi_idx_cb[idx_attr*2] = idx_min
                self.list_roi_idx_cb[idx_attr*2+1] = idx_max
            idx_attr += 1

    def get_arr_in_roi(self, arr, min_max):
        min_val, max_val = min_max
        idx_min = np.argmin(abs(arr-min_val))
        idx_max = np.argmin(abs(arr-max_val))
        if max_val > arr[-1]:
            return arr[idx_min:idx_max+1], idx_min, idx_max
        return arr[idx_min:idx_max], idx_min, idx_max-1

    def check_to_add_obj(self, object_xyz):
        x, y, z = object_xyz
        x_min, y_min, z_min, x_max, y_max, z_max = self.roi_label
        if self.is_roi_check_with_azimuth:
            min_azi, max_azi = self.max_azimtuth_rad
            azimuth_center = np.arctan2(y, x)
            if (azimuth_center < min_azi) or (azimuth_center > max_azi)\
                or (x < x_min) or (y < y_min) or (z < z_min)\
                or (x > x_max) or (y > y_max) or (z > z_max):
                return False
        return True


    def get_cube_polar(self, seq, rdr_frame_id):
        # TODO: get radar cube in the DEAR format
        # return arr_dear
        pass

        
    def get_radar_cube_path(self, seq, rdr_frame_id, cube_dir=None):
        seq_name = self.seq_id_to_name.get(seq, seq)
        cube_dir = cube_dir or getattr(self.cfg.DATASET.DIR, 'RDR_CUBE_DIR', self.rdr_path_name)
        radar_dir = getattr(self.cfg.DATASET.DIR, 'RADAR', 'radar')
        sequences_dir = getattr(self.cfg.DATASET.DIR, 'SEQUENCES_DIR', 'sequences')
        candidates = [
            os.path.join(self.cfg.DATASET.DIR.ROOT_DIR, sequences_dir, seq_name, radar_dir, cube_dir, f'{rdr_frame_id}.npy'),
            os.path.join(self.cfg.DATASET.DIR.ROOT_DIR, 'sequences', seq_name, radar_dir, cube_dir, f'{rdr_frame_id}.npy'),
            os.path.join(self.cfg.DATASET.DIR.ROOT_DIR, seq_name, radar_dir, cube_dir, f'{rdr_frame_id}.npy'),
            os.path.join(self.cfg.DATASET.DIR.ROOT_DIR, seq_name, cube_dir, f'{rdr_frame_id}.npy'),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    def get_radar_cache_path(self, seq, rdr_frame_id):
        cache_root = getattr(self.cfg.DATASET.DIR, 'RDR_CACHE_ROOT', None)
        cache_dir = getattr(self.cfg.DATASET.DIR, 'RDR_CACHE_DIR', None)
        if not cache_root or not cache_dir:
            return None
        seq_name = self.seq_id_to_name.get(seq, seq)
        radar_dir = getattr(self.cfg.DATASET.DIR, 'RADAR', 'radar')
        sequences_dir = getattr(self.cfg.DATASET.DIR, 'SEQUENCES_DIR', 'sequences')
        candidates = [
            os.path.join(cache_root, sequences_dir, seq_name, radar_dir, cache_dir, f'{rdr_frame_id}.npy'),
            os.path.join(cache_root, 'sequences', seq_name, radar_dir, cache_dir, f'{rdr_frame_id}.npy'),
            os.path.join(cache_root, seq_name, radar_dir, cache_dir, f'{rdr_frame_id}.npy'),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def select_doppler_bins(self, arr_cube):
        if 'd' not in self.cfg.DATASET.RDR_TYPE or arr_cube.ndim < 4:
            return arr_cube
        dzyx_cfg = getattr(self.cfg.DATASET, 'DZYX', None)
        if dzyx_cfg is None:
            return arr_cube
        doppler_slice = getattr(dzyx_cfg, 'DOPPLER_SLICE', None)
        if doppler_slice:
            start, end = [int(v) for v in doppler_slice]
            return arr_cube[start:end]
        doppler_bins = getattr(dzyx_cfg, 'DOPPLER_BINS', None)
        if doppler_bins:
            doppler_bins = int(doppler_bins)
            if arr_cube.shape[0] > doppler_bins:
                start = (arr_cube.shape[0] - doppler_bins) // 2
                return arr_cube[start:start+doppler_bins]
        return arr_cube

    def cast_cached_cube(self, arr_cube):
        dzyx_cfg = getattr(self.cfg.DATASET, 'DZYX', None)
        if dzyx_cfg is not None and getattr(dzyx_cfg, 'LOAD_FLOAT32', False):
            return arr_cube.astype(np.float32)
        return arr_cube

    def get_cube(self, seq, rdr_frame_id):
        cache_path = self.get_radar_cache_path(seq, rdr_frame_id)
        if cache_path is not None:
            arr_cube = self.select_doppler_bins(np.load(cache_path, mmap_mode='r'))
            return self.cast_cached_cube(arr_cube)

        arr_cube = np.load(self.get_radar_cube_path(seq, rdr_frame_id), mmap_mode='r')
        idx_z_min, idx_z_max, idx_y_min, idx_y_max, idx_x_min, idx_x_max = self.list_roi_idx_cb
        if 'd' in self.cfg.DATASET.RDR_TYPE:
            arr_cube = arr_cube[:, idx_z_min:idx_z_max+1,idx_y_min:idx_y_max+1,idx_x_min:idx_x_max+1]
        else:
            arr_cube = arr_cube[idx_z_min:idx_z_max+1,idx_y_min:idx_y_max+1,idx_x_min:idx_x_max+1]
        if np.iscomplexobj(arr_cube):
            arr_cube = np.abs(arr_cube).astype(np.float32)
        else:
            arr_cube = arr_cube.astype(np.float32)
        norm_vals = [float(norm_val) for norm_val in self.rad_normalize_values]
        norm_start, norm_scale = norm_vals[0], norm_vals[1]-norm_vals[0]
        # normalize
        arr_cube = (arr_cube - norm_start) / norm_scale
        arr_cube[arr_cube < 0.] = 0.

        return self.select_doppler_bins(arr_cube).astype(np.float32)


    def get_cube_phase(self, seq, rdr_frame_id):
        cube_dir = getattr(self.cfg.DATASET.DIR, 'RDR_PHASE_DIR', 'DZYX_npy_f16_complex')
        arr_cube = np.load(self.get_radar_cube_path(seq, rdr_frame_id, cube_dir=cube_dir))
        if np.iscomplexobj(arr_cube):
            arr_cube = np.stack((arr_cube.real, arr_cube.imag), axis=0).astype(np.float16)
        else:
            arr_cube = arr_cube.astype(np.float16)
        # RoI selection
        idx_z_min, idx_z_max, idx_y_min, idx_y_max, idx_x_min, idx_x_max = self.list_roi_idx_cb
        arr_cube = arr_cube[:, :, idx_z_min:idx_z_max+1,idx_y_min:idx_y_max+1,idx_x_min:idx_x_max+1]
        # data has been normalized
        return arr_cube
    
    def get_pc(self, seq, frame_id, dir_name):
        pc = np.load(os.path.join(self.cfg.DATASET.DIR.ROOT_DIR, self.seq_id_to_name[seq], dir_name, f'{frame_id}.npy'))
        return pc

    def __len__(self):
        return len(self.samples)


    def get_sample_by_idx(self, idx):
        sample = self.samples[idx]
        dict_item = {}
        dict_item['meta'] = {'seq': sample['seq'], 'frame': sample['frame'], 'rdr_frame': sample['rdr_frame']}
        dict_item['poses'] = sample['poses']
        if self.enable_radar:
            # dict_item['rdr_cube'] = self.get_cube_phase(sample['seq'], sample['rdr_frame'])
            dict_item['rdr_cube'] = self.get_cube(sample['seq'], sample['rdr_frame'])
            dict_item['hm_size'] = (len(self.arr_z_cb), len(self.arr_y_cb), len(self.arr_x_cb))
        if self.enable_lidar:
            dict_item['lidar_pc'] = self.get_pc(sample['seq'], sample['frame'], self.cfg.DATASET.DIR.LIDAR)
            dict_item['P_L2R'] = self.P_L2R
        dict_item.update(mode=self.split)
        dict_item, _ = self.pipeline(dict_item, info=self.cfg)
        return dict_item

    def __getitem__(self, idx):
        dict_item = self.get_sample_by_idx(idx)
        return dict_item
        
    @staticmethod
    def collate_fn(batch_list):
        if None in batch_list:
            print('* Exception error (Dataset): collate_fn')
            return None
        enabled_sensors = []
        if 'rdr' in batch_list[0]:
            enabled_sensors.append('rdr')
        if 'lidar' in batch_list[0]:
            enabled_sensors.append('lidar')
        ret = defaultdict(dict)
        for sensor_type in enabled_sensors:
            example_merged = collections.defaultdict(list)
            for example in batch_list:
                for k, v in example[sensor_type].items():
                    example_merged[k].append(v)
            for key, elems in example_merged.items():
                if key in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels", "hm", "anno_pose",
                            "ind", "mask", "cat", "obj_id"]:
                    ret[sensor_type][key] = collections.defaultdict(list)
                    res = []
                    for elem in elems:
                        for idx, ele in enumerate(elem):
                            ret[sensor_type][key][str(idx)].append(torch.as_tensor(ele))
                    for kk, vv in ret[sensor_type][key].items():
                        res.append(torch.stack(vv))
                    ret[sensor_type][key] = res  # [task], task: (batch, num_class_in_task, feat_shape_h, feat_shape_w)
                elif key in ["voxels", "num_points", "num_gt", "voxel_labels", "num_voxels",
                    "cyv_voxels", "cyv_num_points", "cyv_num_voxels"]:
                    ret[sensor_type][key] = torch.as_tensor(np.concatenate(elems, axis=0))
                elif key == "points":
                    ret[sensor_type][key] = [torch.as_tensor(elem) for elem in elems]
                elif key in ["coordinates", "cyv_coordinates"]:
                    coors = []
                    for i, coor in enumerate(elems):
                        coor_pad = np.pad(
                            coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
                        )
                        coors.append(coor_pad)
                    ret[sensor_type][key] = torch.as_tensor(np.concatenate(coors, axis=0))
                elif key in ['rdr_tensor']:
                    elems = np.ascontiguousarray(np.stack(elems, axis=0))
                    ret[sensor_type][key] = torch.from_numpy(elems)
                else:
                    ret[sensor_type][key] = np.stack(elems, axis=0)
        ret = dict(ret)
        meta_list = []
        for example in batch_list:
            meta_list.append(example['meta'])
        ret['meta'] = meta_list

        return ret
    
    def evaluation(self, detections, output_dir=None, testset=False):
        with open(self.label_file, 'r') as f:
            gt = json.load(f)
        joint_names = self.class_names or [str(i) for i in range(15)]
        seq_mpjpe = defaultdict(list) # each element is 1d np array of mpjpe for each keypoint
        seq_abs_mpjpe = defaultdict(list) # each element is 1d np array of absmpjpe for each keypoint
        seq_mrpe = defaultdict(list)
        motion_mpjpe = defaultdict(list)
        motion_abs_mpjpe = defaultdict(list)
        motion_mrpe = defaultdict(list)
        all_mpjpe = []
        all_abs_mpjpe = []
        all_mrpe = []
        for seq_frame_rdr_frame, val in detections.items():
            seq, frame, rdr_frame = seq_frame_rdr_frame.split('/')
            gt_points = gt[seq][frame][0]['pose']
            keypoints = np.array([point[1:4] for point in val['keypoints']], dtype=np.float32)
            gt_points = np.array(gt_points, dtype=np.float32)

            mrpe = np.linalg.norm(keypoints[0] - gt_points[0])
            mpjpe = PJPE(keypoints.copy(), gt_points.copy())
            abs_mpjpe = ABS_PJPE(keypoints, gt_points)
            activity = self.seq_id_to_activity.get(seq, 'UNKNOWN')

            seq_mrpe[seq].append(mrpe)
            seq_mpjpe[seq].append(mpjpe)
            seq_abs_mpjpe[seq].append(abs_mpjpe)
            motion_mrpe[activity].append(mrpe)
            motion_mpjpe[activity].append(mpjpe)
            motion_abs_mpjpe[activity].append(abs_mpjpe)
            all_mrpe.append(mrpe)
            all_mpjpe.append(mpjpe)
            all_abs_mpjpe.append(abs_mpjpe)

        def build_metrics(mrpes, mpjpes, abs_mpjpes):
            mrpes = np.array(mrpes, dtype=np.float32)
            mpjpes = np.array(mpjpes, dtype=np.float32)
            abs_mpjpes = np.array(abs_mpjpes, dtype=np.float32)
            mpjpes_per_joint = np.mean(mpjpes, axis=0) * 1000
            abs_mpjpes_per_joint = np.mean(abs_mpjpes, axis=0) * 1000
            metrics = {
                'num_predictions': int(len(mrpes)),
                'MRPE': float(np.mean(mrpes) * 1000),
                'MPJPE': float(np.mean(mpjpes_per_joint)),
                'ABS_MPJPE': float(np.mean(abs_mpjpes_per_joint)),
            }
            for joint_idx in range(mpjpes_per_joint.shape[0]):
                metrics[f'PJPE_{joint_idx}'] = float(mpjpes_per_joint[joint_idx])
                metrics[f'ABS_PJPE_{joint_idx}'] = float(abs_mpjpes_per_joint[joint_idx])
            return metrics

        seq_res = {}
        for seq in seq_mpjpe.keys():
            seq_name = self.seq_id_to_name[seq]
            seq_res[seq_name] = build_metrics(seq_mrpe[seq], seq_mpjpe[seq], seq_abs_mpjpe[seq])
            seq_res[seq_name]['motion'] = self.seq_id_to_activity.get(seq, 'UNKNOWN')

        motion_res = {}
        for activity in motion_mpjpe.keys():
            motion_res[activity] = build_metrics(
                motion_mrpe[activity],
                motion_mpjpe[activity],
                motion_abs_mpjpe[activity],
            )

        total_results = build_metrics(all_mrpe, all_mpjpe, all_abs_mpjpe)

        joint_res = {}
        all_mpjpe = np.array(all_mpjpe, dtype=np.float32)
        all_abs_mpjpe = np.array(all_abs_mpjpe, dtype=np.float32)
        if all_mpjpe.size:
            mpjpes_per_joint = np.mean(all_mpjpe, axis=0) * 1000
            abs_mpjpes_per_joint = np.mean(all_abs_mpjpe, axis=0) * 1000
            for joint_idx, joint_name in enumerate(joint_names):
                joint_res[joint_name] = {
                    'MRPE': float(abs_mpjpes_per_joint[joint_idx]),
                    'MPJPE': float(mpjpes_per_joint[joint_idx]),
                    'ABS_MPJPE': float(abs_mpjpes_per_joint[joint_idx]),
                }

        res = {}
        res['results'] = total_results
        seq_res['ALL'] = total_results
        res['seq_results'] = seq_res
        res['motion_results'] = motion_res
        res['joint_results'] = joint_res
        return res, None
