# 4Dradar_Processing
## Python raw echo to xyz-Doppler
`raw_echo_to_xyz.py` reimplements the Matlab raw echo processing path and writes cartesian tensors directly to
`(Your Data Path)/RT-Pose/datasets/GT_sequences/(sequence ID)/radar/npy_DZYX_complex`.

The output shape is `Doppler(64) x Z(32) x Y(128) x X(256)` with complex64 values.
Like the original `4Dradar2xyz.py`, this Python generator reverses the saved Y axis (`axis=2`) before writing `.npy`
files, so the generated tensors stay byte-order compatible with the Matlab-derived pipeline.

Process one sequence:
```
python raw_echo_to_xyz.py --dataset_dir (Your Data Path)/RT-Pose/datasets/GT_sequences --sequence 0 --rangemat-correction on --peakvalmat-correction on
```

Use Torch/CUDA for GPU acceleration:
```
python raw_echo_to_xyz.py --dataset_dir (Your Data Path)/RT-Pose/datasets/GT_sequences --sequence 0 --backend torch --gpu-device 1 --x-chunk 256 --rangemat-correction on --peakvalmat-correction on
```

`--backend auto` uses CuPy when it is installed, otherwise Torch/CUDA when available, and otherwise falls back to
NumPy. On the RTX 2080 Ti machine with driver 535 / CUDA 12.2, `torch==2.5.1+cu121` works; avoid the default PyPI
CUDA 13 wheel on that driver.

Single-frame benchmark on sequence 0 frame 2:
* NumPy CPU, `--x-chunk 8`: 86.80 s
* NumPy CPU, `--x-chunk 256`: 31.05 s
* Torch CUDA on RTX 2080 Ti, `--x-chunk 256`: 6.60 s

Process selected frames and optionally keep the intermediate `Doppler x Range x Azimuth x Elevation` mat file:
```
python raw_echo_to_xyz.py --dataset_dir (Your Data Path)/RT-Pose/datasets/GT_sequences --sequence 0 --frames 2 3 --save-drae --rangemat-correction on --peakvalmat-correction on
```

Parallelize by frame:
```
python raw_echo_to_xyz.py --dataset_dir (Your Data Path)/RT-Pose/datasets/GT_sequences --sequence 0 --workers 2 --rangemat-correction on --peakvalmat-correction on
```

Each worker may need more than 1 GB RAM because one output frame is a full 4D complex tensor. Increase `--workers`
only when memory is sufficient. For one GPU, start with `--workers 1` and use the largest `--x-chunk` that fits GPU
memory.

## Step 1
Using Matlab to prepocss raw mmWave data to 4D radar tensor with complex format in `Doppler(64) x Range(256) x Azimuth(128)x Elevation(32)`. In this work, the program is running under **Matlab2022b**.
* Open `main.m` in folder `mmWave-Matlabe`
* Change the `Your Data Path` in `line 38` in the main.m
* **Optional**: change the `line 43` to specific sequences you want to process
* Run `main.m`
* Processed data will be stored in `(Your Data Path)/RT-Pose/datasets/GT_sequences/(sequence ID)/radar/mat`

## Step 2
Using Pyhton to prepocss `.mat file` to **cartesian coordinate** in `Doppler(64) x Z(32) x Y(128) x X(256)` and store as `.npy file`. 

```
conda create -n 4Dradar_preprocess python=3.9
conda activate 4Dradar_preprocess
pip install -r requirements.txt
```
Process all sequences:
```
python 4Dradar2xyz.py --dataset_dir (Your Data Path) 
```
**Optional**: Process specific sequences you want:
```
python 4Dradar2xyz.py --dataset_dir (Your Data Path) --sequence (sequnece ID)
```
