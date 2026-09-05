[AutoPyBindCUDA]
[CUDAKernel]
void frameMIMO_kernel(
    TensorView<double> t_sample,                    // [T]  time sampling = 256
    TensorView<float> total_lengths,                // [F, TX, RX, N]
    TensorView<float> amplitudes,                   // [F, TX, RX, N]
    float fc, float slope, float phi, int max_N,    // max_N is the maximum number of points in these chirps
    TensorView<double> out_real,                    // [TX, RX, F, T]
    TensorView<double> out_imag                     // [TX, RX, F, T]
)
{
    uint t_idx = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint pair_id = cudaBlockIdx().y;
    uint num_rx = (uint)total_lengths.size(2);
    uint tx_id = pair_id / num_rx;
    uint rx_id = pair_id % num_rx;
    uint chirp_id = cudaBlockIdx().z;

    if (t_idx >= t_sample.size(0)) return;

    float c0 = 299792458.0;
    double pi2 = 6.283185307179586;

    double t_val = t_sample[t_idx];
    double fc_d = (double)fc;
    double slope_d = (double)slope;

    double sum_real = 0.0;
    double sum_imag = 0.0;

    for (uint i = 0; i < max_N; ++i) {
        float amp = amplitudes[chirp_id, tx_id, rx_id, i];
        if (amp == 0.0) continue;

        float distance = total_lengths[chirp_id, tx_id, rx_id, i];
        double tof = (double)distance / (double)c0;  // time of flight

        // tx(t) * conj(rx(t-tof)); the common phi term cancels.
        double mixed_phase = pi2 * (fc_d * tof + slope_d * t_val * tof - 0.5 * slope_d * tof * tof);
        double mixed_real = cos(mixed_phase);
        double mixed_imag = sin(mixed_phase);

        sum_real += mixed_real * (double)amp;
        sum_imag += mixed_imag * (double)amp; // sum of all reflections point at this sample point
    }

    out_real[tx_id, rx_id, chirp_id, t_idx] = sum_real;
    out_imag[tx_id, rx_id, chirp_id, t_idx] = sum_imag;
}


// Float32 variant of full MIMO. This keeps the same beat-phase formula as
// frameMIMO_kernel, but uses float accumulators and sincos().
[AutoPyBindCUDA]
[CUDAKernel]
void frameMIMO_kernel_f32(
    TensorView<float> t_sample,
    TensorView<float> total_lengths,                // [F, TX, RX, N]
    TensorView<float> amplitudes,                   // [F, TX, RX, N]
    float fc, float slope, int max_N,
    TensorView<float> out_real,                     // [TX, RX, F, T]
    TensorView<float> out_imag                      // [TX, RX, F, T]
)
{
    uint t_idx = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint pair_id = cudaBlockIdx().y;
    uint num_rx = (uint)total_lengths.size(2);
    uint tx_id = pair_id / num_rx;
    uint rx_id = pair_id % num_rx;
    uint chirp_id = cudaBlockIdx().z;

    if (t_idx >= t_sample.size(0)) return;

    float c0 = 299792458.0f;
    float pi2 = 6.283185307179586f;
    float t_val = t_sample[t_idx];
    float sum_real = 0.0f;
    float sum_imag = 0.0f;

    for (uint i = 0; i < max_N; ++i) {
        float amp = amplitudes[chirp_id, tx_id, rx_id, i];
        if (amp == 0.0f) continue;

        float tof = total_lengths[chirp_id, tx_id, rx_id, i] / c0;
        float mixed_phase = pi2 * (fc * tof + slope * t_val * tof - 0.5f * slope * tof * tof);
        float s, c;
        sincos(mixed_phase, s, c);
        sum_real += c * amp;
        sum_imag += s * amp;
    }

    out_real[tx_id, rx_id, chirp_id, t_idx] = sum_real;
    out_imag[tx_id, rx_id, chirp_id, t_idx] = sum_imag;
}


// Float32 MIMO variant using precomputed phase coefficients:
// phase(t) = phase_const + phase_rate * t.
[AutoPyBindCUDA]
[CUDAKernel]
void frameMIMO_phase_coeff_f32_kernel(
    TensorView<float> t_sample,
    TensorView<float> phase_const,                  // [F, TX, RX, N]
    TensorView<float> phase_rate,                   // [F, TX, RX, N]
    TensorView<float> amplitudes,                   // [F, TX, RX, N]
    int max_N,
    TensorView<float> out_real,                     // [TX, RX, F, T]
    TensorView<float> out_imag                      // [TX, RX, F, T]
)
{
    uint t_idx = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint pair_id = cudaBlockIdx().y;
    uint num_rx = (uint)phase_const.size(2);
    uint tx_id = pair_id / num_rx;
    uint rx_id = pair_id % num_rx;
    uint chirp_id = cudaBlockIdx().z;

    if (t_idx >= t_sample.size(0)) return;

    float t_val = t_sample[t_idx];
    float sum_real = 0.0f;
    float sum_imag = 0.0f;

    for (uint i = 0; i < max_N; ++i) {
        float amp = amplitudes[chirp_id, tx_id, rx_id, i];
        if (amp == 0.0f) continue;

        float phase = phase_const[chirp_id, tx_id, rx_id, i] + phase_rate[chirp_id, tx_id, rx_id, i] * t_val;
        float s, c;
        sincos(phase, s, c);
        sum_real += c * amp;
        sum_imag += s * amp;
    }

    out_real[tx_id, rx_id, chirp_id, t_idx] = sum_real;
    out_imag[tx_id, rx_id, chirp_id, t_idx] = sum_imag;
}


float interp_dipole_1deg(TensorView<float> values, float angle_deg)
{
    float pos = angle_deg + 90.0f;
    if (pos < 0.0f || pos > 180.0f) return 0.0f;
    int left = (int)floor(pos);
    if (left < 0) left = 0;
    if (left > 180) left = 180;
    int right = left + 1;
    if (right > 180) right = 180;
    float w = pos - (float)left;
    return values[left] + w * (values[right] - values[left]);
}

float antenna_gain_from_world_vec(
    TensorView<float> world_from_local,
    TensorView<float> antenna_values,
    float vx,
    float vy,
    float vz
)
{
    float lx = vx * world_from_local[0, 0] + vy * world_from_local[1, 0] + vz * world_from_local[2, 0];
    float ly = vx * world_from_local[0, 1] + vy * world_from_local[1, 1] + vz * world_from_local[2, 1];
    float lz = vx * world_from_local[0, 2] + vy * world_from_local[1, 2] + vz * world_from_local[2, 2];
    float forward = -lz;
    float rad_to_deg = 57.29577951308232f;
    float x_angle = atan2(lx, forward) * rad_to_deg;
    float y_angle = atan2(ly, forward) * rad_to_deg;
    return interp_dipole_1deg(antenna_values, x_angle) * interp_dipole_1deg(antenna_values, y_angle);
}

// Fused float32 MIMO for direct point/intensity samples. This avoids the
// Python-side total path length and amplitude tensors used by frameCuda().
[AutoPyBindCUDA]
[CUDAKernel]
void frameMIMO_fused_f32_kernel(
    TensorView<float> t_sample,                     // [T]
    TensorView<float> points,                       // [F, N, 3]
    TensorView<float> intensities,                  // [F, N]
    TensorView<float> tx_pos,                       // [TX, 3]
    TensorView<float> rx_pos,                       // [RX, 3]
    TensorView<float> world_from_local,             // [3, 3]
    TensorView<float> antenna_values,               // [181], -90..90 degrees
    float fc, float slope, float wavelength, float gain, int max_N,
    TensorView<float> out_real,                     // [TX, RX, F, T]
    TensorView<float> out_imag                      // [TX, RX, F, T]
)
{
    uint t_idx = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint pair_id = cudaBlockIdx().y;
    uint num_rx = (uint)rx_pos.size(0);
    uint tx_id = pair_id / num_rx;
    uint rx_id = pair_id % num_rx;
    uint chirp_id = cudaBlockIdx().z;

    if (t_idx >= t_sample.size(0)) return;

    float c0 = 299792458.0f;
    float pi2 = 6.283185307179586f;
    float inv_4pi = 0.07957747154594767f;
    float t_val = t_sample[t_idx];
    float sum_real = 0.0f;
    float sum_imag = 0.0f;

    float tx_x = tx_pos[tx_id, 0];
    float tx_y = tx_pos[tx_id, 1];
    float tx_z = tx_pos[tx_id, 2];
    float rx_x = rx_pos[rx_id, 0];
    float rx_y = rx_pos[rx_id, 1];
    float rx_z = rx_pos[rx_id, 2];

    for (uint i = 0; i < max_N; ++i) {
        float scatter_power = intensities[chirp_id, i];
        if (scatter_power <= 0.0f) continue;

        float px = points[chirp_id, i, 0];
        float py = points[chirp_id, i, 1];
        float pz = points[chirp_id, i, 2];

        float tx_dx = px - tx_x;
        float tx_dy = py - tx_y;
        float tx_dz = pz - tx_z;
        float rx_dx = px - rx_x;
        float rx_dy = py - rx_y;
        float rx_dz = pz - rx_z;
        float dist_tx = sqrt(tx_dx * tx_dx + tx_dy * tx_dy + tx_dz * tx_dz);
        float dist_rx = sqrt(rx_dx * rx_dx + rx_dy * rx_dy + rx_dz * rx_dz);

        float tx_gain = antenna_gain_from_world_vec(world_from_local, antenna_values, tx_dx, tx_dy, tx_dz);
        float rx_gain = antenna_gain_from_world_vec(world_from_local, antenna_values, rx_dx, rx_dy, rx_dz);
        scatter_power *= max(tx_gain * rx_gain, 0.0f);

        float denom = max(dist_tx * dist_rx, 1.0e-6f);
        float amp = gain * sqrt(max(scatter_power, 0.0f)) * wavelength * inv_4pi / denom;
        if (amp == 0.0f) continue;

        float tof = (dist_tx + dist_rx) / c0;
        float mixed_phase = pi2 * (fc * tof + slope * t_val * tof - 0.5f * slope * tof * tof);
        float s, c;
        sincos(mixed_phase, s, c);
        sum_real += c * amp;
        sum_imag += s * amp;
    }

    out_real[tx_id, rx_id, chirp_id, t_idx] = sum_real;
    out_imag[tx_id, rx_id, chirp_id, t_idx] = sum_imag;
}


// Full MIMO kernel with target-chunk partial sums.
// Grid: (time-sample blocks, TX*RX pairs, F*num_target_chunks)
// Output: [num_chunks, TX, RX, F, T], reduced across chunks by Python.
[AutoPyBindCUDA]
[CUDAKernel]
void frameMIMO_chunked_kernel(
    TensorView<double> t_sample,
    TensorView<float> total_lengths,                // [F, TX, RX, N]
    TensorView<float> amplitudes,                   // [F, TX, RX, N]
    float fc, float slope, int max_N,
    int targets_per_chunk, int num_chunks,
    TensorView<double> partial_real,                // [num_chunks, TX, RX, F, T]
    TensorView<double> partial_imag                 // [num_chunks, TX, RX, F, T]
)
{
    uint t_idx = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint pair_id = cudaBlockIdx().y;
    uint z_id = cudaBlockIdx().z;
    uint chunk_id = z_id % (uint)num_chunks;
    uint chirp_id = z_id / (uint)num_chunks;
    uint num_rx = (uint)total_lengths.size(2);
    uint tx_id = pair_id / num_rx;
    uint rx_id = pair_id % num_rx;

    if (t_idx >= t_sample.size(0)) return;

    uint target_start = chunk_id * (uint)targets_per_chunk;
    uint target_end = min(target_start + (uint)targets_per_chunk, (uint)max_N);

    double c0 = 299792458.0;
    double pi2 = 6.283185307179586;
    double t_val = t_sample[t_idx];
    double fc_d = (double)fc;
    double slope_d = (double)slope;

    double sum_real = 0.0;
    double sum_imag = 0.0;

    for (uint i = target_start; i < target_end; ++i) {
        float amp = amplitudes[chirp_id, tx_id, rx_id, i];
        if (amp == 0.0) continue;

        double tof = (double)total_lengths[chirp_id, tx_id, rx_id, i] / c0;
        double mixed_phase = pi2 * (fc_d * tof + slope_d * t_val * tof - 0.5 * slope_d * tof * tof);
        sum_real += cos(mixed_phase) * (double)amp;
        sum_imag += sin(mixed_phase) * (double)amp;
    }

    partial_real[chunk_id, tx_id, rx_id, chirp_id, t_idx] = sum_real;
    partial_imag[chunk_id, tx_id, rx_id, chirp_id, t_idx] = sum_imag;
}


// Chirp kernel - internal sum version (memory efficient, less parallel)
// Each thread handles one time sample and loops through ALL targets
// Output: [T] - final result directly
[AutoPyBindCUDA]
[CUDAKernel]
void chirp_kernel_internal(
    TensorView<double> t_sample,    // [T] time samples
    TensorView<float> distances,    // [N] round-trip distances
    TensorView<float> amplitudes,   // [N] amplitudes
    float fc,
    float slope,
    int num_targets,
    TensorView<double> out_real,    // [T] output real
    TensorView<double> out_imag     // [T] output imag
)
{
    uint t_idx = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;

    int T = t_sample.size(0);
    if (t_idx >= T) return;

    float c0 = 299792458.0;
    double pi2 = 6.283185307179586;

    double t_val = t_sample[t_idx];
    double fc_d = (double)fc;
    double slope_d = (double)slope;

    double sum_real = 0.0;
    double sum_imag = 0.0;

    for (int i = 0; i < num_targets; ++i) {
        float distance = distances[i];
        float amp = amplitudes[i];

        if (amp == 0.0) continue;

        double tof = (double)distance / (double)c0;
        double mixed_phase = pi2 * (fc_d * tof + slope_d * t_val * tof - 0.5 * slope_d * tof * tof);
        double mixed_real = cos(mixed_phase);
        double mixed_imag = sin(mixed_phase);

        sum_real += mixed_real * (double)amp;
        sum_imag += mixed_imag * (double)amp;
    }

    out_real[t_idx] = sum_real;
    out_imag[t_idx] = sum_imag;
}

// Chirp kernel - external sum version (chunked, more parallel, more memory)
// Grid: (num_t_blocks, num_target_chunks, 1)
// Each block processes a chunk of targets and writes to partial output
// Output: [num_target_chunks, T] - partial sums to be reduced in Python
[AutoPyBindCUDA]
[CUDAKernel]
void chirp_kernel_chunked(
    TensorView<double> t_sample,    // [T] time samples
    TensorView<float> distances,    // [N] round-trip distances
    TensorView<float> amplitudes,   // [N] amplitudes
    float fc,
    float slope,
    int num_targets,
    int targets_per_chunk,
    TensorView<double> out_real,    // [num_chunks, T] output real
    TensorView<double> out_imag     // [num_chunks, T] output imag
)
{
    uint t_idx = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint chunk_idx = cudaBlockIdx().y;

    int T = t_sample.size(0);
    if (t_idx >= T) return;

    // Target range for this chunk
    int target_start = chunk_idx * targets_per_chunk;
    int target_end = min(target_start + targets_per_chunk, num_targets);

    float c0 = 299792458.0;
    double pi2 = 6.283185307179586;

    double t_val = t_sample[t_idx];
    double fc_d = (double)fc;
    double slope_d = (double)slope;

    double sum_real = 0.0;
    double sum_imag = 0.0;

    for (int i = target_start; i < target_end; ++i) {
        float distance = distances[i];
        float amp = amplitudes[i];

        if (amp == 0.0) continue;

        double tof = (double)distance / (double)c0;
        double mixed_phase = pi2 * (fc_d * tof + slope_d * t_val * tof - 0.5 * slope_d * tof * tof);
        double mixed_real = cos(mixed_phase);
        double mixed_imag = sin(mixed_phase);

        sum_real += mixed_real * (double)amp;
        sum_imag += mixed_imag * (double)amp;
    }

    out_real[chunk_idx, t_idx] = sum_real;
    out_imag[chunk_idx, t_idx] = sum_imag;
}

// Chirp kernel - per-target chunked version
// Each thread handles one target and loops through ALL time samples
// Output: [num_chunks, T] - partial sums to be reduced in Python
[AutoPyBindCUDA]
[CUDAKernel]
void chirp_kernel_per_target(
    TensorView<double> t_sample,    // [T] time samples
    TensorView<float> distances,    // [N] round-trip distances
    TensorView<float> amplitudes,   // [N] amplitudes
    float fc,
    float slope,
    int num_targets,
    int T,
    int targets_per_chunk,
    TensorView<float> out_real,     // [num_chunks, T] output real
    TensorView<float> out_imag      // [num_chunks, T] output imag
)
{
    uint thread_idx = cudaThreadIdx().x;
    uint chunk_idx = cudaBlockIdx().x;

    uint target_idx = chunk_idx * targets_per_chunk + thread_idx;
    if (target_idx >= num_targets) return;

    double c0 = 299792458.0;
    double pi2 = 6.283185307179586;

    double distance = (double)distances[target_idx];
    double amp = (double)amplitudes[target_idx];

    if (amp == 0.0) return;

    double tof = distance / c0;
    double fc_d = (double)fc;
    double slope_d = (double)slope;

    // Loop through all time samples
    for (int t_idx = 0; t_idx < T; ++t_idx) {
        double t_val = t_sample[t_idx];

        double mixed_phase = pi2 * (fc_d * tof + slope_d * t_val * tof - 0.5 * slope_d * tof * tof);
        double mixed_real = cos(mixed_phase);
        double mixed_imag = sin(mixed_phase);

        float contrib_real = (float)(mixed_real * amp);
        float contrib_imag = (float)(mixed_imag * amp);

        // Atomic add within chunk (threads in same block write to same chunk output)
        float oldVal;
        out_real.InterlockedAdd(uint2(chunk_idx, t_idx), contrib_real, oldVal);
        out_imag.InterlockedAdd(uint2(chunk_idx, t_idx), contrib_imag, oldVal);
    }
}
