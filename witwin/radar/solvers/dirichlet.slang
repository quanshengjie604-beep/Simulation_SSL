// Dirichlet kernel for radar signal processing
// Forward and backward passes for differentiable rendering

static const float PI = 3.14159265358979323846f;
static const float TWO_PI = 6.28318530717958647692f;

struct Complex
{
    float re;
    float im;
};

Complex cmul(Complex a, Complex b)
{
    return Complex(a.re * b.re - a.im * b.im,
                   a.re * b.im + a.im * b.re);
}

Complex cexp_f(float phase)
{
    float s, c;
    sincos(phase, s, c);
    return Complex(c, s);
}

// Real Dirichlet kernel: S(x) = sin((n+0.5)x) / sin(x/2)
float dirichlet_kernel_real(float x, float n)
{
    float eps = 1e-7f;
    float half_x = x * 0.5f;
    float sin_half, cos_half;
    sincos(half_x, sin_half, cos_half);

    if (abs(sin_half) < eps)
    {
        return 2.0f * n + 1.0f;
    }

    float nph = n + 0.5f;
    float sin_nph, cos_nph;
    sincos(nph * x, sin_nph, cos_nph);
    return sin_nph / sin_half;
}

// Gradient of real Dirichlet kernel w.r.t. x
float dirichlet_kernel_real_grad(float x, float n)
{
    float eps = 1e-7f;
    float half_x = x * 0.5f;
    float sin_half, cos_half;
    sincos(half_x, sin_half, cos_half);

    if (abs(sin_half) < eps)
    {
        return 0.0f;
    }

    float nph = n + 0.5f;
    float sin_nph, cos_nph;
    sincos(nph * x, sin_nph, cos_nph);
    float sin_half_sq = sin_half * sin_half;

    return (nph * cos_nph * sin_half - 0.5f * sin_nph * cos_half) / sin_half_sq;
}

// Dirichlet kernel with linear phase: D_n(x) = e^{-inx} * sin((n+0.5)x) / sin(x/2)
Complex dirichlet_kernel(float x, float n)
{
    float S = dirichlet_kernel_real(x, n);
    Complex phase = cexp_f(-n * x);
    return Complex(phase.re * S, phase.im * S);
}

// Gradient of Dirichlet kernel with linear phase
Complex dirichlet_kernel_grad(float x, float n)
{
    float S = dirichlet_kernel_real(x, n);
    float S_grad = dirichlet_kernel_real_grad(x, n);

    Complex bracket = Complex(S_grad, -n * S);
    Complex phase = cexp_f(-n * x);
    return cmul(bracket, phase);
}

// ============================================================================
// Forward (internal sum): each thread handles one bin, loops through ALL targets
// Memory efficient, less parallel for large N
// Output: output_re (M,), output_im (M,) where M = num_bins
// ============================================================================

[AutoPyBindCUDA]
[CUDAKernel]
void forward_internal(
    TensorView<float> d,          // (N,) distances (one-way)
    TensorView<float> a,          // (N,) amplitudes
    TensorView<float> output_re,  // (M,) output real
    TensorView<float> output_im,  // (M,) output imag
    float n,                      // Dirichlet order
    float k0_per_meter,           // k0 = distance * k0_per_meter
    int num_bins,                 // M (output size, = N_fft / 2)
    int N_fft,                    // Full FFT size
    int num_targets,              // N
    float fc,                     // Carrier frequency (scaled, e.g., 77 for 77 GHz)
    float slope,                  // Chirp slope in Hz/s (e.g., 100e12)
    float t_start                 // ADC start time in seconds
)
{
    uint3 idx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    int bin = int(idx.x);

    if (bin >= num_bins) return;

    float c0 = 299792458.0f;
    float sum_re = 0.0f;
    float sum_im = 0.0f;

    for (int i = 0; i < num_targets; i++)
    {
        float dist = d[i];
        float amp = a[i];

        // Round-trip time of flight
        float tau = 2.0f * dist / c0;

        // Chirp initial phase: phi_0 = 2*pi * [fc*tau + slope*tau*(t_start - 0.5*tau)]
        float phi_0 = TWO_PI * (fc * tau + slope * tau * (t_start - 0.5f * tau));

        // k0 for this target
        float k0 = dist * k0_per_meter;

        // x = 2*pi*(bin - k0) / N_fft
        float x = TWO_PI * (float(bin) - k0) / float(N_fft);

        Complex kernel = dirichlet_kernel(x, n);

        // Apply chirp phase correction
        Complex phase_corr = cexp_f(phi_0);
        Complex result = cmul(kernel, phase_corr);

        sum_re += amp * result.re;
        sum_im += amp * result.im;
    }

    output_re[bin] = sum_re;
    output_im[bin] = sum_im;
}

// ============================================================================
// Forward (chunked/external sum): parallelized over both bins and target chunks
// More parallel, but requires more memory for partial sums
// Output: output_re (num_chunks, M), output_im (num_chunks, M)
// ============================================================================

[AutoPyBindCUDA]
[CUDAKernel]
void forward_chunked(
    TensorView<float> d,          // (N,) distances (one-way)
    TensorView<float> a,          // (N,) amplitudes
    TensorView<float> output_re,  // (num_chunks, M) output real
    TensorView<float> output_im,  // (num_chunks, M) output imag
    float n,                      // Dirichlet order
    float k0_per_meter,
    int num_bins,                 // M
    int N_fft,
    int num_targets,              // N
    int targets_per_chunk,
    float fc,
    float slope,
    float t_start                 // ADC start time in seconds
)
{
    uint bin = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint chunk_idx = cudaBlockIdx().y;

    if (bin >= num_bins) return;

    // Target range for this chunk
    int target_start = chunk_idx * targets_per_chunk;
    int target_end = min(target_start + targets_per_chunk, num_targets);

    float c0 = 299792458.0f;
    float sum_re = 0.0f;
    float sum_im = 0.0f;

    for (int i = target_start; i < target_end; i++)
    {
        float dist = d[i];
        float amp = a[i];

        float tau = 2.0f * dist / c0;
        float phi_0 = TWO_PI * (fc * tau + slope * tau * (t_start - 0.5f * tau));
        float k0 = dist * k0_per_meter;
        float x = TWO_PI * (float(bin) - k0) / float(N_fft);

        Complex kernel = dirichlet_kernel(x, n);
        Complex phase_corr = cexp_f(phi_0);
        Complex result = cmul(kernel, phase_corr);

        sum_re += amp * result.re;
        sum_im += amp * result.im;
    }

    output_re[chunk_idx, bin] = sum_re;
    output_im[chunk_idx, bin] = sum_im;
}

// ============================================================================
// Backward: gradients w.r.t. d and a
// ============================================================================

[AutoPyBindCUDA]
[CUDAKernel]
void backward(
    TensorView<float> d,              // (N,) distances
    TensorView<float> a,              // (N,) amplitudes
    TensorView<float> grad_output_re, // (M,) upstream gradient real
    TensorView<float> grad_output_im, // (M,) upstream gradient imag
    TensorView<float> grad_d,         // (N,) output gradient for d
    TensorView<float> grad_a,         // (N,) output gradient for a
    float n,
    float k0_per_meter,
    int num_bins,
    int N_fft,
    int num_targets,
    float fc,                         // Carrier frequency (scaled)
    float slope,                      // Chirp slope in Hz/s
    float t_start                     // ADC start time in seconds
)
{
    uint3 idx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    int i = int(idx.x);

    if (i >= num_targets) return;

    float c0 = 299792458.0f;
    float dist = d[i];
    float amp = a[i];
    float k0 = dist * k0_per_meter;

    // Round-trip time of flight
    float tau = 2.0f * dist / c0;

    // Chirp initial phase: phi_0 = 2*pi * [fc*tau + slope*tau*(t_start - 0.5*tau)]
    float phi_0 = TWO_PI * (fc * tau + slope * tau * (t_start - 0.5f * tau));
    Complex phase_corr = cexp_f(phi_0);

    // d(phi_0)/d(dist) = 2*pi * 2/c0 * [fc + slope*t_start - slope*tau]
    float dphi0_dd = TWO_PI * 2.0f / c0 * (fc + slope * t_start - slope * tau);

    float dL_dd = 0.0f;
    float dL_da = 0.0f;

    for (int bin = 0; bin < num_bins; bin++)
    {
        float dL_dout_re = grad_output_re[bin];
        float dL_dout_im = grad_output_im[bin];

        float x = TWO_PI * (float(bin) - k0) / float(N_fft);

        Complex kernel = dirichlet_kernel(x, n);
        Complex kernel_grad = dirichlet_kernel_grad(x, n);

        // Apply phase correction
        Complex result = cmul(kernel, phase_corr);
        Complex result_grad = cmul(kernel_grad, phase_corr);

        // Gradient w.r.t. amplitude: dL/da = dL/dout_re * result.re + dL/dout_im * result.im
        dL_da += dL_dout_re * result.re + dL_dout_im * result.im;

        // Gradient w.r.t. x (through kernel)
        float dL_dx = amp * (dL_dout_re * result_grad.re + dL_dout_im * result_grad.im);

        // dx/dd = -TWO_PI * k0_per_meter / N_fft
        float dx_dd = -TWO_PI * k0_per_meter / float(N_fft);

        dL_dd += dL_dx * dx_dd;

        // Gradient w.r.t. phi_0 (through phase correction)
        // d(result)/d(phi_0) = kernel * d(phase_corr)/d(phi_0)
        //                    = kernel * i * phase_corr
        // So: d(result.re)/d(phi_0) = -result.im, d(result.im)/d(phi_0) = result.re
        float dL_dphi0 = amp * (-dL_dout_re * result.im + dL_dout_im * result.re);
        dL_dd += dL_dphi0 * dphi0_dd;
    }

    grad_d[i] = dL_dd;
    grad_a[i] = dL_da;
}

// ============================================================================
// Forward (per-target chunked): each thread handles one target, loops through ALL bins
// Output: output_re (num_chunks, M), output_im (num_chunks, M)
// ============================================================================

[AutoPyBindCUDA]
[CUDAKernel]
void forward_per_target(
    TensorView<float> d,          // (N,) distances (one-way)
    TensorView<float> a,          // (N,) amplitudes
    TensorView<float> output_re,  // (num_chunks, M) output real
    TensorView<float> output_im,  // (num_chunks, M) output imag
    float n,                      // Dirichlet order
    float k0_per_meter,
    int num_bins,                 // M (output size, = N_fft / 2)
    int N_fft,
    int num_targets,              // N
    int targets_per_chunk,
    float fc,
    float slope,
    float t_start                 // ADC start time in seconds
)
{
    uint thread_idx = cudaThreadIdx().x;
    uint chunk_idx = cudaBlockIdx().x;

    uint target_idx = chunk_idx * targets_per_chunk + thread_idx;
    if (target_idx >= num_targets) return;

    float c0 = 299792458.0f;

    float dist = d[target_idx];
    float amp = a[target_idx];

    if (amp == 0.0f) return;

    // Round-trip time of flight
    float tau = 2.0f * dist / c0;

    // Chirp initial phase
    float phi_0 = TWO_PI * (fc * tau + slope * tau * (t_start - 0.5f * tau));

    // k0 for this target
    float k0 = dist * k0_per_meter;

    // Precompute phase correction
    Complex phase_corr = cexp_f(phi_0);

    // Loop through all bins
    for (int bin = 0; bin < num_bins; bin++)
    {
        float x = TWO_PI * (float(bin) - k0) / float(N_fft);

        Complex kernel = dirichlet_kernel(x, n);
        Complex result = cmul(kernel, phase_corr);

        float contrib_re = amp * result.re;
        float contrib_im = amp * result.im;

        // Atomic add within chunk
        float oldVal;
        output_re.InterlockedAdd(uint2(chunk_idx, bin), contrib_re, oldVal);
        output_im.InterlockedAdd(uint2(chunk_idx, bin), contrib_im, oldVal);
    }
}

// ============================================================================
// Backward (per-bin chunked): each thread handles one bin, loops through targets in chunk
// Output: grad_d (num_chunks, N), grad_a (num_chunks, N) - summed in Python
// ============================================================================

[AutoPyBindCUDA]
[CUDAKernel]
void backward_per_bin(
    TensorView<float> d,              // (N,) distances
    TensorView<float> a,              // (N,) amplitudes
    TensorView<float> grad_output_re, // (M,) upstream gradient real
    TensorView<float> grad_output_im, // (M,) upstream gradient imag
    TensorView<float> grad_d,         // (num_chunks, N) output gradient for d
    TensorView<float> grad_a,         // (num_chunks, N) output gradient for a
    float n,
    float k0_per_meter,
    int num_bins,
    int N_fft,
    int num_targets,
    int bins_per_chunk,
    float fc,
    float slope,
    float t_start                     // ADC start time in seconds
)
{
    uint thread_idx = cudaThreadIdx().x;
    uint chunk_idx = cudaBlockIdx().x;

    uint bin = chunk_idx * bins_per_chunk + thread_idx;
    if (bin >= num_bins) return;

    float c0 = 299792458.0f;
    float dL_dout_re = grad_output_re[bin];
    float dL_dout_im = grad_output_im[bin];

    // Skip if no gradient from this bin
    if (dL_dout_re == 0.0f && dL_dout_im == 0.0f) return;

    for (int i = 0; i < num_targets; i++)
    {
        float dist = d[i];
        float amp = a[i];
        float k0 = dist * k0_per_meter;

        // Round-trip time of flight
        float tau = 2.0f * dist / c0;

        // Chirp initial phase
        float phi_0 = TWO_PI * (fc * tau + slope * tau * (t_start - 0.5f * tau));
        Complex phase_corr = cexp_f(phi_0);

        // d(phi_0)/d(dist) = 2*pi * 2/c0 * [fc + slope*t_start - slope*tau]
        float dphi0_dd = TWO_PI * 2.0f / c0 * (fc + slope * t_start - slope * tau);

        float x = TWO_PI * (float(bin) - k0) / float(N_fft);

        Complex kernel = dirichlet_kernel(x, n);
        Complex kernel_grad = dirichlet_kernel_grad(x, n);

        // Apply phase correction
        Complex result = cmul(kernel, phase_corr);
        Complex result_grad = cmul(kernel_grad, phase_corr);

        // Gradient w.r.t. amplitude
        float dL_da_contrib = dL_dout_re * result.re + dL_dout_im * result.im;

        // Gradient w.r.t. x (through kernel)
        float dL_dx = amp * (dL_dout_re * result_grad.re + dL_dout_im * result_grad.im);

        // dx/dd = -TWO_PI * k0_per_meter / N_fft
        float dx_dd = -TWO_PI * k0_per_meter / float(N_fft);

        float dL_dd_contrib = dL_dx * dx_dd;

        // Gradient w.r.t. phi_0 (through phase correction)
        float dL_dphi0 = amp * (-dL_dout_re * result.im + dL_dout_im * result.re);
        dL_dd_contrib += dL_dphi0 * dphi0_dd;

        // Atomic add within chunk (threads in same chunk write to same target)
        float oldVal;
        grad_d.InterlockedAdd(uint2(chunk_idx, i), dL_dd_contrib, oldVal);
        grad_a.InterlockedAdd(uint2(chunk_idx, i), dL_da_contrib, oldVal);
    }
}
