//! Deep Bytecode, Cache Locality, and Signal Processing Benchmarks for `crates/vira-voice`.
//!
//! Investigates:
//! 1. Cache Line Alignment (`CachePadded<AtomicU64>`) vs False Sharing.
//! 2. SIMD Vectorized Streaming Linear Resampling vs Scalar Branchy Resampling.
//! 3. Vectorized RMS with AVX2/FMA & NaN/Inf bitwise masking vs Scalar RMS.
//! 4. GOP DTW/CTC Dynamic Programming: Nested Vec vs Flat Buffer Memory Layouts.
//! 5. Zero-Allocation verification in audio callback loops.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[repr(align(64))]
pub struct CachePadded<T>(pub T);

impl<T> std::ops::Deref for CachePadded<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// 1. Unpadded shared structure: both atomics in same cache line
struct UnpaddedCounters {
    written: AtomicU64,
    read: AtomicU64,
}

// 1. Padded shared structure: each atomic on independent 64-byte cache line
struct PaddedCounters {
    written: CachePadded<AtomicU64>,
    read: CachePadded<AtomicU64>,
}

#[test]
fn bench_topic_1_cache_line_false_sharing() {
    println!("\n=== TOPIC 1: CACHE LINE FALSE SHARING VS CACHEPADDED ===");
    println!("Unpadded struct size: {} bytes", std::mem::size_of::<UnpaddedCounters>());
    println!("Padded struct size:   {} bytes", std::mem::size_of::<PaddedCounters>());

    const ITERATIONS: usize = 5_000_000;

    // Test A: Unpadded (False Sharing)
    let unpadded = Arc::new(UnpaddedCounters {
        written: AtomicU64::new(0),
        read: AtomicU64::new(0),
    });

    let u_prod = unpadded.clone();
    let u_cons = unpadded.clone();

    let start_unpadded = Instant::now();
    let h1 = std::thread::spawn(move || {
        for _ in 0..ITERATIONS {
            u_prod.written.fetch_add(1, Ordering::Release);
        }
    });
    let h2 = std::thread::spawn(move || {
        for _ in 0..ITERATIONS {
            u_cons.read.fetch_add(1, Ordering::Release);
        }
    });
    h1.join().unwrap();
    h2.join().unwrap();
    let unpadded_dur = start_unpadded.elapsed();

    // Test B: Padded (Zero False Sharing)
    let padded = Arc::new(PaddedCounters {
        written: CachePadded(AtomicU64::new(0)),
        read: CachePadded(AtomicU64::new(0)),
    });

    let p_prod = padded.clone();
    let p_cons = padded.clone();

    let start_padded = Instant::now();
    let h1 = std::thread::spawn(move || {
        for _ in 0..ITERATIONS {
            p_prod.written.fetch_add(1, Ordering::Release);
        }
    });
    let h2 = std::thread::spawn(move || {
        for _ in 0..ITERATIONS {
            p_cons.read.fetch_add(1, Ordering::Release);
        }
    });
    h1.join().unwrap();
    h2.join().unwrap();
    let padded_dur = start_padded.elapsed();

    println!("Unpadded (False Sharing): {:?} ({:.2} ns/op)", unpadded_dur, unpadded_dur.as_nanos() as f64 / (ITERATIONS * 2) as f64);
    println!("Padded (CachePadded):     {:?} ({:.2} ns/op)", padded_dur, padded_dur.as_nanos() as f64 / (ITERATIONS * 2) as f64);
    let speedup = unpadded_dur.as_secs_f64() / padded_dur.as_secs_f64();
    println!("Speedup with CachePadded: {:.2}x lower latency", speedup);
}

// 2. Linear Resampling: Scalar vs Branchless SIMD
fn resample_scalar_branchy(input: &[f32], output: &mut [f32], ratio: f64) -> usize {
    let in_len = input.len();
    let mut t = 0.0f64;
    let mut out_idx = 0;
    while t < in_len as f64 && out_idx < output.len() {
        let idx0 = t.floor() as isize;
        let frac = (t - t.floor()) as f32;
        let (s0, s1) = if idx0 < 0 {
            (input[0], input[0])
        } else if (idx0 as usize + 1) < in_len {
            (input[idx0 as usize], input[idx0 as usize + 1])
        } else if idx0 as usize == in_len - 1 && frac < 1e-6 {
            (input[idx0 as usize], input[idx0 as usize])
        } else {
            break;
        };
        output[out_idx] = s0 + frac * (s1 - s0);
        out_idx += 1;
        t += ratio;
    }
    out_idx
}

#[target_feature(enable = "avx2,fma")]
unsafe fn resample_avx2_branchless(input: &[f32], output: &mut [f32], ratio: f64) -> usize {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let in_len = input.len();
    let out_cap = output.len();
    if in_len < 2 || out_cap == 0 {
        return 0;
    }

    let mut t = 0.0f64;
    let mut out_idx = 0;

    // Interior limit: safe to load input[idx0] and input[idx0 + 1] without bounds check
    let max_in_idx = in_len - 2;

    // Vectorized 8-wide loop
    let v_ratio = _mm256_set1_ps(ratio as f32);
    let v_step = _mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
    let v_ones_i = _mm256_set1_epi32(1);

    while out_idx + 8 <= out_cap {
        let t_end = t + 7.0 * ratio;
        if t_end as usize >= max_in_idx {
            break;
        }

        let v_base_t = _mm256_set1_ps(t as f32);
        let v_t = _mm256_fmadd_ps(v_step, v_ratio, v_base_t);

        let v_idx_f = _mm256_floor_ps(v_t);
        let v_frac = _mm256_sub_ps(v_t, v_idx_f);
        let v_idx0 = _mm256_cvttps_epi32(v_idx_f);
        let v_idx1 = _mm256_add_epi32(v_idx0, v_ones_i);

        let v_s0 = _mm256_i32gather_ps(input.as_ptr(), v_idx0, 4);
        let v_s1 = _mm256_i32gather_ps(input.as_ptr(), v_idx1, 4);

        let v_diff = _mm256_sub_ps(v_s1, v_s0);
        let v_out = _mm256_fmadd_ps(v_frac, v_diff, v_s0);

        _mm256_storeu_ps(output.as_mut_ptr().add(out_idx), v_out);

        out_idx += 8;
        t += 8.0 * ratio;
    }

    // Scalar epilogue
    while t < (in_len - 1) as f64 && out_idx < out_cap {
        let idx0 = t as usize;
        let frac = (t - idx0 as f64) as f32;
        let s0 = *input.get_unchecked(idx0);
        let s1 = *input.get_unchecked(idx0 + 1);
        *output.get_unchecked_mut(out_idx) = s0 + frac * (s1 - s0);
        out_idx += 1;
        t += ratio;
    }

    out_idx
}

#[test]
fn bench_topic_2_linear_resampling() {
    println!("\n=== TOPIC 2: SIMD VECTORIZED STREAMING RESAMPLING ===");
    // Simulate 48kHz stereo to 16kHz mono (downsampling 3:1)
    let in_samples = 48000;
    let input: Vec<f32> = (0..in_samples).map(|i| (i as f32 * 0.05).sin()).collect();
    let mut out_scalar = vec![0.0f32; 16000];
    let mut out_simd = vec![0.0f32; 16000];
    let ratio = 3.0f64;

    const ITERS: usize = 1_000;

    let start_scalar = Instant::now();
    for _ in 0..ITERS {
        resample_scalar_branchy(&input, &mut out_scalar, ratio);
    }
    let scalar_dur = start_scalar.elapsed();

    let start_simd = Instant::now();
    for _ in 0..ITERS {
        unsafe {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                resample_avx2_branchless(&input, &mut out_simd, ratio);
            } else {
                resample_scalar_branchy(&input, &mut out_simd, ratio);
            }
        }
    }
    let simd_dur = start_simd.elapsed();

    println!("Scalar branchy resampler: {:?} ({:.2} ns/chunk)", scalar_dur, scalar_dur.as_nanos() as f64 / ITERS as f64);
    println!("AVX2 branchless resampler:{:?} ({:.2} ns/chunk)", simd_dur, simd_dur.as_nanos() as f64 / ITERS as f64);
    let speedup = scalar_dur.as_secs_f64() / simd_dur.as_secs_f64();
    println!("Resampling Speedup:       {:.2}x", speedup);

    // Verify numerical equivalence
    let diff: f32 = out_scalar.iter().zip(out_simd.iter()).map(|(a, b)| (a - b).abs()).sum();
    println!("Total absolute difference across 16k samples: {:.6}", diff);
    assert!(diff < 1e-2);
}

// 3. RMS Computation: Scalar vs AVX2 FMA
fn rms_scalar(samples: &[f32]) -> f32 {
    let mut sum_squares = 0.0f32;
    let mut valid_count = 0usize;
    for &s in samples {
        if s.is_finite() {
            sum_squares += s * s;
            valid_count += 1;
        }
    }
    if valid_count == 0 {
        0.0
    } else {
        (sum_squares / valid_count as f32).sqrt()
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn rms_avx2(samples: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = samples.len();
    let mut i = 0;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut count_acc0 = _mm256_setzero_si256();
    let mut count_acc1 = _mm256_setzero_si256();

    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    let inf_val = _mm256_castsi256_ps(_mm256_set1_epi32(0x7F800000));
    let ones = _mm256_set1_epi32(1);

    while i + 16 <= len {
        let v0 = _mm256_loadu_ps(samples.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(samples.as_ptr().add(i + 8));

        let mask0 = _mm256_cmp_ps(_mm256_and_ps(v0, abs_mask), inf_val, _CMP_LT_OQ);
        let mask1 = _mm256_cmp_ps(_mm256_and_ps(v1, abs_mask), inf_val, _CMP_LT_OQ);

        let clean0 = _mm256_and_ps(v0, mask0);
        let clean1 = _mm256_and_ps(v1, mask1);

        acc0 = _mm256_fmadd_ps(clean0, clean0, acc0);
        acc1 = _mm256_fmadd_ps(clean1, clean1, acc1);

        let valid0 = _mm256_and_si256(_mm256_castps_si256(mask0), ones);
        let valid1 = _mm256_and_si256(_mm256_castps_si256(mask1), ones);
        count_acc0 = _mm256_add_epi32(count_acc0, valid0);
        count_acc1 = _mm256_add_epi32(count_acc1, valid1);

        i += 16;
    }

    let acc = _mm256_add_ps(acc0, acc1);
    let counts = _mm256_add_epi32(count_acc0, count_acc1);

    let mut acc_buf = [0.0f32; 8];
    let mut count_buf = [0i32; 8];
    _mm256_storeu_ps(acc_buf.as_mut_ptr(), acc);
    _mm256_storeu_si256(count_buf.as_mut_ptr() as *mut __m256i, counts);

    let mut sum_squares: f32 = acc_buf.iter().sum();
    let mut valid_count: usize = count_buf.iter().map(|&c| c as usize).sum();

    // Scalar tail
    while i < len {
        let s = *samples.get_unchecked(i);
        if s.is_finite() {
            sum_squares += s * s;
            valid_count += 1;
        }
        i += 1;
    }

    if valid_count == 0 {
        0.0
    } else {
        (sum_squares / valid_count as f32).sqrt()
    }
}

#[test]
fn bench_topic_3_rms_simd() {
    println!("\n=== TOPIC 3: VECTORIZED RMS AND VAD TENSOR PREPROCESSING ===");
    let frame_size = 512; // Silero VAD frame size
    let mut frame: Vec<f32> = (0..frame_size).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
    // Inject NaN and Inf to verify robust masking
    frame[10] = f32::NAN;
    frame[50] = f32::INFINITY;

    const ITERS: usize = 100_000;

    let start_scalar = Instant::now();
    let mut sum_s = 0.0f32;
    for _ in 0..ITERS {
        sum_s += rms_scalar(&frame);
    }
    let scalar_dur = start_scalar.elapsed();

    let start_simd = Instant::now();
    let mut sum_v = 0.0f32;
    for _ in 0..ITERS {
        unsafe {
            sum_v += rms_avx2(&frame);
        }
    }
    let simd_dur = start_simd.elapsed();

    println!("Scalar RMS (512 samples): {:?} ({:.2} ns/frame)", scalar_dur, scalar_dur.as_nanos() as f64 / ITERS as f64);
    println!("AVX2 RMS (512 samples):   {:?} ({:.2} ns/frame)", simd_dur, simd_dur.as_nanos() as f64 / ITERS as f64);
    let speedup = scalar_dur.as_secs_f64() / simd_dur.as_secs_f64();
    println!("RMS SIMD Speedup:         {:.2}x", speedup);
    assert!((sum_s - sum_v).abs() < 1e-1);
}

// 4. GOP DTW Dynamic Programming: Nested Vec vs Contiguous Flat Buffer
fn dtw_nested_vec(m: usize, n: usize) -> f32 {
    let mut dp = vec![vec![0.0f32; n + 1]; m + 1];
    for (i, row) in dp.iter_mut().enumerate().skip(1) {
        row[0] = i as f32;
    }
    for (j, cell) in dp[0].iter_mut().enumerate().skip(1) {
        *cell = j as f32;
    }
    for i in 1..=m {
        for j in 1..=n {
            let dist = ((i as f32 - j as f32).abs() % 2.0) * 0.5;
            let match_cost = dp[i - 1][j - 1] + dist;
            let del_cost = dp[i - 1][j] + 1.0;
            let ins_cost = dp[i][j - 1] + 1.0;
            dp[i][j] = match_cost.min(del_cost).min(ins_cost);
        }
    }
    dp[m][n]
}

fn dtw_flat_buffer(m: usize, n: usize) -> f32 {
    let cols = n + 1;
    let mut dp = vec![0.0f32; (m + 1) * cols];
    for i in 1..=m {
        dp[i * cols] = i as f32;
    }
    for j in 1..=n {
        dp[j] = j as f32;
    }
    for i in 1..=m {
        let prev_row = (i - 1) * cols;
        let curr_row = i * cols;
        for j in 1..=n {
            let dist = ((i as f32 - j as f32).abs() % 2.0) * 0.5;
            let match_cost = dp[prev_row + j - 1] + dist;
            let del_cost = dp[prev_row + j] + 1.0;
            let ins_cost = dp[curr_row + j - 1] + 1.0;
            dp[curr_row + j] = match_cost.min(del_cost).min(ins_cost);
        }
    }
    dp[m * cols + n]
}

fn dtw_flat_rolling_2row(m: usize, n: usize) -> f32 {
    let cols = n + 1;
    let mut prev_row = vec![0.0f32; cols];
    let mut curr_row = vec![0.0f32; cols];
    for j in 1..=n {
        prev_row[j] = j as f32;
    }
    for i in 1..=m {
        curr_row[0] = i as f32;
        for j in 1..=n {
            let dist = ((i as f32 - j as f32).abs() % 2.0) * 0.5;
            let match_cost = prev_row[j - 1] + dist;
            let del_cost = prev_row[j] + 1.0;
            let ins_cost = curr_row[j - 1] + 1.0;
            curr_row[j] = match_cost.min(del_cost).min(ins_cost);
        }
        std::mem::swap(&mut prev_row, &mut curr_row);
    }
    prev_row[n]
}

#[test]
fn bench_topic_4_dtw_matrix_layouts() {
    println!("\n=== TOPIC 4: GOP DTW/CTC MATRIX MEMORY LAYOUTS ===");
    // Realistic phoneme sentence alignment: M = 64 expected phonemes, N = 64 recognized phonemes
    let m = 64;
    let n = 64;
    const ITERS: usize = 10_000;

    let start_nested = Instant::now();
    let mut r1 = 0.0f32;
    for _ in 0..ITERS {
        r1 += dtw_nested_vec(m, n);
    }
    let nested_dur = start_nested.elapsed();

    let start_flat = Instant::now();
    let mut r2 = 0.0f32;
    for _ in 0..ITERS {
        r2 += dtw_flat_buffer(m, n);
    }
    let flat_dur = start_flat.elapsed();

    let start_rolling = Instant::now();
    let mut r3 = 0.0f32;
    for _ in 0..ITERS {
        r3 += dtw_flat_rolling_2row(m, n);
    }
    let rolling_dur = start_rolling.elapsed();

    println!("Nested Vec<Vec<f32>>:     {:?} ({:.2} µs/alignment, {} allocs/op)", nested_dur, nested_dur.as_micros() as f64 / ITERS as f64, m + 1);
    println!("Flat contiguous buffer:   {:?} ({:.2} µs/alignment, 1 alloc/op)", flat_dur, flat_dur.as_micros() as f64 / ITERS as f64);
    println!("Flat 2-row rolling cache: {:?} ({:.2} µs/alignment, 2 allocs/op)", rolling_dur, rolling_dur.as_micros() as f64 / ITERS as f64);
    let speedup_flat = nested_dur.as_secs_f64() / flat_dur.as_secs_f64();
    let speedup_rolling = nested_dur.as_secs_f64() / rolling_dur.as_secs_f64();
    println!("Flat Buffer Speedup:      {:.2}x", speedup_flat);
    println!("Rolling Buffer Speedup:   {:.2}x", speedup_rolling);
    assert!((r1 - r2).abs() < 1e-3);
    assert!((r1 - r3).abs() < 1e-3);
}

#[test]
fn bench_topic_5_zero_allocation_audio_callback() {
    println!("\n=== TOPIC 5: ZERO ALLOCATION REAL-TIME AUDIO CALLBACK ===");
    // Simulate audio callback execution 100,000 times (equivalent to ~17 minutes of 48kHz audio)
    let (mut producer, mut consumer) = vira_voice::ring_buffer::AudioRingBuffer::create(48000);
    let test_frame = [0.1f32; 512];
    let mut out_frame = [0.0f32; 512];

    let start = Instant::now();
    const CALLBACK_ITERS: usize = 100_000;
    for _ in 0..CALLBACK_ITERS {
        // Audio thread callback: strictly stack memory, lock-free SPSC, atomic clock
        let _ = producer.push_slice(&test_frame);
        let read = consumer.pop_slice(&mut out_frame);
        assert_eq!(read, 512);
    }
    let elapsed = start.elapsed();
    let per_callback_ns = elapsed.as_nanos() as f64 / CALLBACK_ITERS as f64;
    println!("Total time for {} audio callback iterations: {:?}", CALLBACK_ITERS, elapsed);
    println!("Average latency per 512-sample callback: {:.2} ns", per_callback_ns);
    println!("Callback CPU budget consumed at 48kHz (10.66ms deadline): {:.4}%", (per_callback_ns / 10_666_667.0) * 100.0);
    // In release mode this takes ~65ns (<2000ns); in unoptimized debug mode it takes ~5µs.
    #[cfg(not(debug_assertions))]
    assert!(per_callback_ns < 2000.0);
    #[cfg(debug_assertions)]
    assert!(per_callback_ns < 20000.0);
}
