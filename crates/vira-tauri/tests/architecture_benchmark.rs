use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex as StdMutex, RwLock as StdRwLock};
use std::time::{Duration, Instant};

// ============================================================================
// 1. Serialization Benchmark Structs
// ============================================================================

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, PartialEq)]
struct JsonVisemePayload {
    aa: f32,
    ih: f32,
    ou: f32,
    ee: f32,
    oh: f32,
    sample_clock: u64,
    timestamp_ms: u64,
    is_speaking: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct BinaryVisemePayload {
    aa: f32,
    ih: f32,
    ou: f32,
    ee: f32,
    oh: f32,
    is_speaking: u8,
    _pad: [u8; 3],
    sample_clock: u64,
    timestamp_ms: u64,
}

impl BinaryVisemePayload {
    #[inline(always)]
    pub fn to_bytes(&self) -> [u8; 40] {
        let mut buf = [0u8; 40];
        buf[0..4].copy_from_slice(&self.aa.to_le_bytes());
        buf[4..8].copy_from_slice(&self.ih.to_le_bytes());
        buf[8..12].copy_from_slice(&self.ou.to_le_bytes());
        buf[12..16].copy_from_slice(&self.ee.to_le_bytes());
        buf[16..20].copy_from_slice(&self.oh.to_le_bytes());
        buf[20] = self.is_speaking;
        buf[21..24].copy_from_slice(&self._pad);
        buf[24..32].copy_from_slice(&self.sample_clock.to_le_bytes());
        buf[32..40].copy_from_slice(&self.timestamp_ms.to_le_bytes());
        buf
    }

    #[inline(always)]
    pub fn from_bytes(bytes: &[u8; 40]) -> Self {
        let aa = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let ih = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let ou = f32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let ee = f32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let oh = f32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let is_speaking = bytes[20];
        let _pad = [bytes[21], bytes[22], bytes[23]];
        let sample_clock = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
        let timestamp_ms = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
        Self {
            aa,
            ih,
            ou,
            ee,
            oh,
            is_speaking,
            _pad,
            sample_clock,
            timestamp_ms,
        }
    }
}

// ============================================================================
// Tests & Empirical Benchmarks
// ============================================================================

#[test]
fn bench_serialization_json_vs_binary() {
    let payload = JsonVisemePayload {
        aa: 0.85,
        ih: 0.12,
        ou: 0.05,
        ee: 0.0,
        oh: 0.0,
        sample_clock: 96000,
        timestamp_ms: 2000,
        is_speaking: true,
    };

    let bin_payload = BinaryVisemePayload {
        aa: 0.85,
        ih: 0.12,
        ou: 0.05,
        ee: 0.0,
        oh: 0.0,
        is_speaking: 1,
        _pad: [0; 3],
        sample_clock: 96000,
        timestamp_ms: 2000,
    };

    // Correctness check
    let json_str = serde_json::to_string(&payload).unwrap();
    let bin_bytes = bin_payload.to_bytes();
    let bin_decoded = BinaryVisemePayload::from_bytes(&bin_bytes);
    assert_eq!(bin_payload, bin_decoded);

    let iters = 200_000;

    // 1. JSON String Serialization (Current Tauri path)
    let start = Instant::now();
    let mut _json_bytes_total = 0;
    for _ in 0..iters {
        let s = serde_json::to_string(&payload).unwrap();
        _json_bytes_total += s.len();
    }
    let json_str_dur = start.elapsed();
    std::hint::black_box(_json_bytes_total);

    // 2. JSON Value Serialization (Current vira-tauri `to_value` step)
    let start = Instant::now();
    for _ in 0..iters {
        let v = serde_json::to_value(&payload).unwrap();
        std::hint::black_box(v);
    }
    let json_val_dur = start.elapsed();

    // 3. Raw Binary Pack (Proposed Zero-Copy ArrayBuffer path)
    let start = Instant::now();
    let mut _bin_bytes_total = 0;
    for _ in 0..iters {
        let b = bin_payload.to_bytes();
        _bin_bytes_total += b.len();
    }
    let bin_dur = start.elapsed();
    std::hint::black_box(_bin_bytes_total);

    println!("\n========================================================");
    println!("=== BENCHMARK 1: 60Hz Viseme IPC Serialization ({iters} iters) ===");
    println!("========================================================");
    println!(
        "JSON to_string : {:?} ({:.2} ns/op) | Wire Size: {} bytes",
        json_str_dur,
        json_str_dur.as_nanos() as f64 / iters as f64,
        json_str.len()
    );
    println!(
        "JSON to_value  : {:?} ({:.2} ns/op) | Allocates serde_json::Value tree",
        json_val_dur,
        json_val_dur.as_nanos() as f64 / iters as f64
    );
    println!(
        "Binary Packed  : {:?} ({:.2} ns/op) | Wire Size: {} bytes (0 allocations!)",
        bin_dur,
        bin_dur.as_nanos() as f64 / iters as f64,
        bin_bytes.len()
    );
    let speedup = json_str_dur.as_nanos() as f64 / bin_dur.as_nanos().max(1) as f64;
    let byte_reduction = (1.0 - (bin_bytes.len() as f64 / json_str.len() as f64)) * 100.0;
    println!(">>> SPEEDUP: {:.1}x faster", speedup);
    println!(">>> WIRE REDUCTION: {:.1}% smaller payload", byte_reduction);
    println!("========================================================\n");

    assert!(bin_dur < json_str_dur);
}

#[test]
fn bench_lock_contention_std_mutex_vs_atomic() {
    let num_threads = 8;
    let ops_per_thread = 50_000;

    // 1. std::sync::Mutex
    let std_mutex = Arc::new(StdMutex::new(0u8));
    let start = Instant::now();
    let mut handles = Vec::new();
    for _ in 0..num_threads {
        let m = std_mutex.clone();
        handles.push(std::thread::spawn(move || {
            for _ in 0..ops_per_thread {
                let mut guard = m.lock().unwrap();
                *guard = (*guard).wrapping_add(1);
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let std_mutex_dur = start.elapsed();

    // 2. std::sync::RwLock (typical read-heavy 90% read, 10% write)
    let std_rwlock = Arc::new(StdRwLock::new(0u8));
    let start = Instant::now();
    let mut handles = Vec::new();
    for t in 0..num_threads {
        let rw = std_rwlock.clone();
        handles.push(std::thread::spawn(move || {
            if t == 0 {
                // Writer thread
                for _ in 0..ops_per_thread {
                    let mut guard = rw.write().unwrap();
                    *guard = (*guard).wrapping_add(1);
                }
            } else {
                // Reader threads
                for _ in 0..ops_per_thread {
                    let guard = rw.read().unwrap();
                    std::hint::black_box(*guard);
                }
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let std_rwlock_dur = start.elapsed();

    // 3. Lock-free AtomicU8 (Proposed for ConversationState)
    let atomic_val = Arc::new(AtomicU8::new(0));
    let start = Instant::now();
    let mut handles = Vec::new();
    for t in 0..num_threads {
        let a = atomic_val.clone();
        handles.push(std::thread::spawn(move || {
            if t == 0 {
                // Writer thread
                for _ in 0..ops_per_thread {
                    a.fetch_add(1, Ordering::Release);
                }
            } else {
                // Reader threads
                for _ in 0..ops_per_thread {
                    let val = a.load(Ordering::Acquire);
                    std::hint::black_box(val);
                }
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let atomic_dur = start.elapsed();

    let total_ops = num_threads * ops_per_thread;
    println!("========================================================");
    println!("=== BENCHMARK 2: Lock Contention ({num_threads} threads, {total_ops} ops) ===");
    println!("========================================================");
    println!(
        "std::sync::Mutex   : {:?} ({:.2} ns/op)",
        std_mutex_dur,
        std_mutex_dur.as_nanos() as f64 / total_ops as f64
    );
    println!(
        "std::sync::RwLock  : {:?} ({:.2} ns/op) [1 writer, 7 readers]",
        std_rwlock_dur,
        std_rwlock_dur.as_nanos() as f64 / total_ops as f64
    );
    println!(
        "AtomicU8 Wait-Free : {:?} ({:.2} ns/op) [1 writer, 7 readers]",
        atomic_dur,
        atomic_dur.as_nanos() as f64 / total_ops as f64
    );
    let atomic_speedup = std_rwlock_dur.as_nanos() as f64 / atomic_dur.as_nanos().max(1) as f64;
    println!(">>> SPEEDUP (Atomic vs StdRwLock): {:.1}x faster", atomic_speedup);
    println!("========================================================\n");

    assert!(atomic_dur < std_mutex_dur);
}

#[tokio::test]
async fn bench_event_conflation_vs_unbounded_queue() {
    // Test comparing unbounded queuing (where slow consumer causes massive latency lag)
    // vs Atomic Latest-Slot (conflated sampling, 0 lag).

    // Pattern A: Bounded queue with overflow / lag
    let (tx, mut rx) = tokio::sync::mpsc::channel::<u64>(1000);
    let sent_count = 5000;

    let producer = tokio::spawn(async move {
        for i in 0..sent_count {
            let _ = tx.send(i).await;
        }
    });

    let consumer = tokio::spawn(async move {
        let mut received = 0;
        while let Some(_val) = rx.recv().await {
            received += 1;
            // simulate rendering work or slow IPC
            tokio::task::yield_now().await;
            if received == sent_count {
                break;
            }
        }
        received
    });

    let start_queue = Instant::now();
    let _ = tokio::join!(producer, consumer);
    let queue_dur = start_queue.elapsed();

    // Pattern B: Atomic Conflated Slot (AtomicU64)
    let slot = Arc::new(AtomicU64::new(0));
    let stop_flag = Arc::new(AtomicBool::new(false));

    let slot_prod = slot.clone();
    let stop_prod = stop_flag.clone();
    let prod_handle = tokio::spawn(async move {
        for i in 0..sent_count {
            slot_prod.store(i, Ordering::Release);
        }
        stop_prod.store(true, Ordering::Release);
    });

    let slot_cons = slot.clone();
    let stop_cons = stop_flag.clone();
    let cons_handle = tokio::spawn(async move {
        let mut sample_count = 0;
        let mut last_seen = u64::MAX;
        while !stop_cons.load(Ordering::Acquire) {
            let val = slot_cons.load(Ordering::Acquire);
            if val != last_seen {
                last_seen = val;
                sample_count += 1;
            }
            tokio::task::yield_now().await;
        }
        sample_count
    });

    let start_slot = Instant::now();
    let (_, samples) = tokio::join!(prod_handle, cons_handle);
    let slot_dur = start_slot.elapsed();

    println!("========================================================");
    println!("=== BENCHMARK 3: Event Backpressure & Conflation ===");
    println!("========================================================");
    println!("MPSC Queue (5000 events)   : {:?} (enforces full FIFO delivery)", queue_dur);
    println!(
        "Conflated Slot (5000 events) : {:?} (consumed {} fresh frames, dropped stale)",
        slot_dur,
        samples.unwrap()
    );
    println!("========================================================\n");
}

#[tokio::test]
async fn bench_tokio_interval_precision_under_load() {
    let target_interval = Duration::from_millis(16); // 60Hz
    let num_ticks = 50;

    // Simulate background CPU compute thread (like ONNX inference or TTS)
    let stop_load = Arc::new(AtomicBool::new(false));
    let stop_load_clone = stop_load.clone();
    let load_handle = std::thread::spawn(move || {
        let mut x = 1.0f32;
        while !stop_load_clone.load(Ordering::Relaxed) {
            for _ in 0..10_000 {
                x = (x * 1.0001).sin().cos();
            }
            std::hint::black_box(x);
        }
    });

    let mut interval = tokio::time::interval(target_interval);
    interval.tick().await; // first tick returns immediately

    let mut intervals = Vec::with_capacity(num_ticks);
    let mut last_tick = Instant::now();

    for _ in 0..num_ticks {
        interval.tick().await;
        let now = Instant::now();
        let elapsed = now.duration_since(last_tick);
        last_tick = now;
        intervals.push(elapsed);
    }

    stop_load.store(true, Ordering::Relaxed);
    load_handle.join().unwrap();

    let diffs_us: Vec<i64> = intervals
        .iter()
        .map(|d| (d.as_micros() as i64) - (target_interval.as_micros() as i64))
        .collect();

    let mean_us = diffs_us.iter().sum::<i64>() as f64 / diffs_us.len() as f64;
    let max_jitter_us = diffs_us.iter().map(|d| d.abs()).max().unwrap_or(0);

    println!("========================================================");
    println!("=== BENCHMARK 4: Tokio 60Hz Interval Precision Under CPU Load ===");
    println!("========================================================");
    println!("Target Interval : 16,000 µs (16.00 ms)");
    println!("Mean Deviation  : {:.2} µs ({:.3} ms)", mean_us, mean_us / 1000.0);
    println!("Max Jitter      : {} µs ({:.3} ms)", max_jitter_us, max_jitter_us as f64 / 1000.0);
    println!("========================================================\n");
}
