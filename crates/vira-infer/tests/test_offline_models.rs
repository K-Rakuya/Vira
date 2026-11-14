use std::fs::{create_dir_all, File};
use std::io::Write;
use tempfile::tempdir;
use vira_infer::OfflineModelPaths;

#[test]
fn test_offline_models_path_resolution_and_defaults() {
    let dir = OfflineModelPaths::default_dir();
    assert!(!dir.as_os_str().is_empty());

    let empty_temp = tempdir().expect("tempdir");
    let paths = OfflineModelPaths::new(empty_temp.path());
    assert!(!paths.is_piper_available());
    assert!(!paths.is_kokoro_available());
    assert!(!paths.is_tts_available());
    assert!(!paths.is_stt_available());
    assert!(!paths.is_vad_available());
}

#[test]
fn test_offline_models_mock_files_detection() {
    let temp = tempdir().expect("tempdir");
    let base = temp.path().to_path_buf();

    let paths = OfflineModelPaths::new(&base);
    assert!(!paths.is_tts_available());

    // Create Piper mock model and tokens
    let piper_dir = base.join("piper");
    create_dir_all(&piper_dir).expect("create piper dir");
    File::create(piper_dir.join("vi_VN-vais1000-medium.onnx"))
        .expect("create file")
        .write_all(b"mock onnx")
        .expect("write");
    File::create(piper_dir.join("tokens.txt"))
        .expect("create tokens")
        .write_all(b"mock tokens")
        .expect("write");

    assert!(paths.is_piper_available());
    assert!(paths.is_tts_available());
    assert!(!paths.is_kokoro_available());

    // Create Kokoro mock model, voices, and tokens
    let kokoro_dir = base.join("kokoro");
    create_dir_all(&kokoro_dir).expect("create kokoro dir");
    File::create(kokoro_dir.join("kokoro-multi-lang-v1_0.onnx"))
        .expect("create model")
        .write_all(b"mock kokoro")
        .expect("write");
    File::create(kokoro_dir.join("voices.bin"))
        .expect("create voices")
        .write_all(b"mock voices")
        .expect("write");
    File::create(kokoro_dir.join("tokens.txt"))
        .expect("create tokens")
        .write_all(b"mock tokens")
        .expect("write");

    assert!(paths.is_kokoro_available());

    // Create Zipformer mock files
    let stt_dir = base.join("streaming-zipformer");
    create_dir_all(&stt_dir).expect("create stt dir");
    File::create(stt_dir.join("encoder.onnx")).expect("create");
    File::create(stt_dir.join("decoder.onnx")).expect("create");
    File::create(stt_dir.join("joiner.onnx")).expect("create");
    File::create(stt_dir.join("tokens.txt")).expect("create");

    assert!(paths.is_stt_available());
}
