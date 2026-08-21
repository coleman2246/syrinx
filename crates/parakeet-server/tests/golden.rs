//! Golden-audio tests against the real GPU backend.
//!
//! Requires the `cuda` feature, a GPU, and the model on disk, so these are
//! `#[ignore]`d and excluded from a normal `cargo test`. Run explicitly:
//!
//! ```text
//! PARAKEET_MODEL_DIR=~/.local/share/parakeet-dictation/nemotron \
//! ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so \
//!   cargo test -p parakeet-server --features cuda --test golden -- --ignored --nocapture
//! ```

#![cfg(feature = "cuda")]

use parakeet_proto::Mode;
use parakeet_server::asr::parakeet::ParakeetBackend;
use parakeet_server::session::Session;
use std::path::PathBuf;

fn model_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("PARAKEET_MODEL_DIR")
            .expect("set PARAKEET_MODEL_DIR to the Nemotron model directory"),
    )
}

fn load_fixture(name: &str) -> Vec<f32> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden")
        .join(name);
    let mut r = hound::WavReader::open(&path).expect("open fixture");
    let spec = r.spec();
    assert_eq!(spec.sample_rate, 16_000, "fixture must be 16 kHz");
    assert_eq!(spec.channels, 1, "fixture must be mono");
    r.samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect()
}

#[test]
#[ignore = "requires a GPU and the model on disk"]
fn transcribes_golden_audio() {
    let backend = ParakeetBackend::load_cuda(&model_dir()).expect("load model on CUDA");
    let mut session = Session::new(Mode::Live, &backend, "golden".into());

    let audio = load_fixture("fox.wav");
    let mut text = String::new();
    for m in session.push_audio(&audio).unwrap() {
        if let parakeet_proto::ServerMessage::TranscriptCommit { text: t, .. } = m {
            text.push_str(&t);
        }
    }
    for m in session.finish().unwrap() {
        if let parakeet_proto::ServerMessage::TranscriptCommit { text: t, .. } = m {
            text.push_str(&t);
        }
    }

    let got = text.to_lowercase();
    println!("golden transcript: {got:?}");

    // The fixture is espeak-synthesised, which is robotic enough that a couple
    // of words are reliably misheard ("brown" -> "round"). Asserting exact text
    // would pin those artifacts rather than model quality, and asserting
    // individual words is brittle for the same reason.
    //
    // A word-overlap ratio still fails loudly if the model regresses or the
    // execution provider silently changes, without breaking on a single
    // stubborn word.
    let expected = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"];
    let hits = expected.iter().filter(|w| got.contains(*w)).count();
    let ratio = hits as f32 / expected.len() as f32;
    assert!(
        ratio >= 0.7,
        "only {hits}/{} expected words present (ratio {ratio:.2}): {got:?}",
        expected.len()
    );
}

#[test]
#[ignore = "requires a GPU and the model on disk"]
fn streams_share_one_model_and_decode_independently() {
    // The property that makes multi-client viable on 8 GB: one set of weights,
    // independent decoder state per session.
    let backend = ParakeetBackend::load_cuda(&model_dir()).expect("load model on CUDA");
    let audio = load_fixture("fox.wav");

    let mut a = Session::new(Mode::Live, &backend, "a".into());
    let mut b = Session::new(Mode::Live, &backend, "b".into());

    let collect = |s: &mut Session, audio: &[f32]| -> String {
        let mut out = String::new();
        for m in s.push_audio(audio).unwrap().into_iter().chain(s.finish().unwrap()) {
            if let parakeet_proto::ServerMessage::TranscriptCommit { text, .. } = m {
                out.push_str(&text);
            }
        }
        out
    };

    let ta = collect(&mut a, &audio);
    let tb = collect(&mut b, &audio);
    assert_eq!(
        ta.trim(),
        tb.trim(),
        "two streams over one shared model must decode identically"
    );
}
