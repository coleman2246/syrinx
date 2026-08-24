//! C ABI over the syrinx client, for Swift.
//!
//! The division is the one the desktop clients already use: everything from
//! the microphone onwards is Rust -- protocol, WebSocket, TLS, streaming
//! state, reconnection -- and the platform supplies only what genuinely needs
//! native APIs. On iOS that is audio capture, the application lifecycle, and
//! the keyboard.
//!
//! Audio arrives through [`syrinx_push_audio`] rather than being captured
//! here. A keyboard extension cannot open the microphone at all, so capture
//! has to live on the Swift side regardless; `SessionOptions::external_audio`
//! is the seam that makes that possible without the session knowing.
//!
//! # Contract for the caller
//!
//! - Samples are **16 kHz mono f32**. Resample before pushing; the session
//!   does not, because every capture backend already normalises.
//! - Every `char *` returned is owned by the caller and must be released with
//!   [`syrinx_string_free`]. They are not static.
//! - A handle is not thread-safe to *free* concurrently, but pushing audio and
//!   taking text from different threads is fine -- the state behind it is
//!   mutex-guarded, which is what lets an audio callback push while the UI
//!   polls.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float};
use std::ptr;

use syrinx_client::mode::OutputMode;
use syrinx_client::session::{SessionHandle, SessionOptions, Status};

/// An open session. Opaque to Swift.
pub struct SyrinxSession {
    handle: SessionHandle,
    audio: tokio::sync::mpsc::Sender<Vec<f32>>,
    /// How much of the transcript the caller has already been given, in
    /// characters. Text is handed over once and never repeated: a keyboard
    /// inserts what it receives, and re-delivering a fragment would type it
    /// twice.
    delivered: usize,
}

/// Session status, matching `syrinx_client::session::Status`.
#[repr(i32)]
pub enum SyrinxStatus {
    Idle = 0,
    Connecting = 1,
    Listening = 2,
    Stopping = 3,
    Transcribing = 4,
}

/// Start a session against `url` with `token`.
///
/// Returns null if either argument is not valid UTF-8. The connection itself
/// is established asynchronously, so a non-null return means "starting", not
/// "connected" -- poll [`syrinx_status`] and [`syrinx_take_error`].
///
/// # Safety
/// `url` and `token` must be valid, NUL-terminated C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syrinx_start(
    url: *const c_char,
    token: *const c_char,
) -> *mut SyrinxSession {
    let (Some(url), Some(token)) = (unsafe { as_str(url) }, unsafe { as_str(token) }) else {
        return ptr::null_mut();
    };

    // rustls refuses to guess its backend and panics at the first connection
    // if none is installed -- which would take the host application down
    // mid-sentence rather than return an error.
    syrinx_client::install_crypto_provider();

    // Bounded: if the network stalls, the audio callback must not be able to
    // grow this without limit. Dropping the oldest audio is the right failure
    // for dictation -- 32 chunks is about eighteen seconds.
    let (tx, rx) = tokio::sync::mpsc::channel::<Vec<f32>>(32);

    let handle = syrinx_client::session::start(
        SessionOptions {
            url: url.to_string(),
            token: token.to_string(),
            // Ignored when external_audio is set, but the field is not
            // optional.
            sources: Vec::new(),
            // Transcribe, never Type: typing at the cursor is the host
            // application's job through UITextDocumentProxy, and the injection
            // backends are all desktop ones.
            mode: OutputMode::Transcribe,
            // The keyboard extension has no config file and no UI to expose
            // this, and iOS gets no diarization models. A future host app
            // wanting labels wires its own toggle through this crate.
            diarize: false,
            label: None,
            inject: Default::default(),
            stream: None,
            external_audio: Some(rx),
        },
        || {},
    );

    Box::into_raw(Box::new(SyrinxSession {
        handle,
        audio: tx,
        delivered: 0,
    }))
}

/// Feed 16 kHz mono samples.
///
/// Returns false if the session has ended or the queue is full. A full queue
/// means the network is not keeping up; the caller should drop the buffer
/// rather than block, since an audio callback that blocks will glitch.
///
/// # Safety
/// `samples` must point to `len` floats, and `session` must come from
/// [`syrinx_start`] and not yet be stopped.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syrinx_push_audio(
    session: *mut SyrinxSession,
    samples: *const c_float,
    len: usize,
) -> bool {
    let Some(s) = (unsafe { session.as_mut() }) else {
        return false;
    };
    if samples.is_null() || len == 0 {
        return true;
    }
    let slice = unsafe { std::slice::from_raw_parts(samples, len) };
    // try_send, never send: this is called from a realtime audio thread.
    s.audio.try_send(slice.to_vec()).is_ok()
}

/// Take transcript text not yet handed over, or null if there is none.
///
/// # Safety
/// `session` must come from [`syrinx_start`]. Free the result with
/// [`syrinx_string_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syrinx_take_text(session: *mut SyrinxSession) -> *mut c_char {
    let Some(s) = (unsafe { session.as_mut() }) else {
        return ptr::null_mut();
    };
    let transcript = s.handle.state().transcript;
    // Counted in chars, not bytes: slicing a UTF-8 string on a byte index can
    // land inside a multi-byte character and panic.
    let fresh: String = transcript.chars().skip(s.delivered).collect();
    if fresh.is_empty() {
        return ptr::null_mut();
    }
    s.delivered = transcript.chars().count();
    to_c(fresh)
}

/// Current status.
///
/// # Safety
/// `session` must come from [`syrinx_start`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syrinx_status(session: *mut SyrinxSession) -> i32 {
    let Some(s) = (unsafe { session.as_ref() }) else {
        return SyrinxStatus::Idle as i32;
    };
    match s.handle.state().status {
        Status::Idle => SyrinxStatus::Idle as i32,
        Status::Connecting => SyrinxStatus::Connecting as i32,
        Status::Listening => SyrinxStatus::Listening as i32,
        Status::Stopping => SyrinxStatus::Stopping as i32,
        Status::Transcribing => SyrinxStatus::Transcribing as i32,
    }
}

/// Copy the current spectrum into `out`, returning how many bands were written.
///
/// The same bands the desktop overlay draws, computed by the session from the
/// audio it is actually sending, so the phone's meter cannot disagree with
/// what was transcribed. Recomputing them on the Swift side would be a second
/// implementation of something that already exists and already runs.
///
/// Writes nothing and returns 0 when there is no session or no room, so a
/// caller that ignores the result draws an empty meter rather than reading
/// uninitialised memory.
///
/// # Safety
/// `out` must point to `cap` floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syrinx_levels(
    session: *mut SyrinxSession,
    out: *mut c_float,
    cap: usize,
) -> usize {
    let Some(s) = (unsafe { session.as_ref() }) else {
        return 0;
    };
    if out.is_null() || cap == 0 {
        return 0;
    }
    let levels = s.handle.state().levels;
    let n = levels.len().min(cap);
    unsafe { ptr::copy_nonoverlapping(levels.as_ptr(), out, n) };
    n
}

/// The session's error, or null. Does not clear it.
///
/// # Safety
/// `session` must come from [`syrinx_start`]. Free the result with
/// [`syrinx_string_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syrinx_take_error(session: *mut SyrinxSession) -> *mut c_char {
    let Some(s) = (unsafe { session.as_ref() }) else {
        return ptr::null_mut();
    };
    match s.handle.state().error {
        Some(e) => to_c(e),
        None => ptr::null_mut(),
    }
}

/// Stop the session and release it. The pointer is invalid afterwards.
///
/// # Safety
/// `session` must come from [`syrinx_start`] and must not be used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syrinx_stop(session: *mut SyrinxSession) {
    if session.is_null() {
        return;
    }
    let mut s = unsafe { Box::from_raw(session) };
    s.handle.stop();
}

/// Release a string returned by this library.
///
/// # Safety
/// `s` must have come from one of the functions above, and be freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syrinx_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { CString::from_raw(s) });
    }
}

/// The library version, for checking the framework matches the app.
#[unsafe(no_mangle)]
pub extern "C" fn syrinx_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// The sample rate the caller must resample to.
#[unsafe(no_mangle)]
pub extern "C" fn syrinx_sample_rate() -> u32 {
    syrinx_proto::SAMPLE_RATE
}

unsafe fn as_str<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(p) }.to_str().ok()
}

/// Interior NULs cannot survive a C string; a transcript containing one would
/// otherwise silently truncate at that point.
fn to_c(s: String) -> *mut c_char {
    match CString::new(s.replace('\0', "")) {
        Ok(c) => c.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_null_url_is_refused_rather_than_dereferenced() {
        let token = CString::new("t").unwrap();
        assert!(unsafe { syrinx_start(ptr::null(), token.as_ptr()) }.is_null());
    }

    #[test]
    fn pushing_to_a_null_session_is_not_a_crash() {
        // Swift can call this after stop in a race; it must be inert.
        let samples = [0.0f32; 4];
        assert!(!unsafe { syrinx_push_audio(ptr::null_mut(), samples.as_ptr(), 4) });
        assert!(unsafe { syrinx_take_text(ptr::null_mut()) }.is_null());
        assert!(unsafe { syrinx_take_error(ptr::null_mut()) }.is_null());
        unsafe { syrinx_stop(ptr::null_mut()) };
        unsafe { syrinx_string_free(ptr::null_mut()) };
    }

    #[test]
    fn strings_survive_the_round_trip_and_free() {
        let p = to_c("hello \u{e9} \u{1F600}".into());
        assert!(!p.is_null());
        let back = unsafe { CStr::from_ptr(p) }.to_str().unwrap().to_string();
        assert_eq!(back, "hello \u{e9} \u{1F600}");
        unsafe { syrinx_string_free(p) };
    }

    #[test]
    fn an_interior_nul_does_not_truncate_the_transcript() {
        // CString::new rejects interior NULs; returning null there would lose
        // the text entirely, so they are stripped instead.
        let p = to_c("before\u{0}after".into());
        assert!(!p.is_null());
        let back = unsafe { CStr::from_ptr(p) }.to_str().unwrap().to_string();
        assert_eq!(back, "beforeafter");
        unsafe { syrinx_string_free(p) };
    }

    #[test]
    fn the_sample_rate_matches_the_protocol() {
        // Swift resamples to whatever this says; a disagreement would be
        // chipmunk audio, not an error.
        assert_eq!(syrinx_sample_rate(), 16_000);
    }

    #[test]
    fn the_version_string_is_nul_terminated() {
        let v = unsafe { CStr::from_ptr(syrinx_version()) }.to_str().unwrap();
        assert!(!v.is_empty());
    }

    #[test]
    fn levels_never_write_past_the_buffer() {
        // A null session is the case a caller hits after stopping, and it has
        // to leave the buffer alone rather than half-fill it.
        let mut buf = [7.0f32; 4];
        let n = unsafe { syrinx_levels(ptr::null_mut(), buf.as_mut_ptr(), buf.len()) };
        assert_eq!(n, 0);
        assert_eq!(buf, [7.0; 4]);

        let n = unsafe { syrinx_levels(ptr::null_mut(), ptr::null_mut(), 0) };
        assert_eq!(n, 0);
    }
}
