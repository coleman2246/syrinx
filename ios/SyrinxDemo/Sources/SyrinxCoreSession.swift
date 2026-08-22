import Foundation
import SyrinxCore

/// Swift face of the Rust core.
///
/// The Rust side owns the protocol, the WebSocket, TLS and the streaming
/// state. This type exists only to make that safe to touch from Swift: it
/// guarantees the handle is freed exactly once and that every `char *` the
/// library hands back is released.
final class SyrinxCoreSession {
    /// Mirrors the Rust `Status`.
    enum Status: Int32 {
        case idle = 0, connecting = 1, listening = 2, stopping = 3, transcribing = 4

        var label: String {
            switch self {
            case .idle: return "Idle"
            case .connecting: return "Connecting…"
            case .listening: return "Listening"
            case .stopping: return "Stopping…"
            case .transcribing: return "Transcribing"
            }
        }
    }

    private var handle: OpaquePointer?

    /// The rate the core expects. Read from Rust rather than hardcoded, so the
    /// two cannot disagree — a mismatch here is not an error, it is chipmunks.
    static var sampleRate: Double { Double(syrinx_sample_rate()) }

    init?(url: String, token: String) {
        guard let h = url.withCString({ u in
            token.withCString { t in syrinx_start(u, t) }
        }) else { return nil }
        handle = h
    }

    /// Push 16 kHz mono samples. Returns false when the core cannot take them,
    /// which on a full queue means the network is behind — the caller should
    /// drop the buffer rather than wait, because this runs on an audio thread.
    @discardableResult
    func push(_ samples: UnsafePointer<Float>, count: Int) -> Bool {
        guard let h = handle else { return false }
        return syrinx_push_audio(h, samples, count)
    }

    /// Transcript text not yet seen. Never repeats, so the caller appends.
    func takeText() -> String? {
        guard let h = handle, let c = syrinx_take_text(h) else { return nil }
        defer { syrinx_string_free(c) }
        return String(cString: c)
    }

    func takeError() -> String? {
        guard let h = handle, let c = syrinx_take_error(h) else { return nil }
        defer { syrinx_string_free(c) }
        return String(cString: c)
    }

    /// The current spectrum, as the desktop overlay draws it.
    ///
    /// Sized generously and trimmed to what Rust reports, so adding a band on
    /// that side does not silently truncate here.
    func levels() -> [Float] {
        guard let h = handle else { return [] }
        var buf = [Float](repeating: 0, count: 32)
        let n = buf.withUnsafeMutableBufferPointer { syrinx_levels(h, $0.baseAddress, $0.count) }
        return Array(buf.prefix(n))
    }

    var status: Status {
        guard let h = handle else { return .idle }
        return Status(rawValue: syrinx_status(h)) ?? .idle
    }

    /// Idempotent: nulling the handle first means a second call, or a call
    /// racing with deinit, cannot double-free.
    func stop() {
        guard let h = handle else { return }
        handle = nil
        syrinx_stop(h)
    }

    deinit { stop() }
}
