import AVFoundation

/// Microphone capture, resampled to what the core wants.
///
/// This is the part that genuinely has to be native. Everything downstream of
/// the callback is Rust.
///
/// The hardware format is not negotiable — the input node reports whatever the
/// device is doing, typically 48 kHz stereo — so a converter sits between it
/// and the 16 kHz mono the model expects. Getting this wrong does not fail
/// loudly; it transcribes gibberish, which is much harder to diagnose than a
/// crash.
final class AudioCapture {
    private let engine = AVAudioEngine()
    private var converter: AVAudioConverter?
    private let target: AVAudioFormat

    /// Called on the audio thread with 16 kHz mono samples. Must not block.
    private let onSamples: (UnsafePointer<Float>, Int) -> Void

    init?(onSamples: @escaping (UnsafePointer<Float>, Int) -> Void) {
        guard let fmt = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: SyrinxCoreSession.sampleRate,
            channels: 1,
            interleaved: false
        ) else { return nil }
        self.target = fmt
        self.onSamples = onSamples
    }

    func start() throws {
        do {
            try attempt(mode: .measurement)
        } catch {
            // .measurement suppresses the input processing that would fight
            // the recogniser, which is why it is tried first. It also pins the
            // route hard, and a route the engine cannot open fails with 'what'
            // and nothing further. The plain mode transcribes slightly worse
            // than not transcribing at all.
            AudioSession.note("measurement mode failed: \(AudioSession.describe(error))")
            teardown()
            try attempt(mode: .default)
        }
    }

    private func attempt(mode: AVAudioSession.Mode) throws {
        // Already active since launch in the normal case. Activating here is
        // the call iOS refuses in the background, so it must not be the first
        // time it happens.
        try AudioSession.ensureActive(mode: mode)

        let input = engine.inputNode
        let hardware = input.outputFormat(forBus: 0)
        guard hardware.sampleRate > 0 else {
            throw NSError(domain: "Syrinx", code: 1, userInfo: [
                NSLocalizedDescriptionKey:
                    "the microphone reported no format — is another app holding it?"
            ])
        }
        converter = AVAudioConverter(from: hardware, to: target)

        let ratio = target.sampleRate / hardware.sampleRate
        input.installTap(onBus: 0, bufferSize: 4096, format: hardware) { [weak self] buf, _ in
            guard let self, let converter = self.converter else { return }

            // Capacity must be computed from the ratio, rounded up: too small
            // and the converter silently truncates.
            let capacity = AVAudioFrameCount((Double(buf.frameLength) * ratio).rounded(.up)) + 1
            guard let out = AVAudioPCMBuffer(pcmFormat: self.target, frameCapacity: capacity)
            else { return }

            var pushed = false
            var err: NSError?
            converter.convert(to: out, error: &err) { _, status in
                // The converter asks repeatedly; hand over the input once and
                // then report endOfStream, or it spins.
                if pushed {
                    status.pointee = .noDataNow
                    return nil
                }
                pushed = true
                status.pointee = .haveData
                return buf
            }
            if err != nil { return }

            guard out.frameLength > 0, let ch = out.floatChannelData?[0] else { return }
            self.onSamples(ch, Int(out.frameLength))
        }

        engine.prepare()
        do {
            try engine.start()
        } catch {
            // Distinct from the session failing: the session can be perfectly
            // healthy and the engine still refuse, and the two need different
            // answers.
            throw NSError(domain: "Syrinx.audio", code: (error as NSError).code, userInfo: [
                NSLocalizedDescriptionKey: "starting the audio engine: \(AudioSession.describe(error))"
            ])
        }
    }

    /// Undo a failed attempt, so the next one starts from a clean engine
    /// rather than one carrying a tap and a half-built graph.
    private func teardown() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        engine.reset()
        converter = nil
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        converter = nil
        // The session outlives capture: dropping it would mean activating
        // again for the next one, from the background, which cannot be done.
    }

    /// Decoding lives with the session, which is where most of these come
    /// from; this only forwards so callers have one place to ask.
    static func describe(_ error: Error) -> String { AudioSession.describe(error) }

    /// Ask for microphone permission. Without this the tap delivers silence
    /// rather than failing, which looks exactly like a broken server.
    static func requestPermission(_ done: @escaping (Bool) -> Void) {
        if #available(iOS 17.0, *) {
            AVAudioApplication.requestRecordPermission { ok in
                DispatchQueue.main.async { done(ok) }
            }
        } else {
            AVAudioSession.sharedInstance().requestRecordPermission { ok in
                DispatchQueue.main.async { done(ok) }
            }
        }
    }
}
