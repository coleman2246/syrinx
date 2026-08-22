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
        let session = AVAudioSession.sharedInstance()
        // .record rather than .playAndRecord: we never play anything, and
        // asking for playback would duck other audio for no reason.
        // .record with the audio background mode keeps capture alive while the
        // user is in another app typing -- which is the entire point.
        try session.setCategory(.record, mode: .measurement, options: [.duckOthers])
        try session.setActive(true, options: [])

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
        try engine.start()
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        converter = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
    }

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
