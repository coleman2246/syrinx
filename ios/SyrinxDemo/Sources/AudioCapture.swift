import AVFoundation

/// The microphone, opened once and left open.
///
/// iOS does not forbid a background app from recording. It forbids it from
/// *beginning* to: activating a session, setting a category, starting an
/// engine. Every failure the keyboard hit was one of those, and no amount of
/// choosing better categories changes it, because the refusal is about when
/// the call happens rather than what it asks for.
///
/// So nothing begins in the background. The session, the engine and the tap
/// are all brought up at launch, while the app is on screen and permitted to,
/// and then left running for the life of the process. Starting dictation
/// afterwards sets a closure; stopping it clears one. There is no audio call
/// left in that path for iOS to refuse.
///
/// A running engine is also its own keep-alive, so the silent-playback trick
/// that used to hold the app resident is gone. What remains is the honest
/// version of the same bargain: the microphone indicator stays lit for as long
/// as Syrinx is resident, because the microphone is genuinely open.
///
/// The hardware format is not negotiable -- the input node reports whatever
/// the device is doing, typically 48 kHz stereo -- so a converter sits between
/// it and the 16 kHz mono the model expects. Getting this wrong does not fail
/// loudly; it transcribes gibberish, which is much harder to diagnose than a
/// crash.
final class AudioCapture {
    private let engine = AVAudioEngine()
    private var converter: AVAudioConverter?
    private let target: AVAudioFormat

    /// Where samples go right now, or nil when the microphone is open and
    /// nobody is listening. Swapping this is the whole of starting and
    /// stopping dictation.
    private let lock = NSLock()
    private var sink: ((UnsafePointer<Float>, Int) -> Void)?

    private(set) var isOpen = false

    init?() {
        guard let fmt = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: SyrinxCoreSession.sampleRate,
            channels: 1,
            interleaved: false
        ) else { return nil }
        self.target = fmt
    }

    /// Open the microphone. Must be called while the app is in the foreground.
    func open() throws {
        guard !isOpen else { return }
        try AudioSession.ensureActive()

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
            guard let self else { return }
            // Cheapest possible check first: most of the time the microphone
            // is open and nothing is being dictated, and that path should cost
            // nothing but a lock and a nil test.
            self.lock.lock()
            let sink = self.sink
            self.lock.unlock()
            guard let sink, let converter = self.converter else { return }

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
            sink(ch, Int(out.frameLength))
        }

        engine.prepare()
        do {
            try engine.start()
        } catch {
            // Distinct from the session failing: the session can be healthy
            // and the engine still refuse, and the two need different answers.
            teardown()
            throw NSError(domain: "Syrinx.audio", code: (error as NSError).code, userInfo: [
                NSLocalizedDescriptionKey:
                    "starting the audio engine: \(AudioSession.describe(error))"
            ])
        }
        isOpen = true
    }

    /// Release the microphone, which also clears the indicator.
    func close() {
        guard isOpen else { return }
        teardown()
        isOpen = false
    }

    private func teardown() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        engine.reset()
        converter = nil
    }

    /// Start or stop delivering. The only thing dictation does to audio.
    func deliver(to sink: ((UnsafePointer<Float>, Int) -> Void)?) {
        lock.lock()
        self.sink = sink
        lock.unlock()
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
