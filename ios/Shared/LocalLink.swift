import Foundation
import Network

/// A loopback channel between the app and its keyboard extension.
///
/// The shared container is the natural way for the two to talk, but it needs
/// the App Groups entitlement, and a sideloaded build gets whatever its signer
/// chose to grant. The clipboard was the fallback and it does not work: iOS
/// prompts before one process reads what another wrote, and rewriting the
/// user's clipboard several times a second is hostile even when it succeeds.
///
/// Both processes can open sockets, though, and 127.0.0.1 never leaves the
/// phone. The app listens, the keyboard connects. No entitlement is involved,
/// only Full Access, which the keyboard already needs to reach a network at
/// all. It also gives the keyboard a control channel, so its mic button can
/// start capture in the app rather than only reporting that it cannot.
///
/// One request per connection, one line each way. The reply is always a JSON
/// array holding one string, because a bare JSON string is a fragment and not
/// every decoder accepts one at the top level.
///
///     TAKE <secret>   -> ["text", "0.1,0.4,..." spectrum, "1" if capturing]
///     START <secret>  -> ["1"]
///     STOP <secret>   -> ["0"]
///     STATE <secret>  -> ["1"] while capturing, ["0"] otherwise
///
/// The meter rides along with TAKE rather than having a verb of its own: the
/// keyboard wants both at the same moment and at the same rate, and a second
/// round trip per frame would double the traffic to say nothing new.
///
/// The secret only stops unrelated software on the device from stumbling into
/// the port. It is a constant in a public repository and protects nothing from
/// anyone who looks: any app on the phone can reach a loopback listener. The
/// exposure is narrower than the clipboard it replaces, which every app can
/// read, but it is not nothing, which is why the listener only runs when the
/// shared container is unavailable.
enum LocalLinkProtocol {
    static let port = NWEndpoint.Port(rawValue: 47632)!
    static let secret = "syrinx-loopback-1"

    static func encode(_ items: String...) -> Data {
        (try? JSONEncoder().encode(items)) ?? Data("[\"\"]".utf8)
    }

    static func decode(_ d: Data) -> [String]? {
        try? JSONDecoder().decode([String].self, from: d)
    }
}

/// The app's half: holds text until the keyboard collects it.
final class LocalLink {
    static let shared = LocalLink()

    private let queue = DispatchQueue(label: "space.aragonite.syrinx.link")
    private let lock = NSLock()
    private var listener: NWListener?
    private var pending = ""
    private var capturing = false
    private var levels: [Float] = []

    /// Whether the listener came up, and what went wrong if not.
    ///
    /// A listener that fails to bind is silent -- the keyboard just sees an
    /// unreachable app, which looks the same as an app that is not running.
    /// Reported so the two can be told apart without a debugger.
    private(set) var listenerState = "not started"
    /// Requests served, so the diagnostics can distinguish "the keyboard never
    /// reached us" from "it reached us and there was nothing to send".
    private(set) var served = 0

    /// Set by the app. Called when the keyboard asks capture to start or stop.
    var onCapture: ((Bool) -> Void)?

    func startServing() {
        guard listener == nil else { return }
        let params = NWParameters.tcp
        // Loopback only. Binding every interface would put the transcript on
        // the local network, which is a different and much worse thing.
        params.requiredLocalEndpoint = .hostPort(host: "127.0.0.1",
                                                 port: LocalLinkProtocol.port)
        params.allowLocalEndpointReuse = true
        let l: NWListener
        do {
            l = try NWListener(using: params)
        } catch {
            listenerState = "bind failed: \(error)"
            return
        }
        l.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready:          self?.listenerState = "listening"
            case .failed(let e):  self?.listenerState = "failed: \(e)"
            case .cancelled:      self?.listenerState = "cancelled"
            case .waiting(let e): self?.listenerState = "waiting: \(e)"
            default:              break
            }
        }
        l.newConnectionHandler = { [weak self] c in self?.serve(c) }
        l.start(queue: queue)
        listener = l
    }

    func stopServing() {
        listener?.cancel()
        listener = nil
    }

    /// Called by the app as text arrives.
    func publish(_ text: String) {
        lock.lock(); pending += text; lock.unlock()
    }

    /// Called by the app so STATE reports the truth.
    func setCapturing(_ on: Bool) {
        lock.lock()
        capturing = on
        // A meter left at its last value would keep bouncing after the
        // microphone closed, which reads as "still listening".
        if !on { levels = [] }
        lock.unlock()
    }

    /// Called by the app as the spectrum changes.
    func setLevels(_ l: [Float]) {
        lock.lock(); levels = l; lock.unlock()
    }

    private func serve(_ c: NWConnection) {
        c.start(queue: queue)
        c.receive(minimumIncompleteLength: 1, maximumLength: 256) { [weak self] data, _, _, _ in
            guard let self, let data else { c.cancel(); return }
            let line = String(decoding: data, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            self.served += 1
            let reply = self.handle(line)
            c.send(content: reply, completion: .contentProcessed { _ in c.cancel() })
        }
    }

    /// A line-by-line report of everything that decides whether this works.
    var diagnostics: String {
        lock.lock()
        let queued = pending.count
        let on = capturing
        lock.unlock()
        return """
        listener: \(listenerState)
        port: \(LocalLinkProtocol.port.rawValue)
        requests served: \(served)
        queued chars: \(queued)
        capturing: \(on)
        """
    }

    private func handle(_ line: String) -> Data {
        let parts = line.split(separator: " ", maxSplits: 1).map(String.init)
        guard parts.count == 2, parts[1] == LocalLinkProtocol.secret else {
            return LocalLinkProtocol.encode("")
        }
        switch parts[0] {
        case "TAKE":
            lock.lock()
            let text = pending
            pending = ""
            let meter = levels.map { String(format: "%.3f", $0) }.joined(separator: ",")
            let on = capturing
            lock.unlock()
            // Whether capture is running belongs in every frame, not only in
            // the reply to STATE. The keyboard cannot know that dictation was
            // started from the app otherwise, and would sit with a dead meter
            // and an unlit button while text arrived.
            return LocalLinkProtocol.encode(text, meter, on ? "1" : "0")
        case "START", "STOP":
            let wanted = parts[0] == "START"
            DispatchQueue.main.async { self.onCapture?(wanted) }
            return LocalLinkProtocol.encode(wanted ? "1" : "0")
        default:
            lock.lock(); let on = capturing; lock.unlock()
            return LocalLinkProtocol.encode(on ? "1" : "0")
        }
    }
}

/// The keyboard's half.
enum LocalLinkClient {
    /// A dead app must fail fast: this runs on a repeating poll while someone
    /// is speaking, and a request that outlives the next tick would pile up.
    private static let timeout: DispatchTimeInterval = .milliseconds(700)

    /// `nil` means the app is not listening, which is a different state from
    /// "listening, nothing to say" and the keyboard reports it differently.
    static func send(_ verb: String, then completion: @escaping ([String]?) -> Void) {
        let c = NWConnection(host: "127.0.0.1", port: LocalLinkProtocol.port, using: .tcp)
        let queue = DispatchQueue(label: "space.aragonite.syrinx.link.client")
        // Guards against calling back twice: a timeout racing a reply, or a
        // connection that fails after it has already answered.
        let done = Atomic(false)
        func finish(_ value: [String]?) {
            guard done.swap(true) == false else { return }
            c.cancel()
            DispatchQueue.main.async { completion(value) }
        }

        c.stateUpdateHandler = { state in
            switch state {
            case .ready:
                c.send(content: Data("\(verb) \(LocalLinkProtocol.secret)\n".utf8),
                       completion: .contentProcessed { _ in })
                c.receive(minimumIncompleteLength: 1, maximumLength: 65536) { data, _, _, _ in
                    finish(data.flatMap(LocalLinkProtocol.decode))
                }
            // .waiting is what a refused connection looks like: nobody is
            // listening, and Network.framework would otherwise retry forever.
            case .failed, .cancelled, .waiting:
                finish(nil)
            default:
                break
            }
        }
        c.start(queue: queue)
        queue.asyncAfter(deadline: .now() + timeout) { finish(nil) }
    }
}

/// Minimal one-shot flag. `NSLock` around a `Bool` rather than a dependency.
final class Atomic {
    private let lock = NSLock()
    private var value: Bool
    init(_ v: Bool) { value = v }
    func swap(_ new: Bool) -> Bool {
        lock.lock(); defer { lock.unlock() }
        let old = value
        value = new
        return old
    }
}
