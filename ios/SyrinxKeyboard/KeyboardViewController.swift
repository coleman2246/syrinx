import UIKit

/// The dictation keyboard.
///
/// It inserts text; it does not capture audio. A keyboard extension has no
/// microphone access on any iOS version, so the container app records and this
/// only collects what the app has transcribed and types it at the cursor.
///
/// Insertion is append-only, which is not a simplification but a requirement:
/// `UITextDocumentProxy` edits whatever field has focus, and by the time a
/// revision arrived the user may have moved the cursor or typed something
/// themselves. Deleting characters would destroy their work. The same
/// constraint the desktop clients run under.
final class KeyboardViewController: UIInputViewController {
    private let micButton = UIButton(type: .system)
    private let statusLabel = UILabel()
    private let nextKeyboard = UIButton(type: .system)
    private let infoButton = UIButton(type: .system)
    private let meter = MeterView()
    /// The last few words inserted, shown under the bars.
    ///
    /// Taken from what was typed rather than fetched separately: the keyboard
    /// already knows, and a second source could disagree with the cursor.
    private var recent: [String] = []
    private var poll: Timer?
    private var tick = 0
    private var capturing = false
    /// Whether the last exchange reached the app. The app is only alive in the
    /// background while it holds an audio session, so "not running" is a
    /// normal state the user has to be told about rather than an error.
    private var reachable = false
    /// One request at a time. The poll fires faster than a dead app times out,
    /// and overlapping requests would queue up behind each other.
    private var inFlight = false

    override func viewDidLoad() {
        super.viewDidLoad()
        buildUI()
        render()
        Handoff.captureState { [weak self] state in
            self?.capturing = state ?? false
            self?.reachable = state != nil
            self?.render()
        }
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        // 5 Hz: text arrives in fragments and the keyboard is on screen while
        // someone is speaking, so this needs to feel immediate without
        // spinning.
        // 10 Hz, matching the rate the app recomputes the spectrum at, so the
        // bars move as smoothly as there is data to move them.
        tick = 0
        poll = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            self?.drain()
        }
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        poll?.invalidate()
        poll = nil
    }

    private func drain() {
        guard hasFullAccess, !inFlight else { return }
        tick += 1
        // Idle, there is nothing to draw and nothing to type, so polling ten
        // times a second is ten times a second of nothing. Twice a second is
        // enough to notice the app coming back.
        guard capturing || tick % 5 == 0 else { return }

        inFlight = true
        Handoff.take { [weak self] frame in
            guard let self else { return }
            self.inFlight = false
            let was = self.reachable
            self.reachable = frame != nil
            guard let frame else {
                self.meter.clear()
                if was { self.render() }
                return
            }
            if !frame.text.isEmpty {
                self.textDocumentProxy.insertText(frame.text)
                self.remember(frame.text)
            }
            if self.capturing {
                self.meter.show(levels: frame.levels, caption: self.recent.joined(separator: " "))
            }
            if was != self.reachable { self.render() }
        }
    }

    /// Keep the tail of what was typed, for the caption.
    private func remember(_ text: String) {
        recent.append(contentsOf: text.split(separator: " ").map(String.init))
        // Six words is what fits on one line at this size, and is enough to
        // recognise your own sentence.
        if recent.count > 6 { recent.removeFirst(recent.count - 6) }
    }

    @objc private func toggleMic() {
        capturing.toggle()
        render()
        if !capturing { meter.clear(); recent = [] }
        Handoff.setWantsCapture(capturing) { [weak self] delivered in
            guard let self else { return }
            self.reachable = delivered
            // The app was not there to hear it, so nothing is capturing
            // whatever the button now looks like.
            if !delivered { self.capturing = false }
            self.render()
        }
    }

    private func render() {
        let name = capturing ? "mic.fill" : "mic"
        micButton.setImage(UIImage(systemName: name), for: .normal)

        // Full Access is what gives an extension a network, and without one
        // there is no way to reach the app at all. Worth naming the exact
        // setting: it is four levels deep and easy to miss.
        guard hasFullAccess else {
            statusLabel.text = "Turn on Full Access: Settings › General › Keyboard › Keyboards › Syrinx"
            statusLabel.numberOfLines = 2
            micButton.isEnabled = false
            micButton.tintColor = .tertiaryLabel
            return
        }

        // The app is only resident while it holds an audio session, so before
        // the first dictation it has to be opened by hand. Saying so beats
        // letting the user tap a mic that cannot reach anything.
        guard reachable else {
            statusLabel.text = "Open Syrinx once to start — it keeps running in the background"
            statusLabel.numberOfLines = 2
            micButton.isEnabled = true
            micButton.tintColor = .tertiaryLabel
            return
        }

        micButton.isEnabled = true
        micButton.tintColor = capturing ? .systemRed : .label
        statusLabel.text = capturing ? "Listening — speak" : "Tap the mic to dictate"
    }

    /// Type a report of everything that decides whether this works.
    ///
    /// A sideloaded keyboard extension has no console anyone can get at, no
    /// debugger attached, and one line of its own UI to say anything in. What
    /// it does have is the ability to type into whatever field has focus, so
    /// the text field becomes the diagnostic channel. Tap it anywhere text can
    /// be entered and the answer is on screen, copyable.
    @objc private func typeDiagnostics() {
        let groups = Handoff.provisionedGroups()
        textDocumentProxy.insertText("""
        — syrinx keyboard —
        full access: \(hasFullAccess)
        bundle: \(Bundle.main.bundleIdentifier ?? "?")
        ios: \(UIDevice.current.systemVersion)
        granted group: \(Handoff.appGroup ?? "none")
        profile groups: \(groups.isEmpty ? "none" : groups.joined(separator: ", "))
        channel: \(Handoff.channelDescription)
        probing app…
        """)
        let began = Date()
        LocalLinkClient.send("STATE") { [weak self] reply in
            let ms = Int(Date().timeIntervalSince(began) * 1000)
            let outcome = reply.map { "reached, capturing=\($0 == "1")" } ?? "UNREACHABLE"
            self?.textDocumentProxy.insertText("\napp: \(outcome) (\(ms) ms)\n— end —\n")
        }
    }

    private func buildUI() {
        view.backgroundColor = .secondarySystemBackground

        micButton.addTarget(self, action: #selector(toggleMic), for: .touchUpInside)
        micButton.setPreferredSymbolConfiguration(
            .init(pointSize: 28, weight: .regular), forImageIn: .normal)

        statusLabel.font = .preferredFont(forTextStyle: .footnote)
        statusLabel.textColor = .secondaryLabel

        // Required: a custom keyboard must offer a way back to the others, or
        // the user is stuck in it.
        nextKeyboard.setImage(UIImage(systemName: "globe"), for: .normal)
        nextKeyboard.addTarget(self,
                               action: #selector(handleInputModeList(from:with:)),
                               for: .allTouchEvents)

        infoButton.setImage(UIImage(systemName: "info.circle"), for: .normal)
        infoButton.addTarget(self, action: #selector(typeDiagnostics), for: .touchUpInside)

        let controls = UIStackView(arrangedSubviews: [nextKeyboard, micButton, statusLabel, infoButton])
        controls.axis = .horizontal
        controls.spacing = 16
        controls.alignment = .center
        controls.isLayoutMarginsRelativeArrangement = true
        controls.directionalLayoutMargins = .init(top: 6, leading: 16, bottom: 0, trailing: 16)

        let stack = UIStackView(arrangedSubviews: [controls, meter])
        stack.axis = .vertical
        stack.spacing = 2
        stack.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(stack)

        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            stack.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            stack.topAnchor.constraint(equalTo: view.topAnchor),
            controls.heightAnchor.constraint(equalToConstant: 44),
            // A keyboard has no intrinsic height; without this it collapses.
            view.heightAnchor.constraint(equalToConstant: 132),
        ])
    }
}
