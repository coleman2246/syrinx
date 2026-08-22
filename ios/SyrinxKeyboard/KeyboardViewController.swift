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
    private var poll: Timer?
    private var capturing = false

    override func viewDidLoad() {
        super.viewDidLoad()
        buildUI()
        capturing = Handoff.wantsCapture
        render()
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        // 5 Hz: text arrives in fragments and the keyboard is on screen while
        // someone is speaking, so this needs to feel immediate without
        // spinning.
        poll = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
            self?.drain()
        }
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        poll?.invalidate()
        poll = nil
    }

    private func drain() {
        guard let text = Handoff.take(), !text.isEmpty else { return }
        textDocumentProxy.insertText(text)
    }

    @objc private func toggleMic() {
        capturing.toggle()
        Handoff.wantsCapture = capturing
        render()

        if !Handoff.usingSharedContainer {
            // Without a shared container the app cannot see the flag, so the
            // user has to start capture in the app themselves. Saying so beats
            // a button that appears to do nothing.
            statusLabel.text = "Open Syrinx to start dictating"
        }
    }

    private func render() {
        let name = capturing ? "mic.fill" : "mic"
        micButton.setImage(UIImage(systemName: name), for: .normal)
        micButton.tintColor = capturing ? .systemRed : .label
        statusLabel.text = capturing ? "Listening — speak" : "Tap the mic to dictate"
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

        let stack = UIStackView(arrangedSubviews: [nextKeyboard, micButton, statusLabel])
        stack.axis = .horizontal
        stack.spacing = 16
        stack.alignment = .center
        stack.isLayoutMarginsRelativeArrangement = true
        stack.directionalLayoutMargins = .init(top: 8, leading: 16, bottom: 8, trailing: 16)
        stack.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(stack)

        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            stack.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            stack.topAnchor.constraint(equalTo: view.topAnchor),
            // A keyboard has no intrinsic height; without this it collapses.
            view.heightAnchor.constraint(equalToConstant: 88),
        ])
    }
}
