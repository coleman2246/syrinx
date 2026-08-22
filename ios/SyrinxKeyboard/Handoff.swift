import Foundation
import UIKit

/// How the app passes transcript text to the keyboard extension.
///
/// A keyboard extension cannot open the microphone — that is a security
/// boundary, not an oversight, and it has not moved in iOS 26. So the app
/// captures and transcribes, and the keyboard only inserts. This is the
/// channel between them.
///
/// Two channels, because the good one is not always available. Full Access
/// grants a shared container, but the App Groups *entitlement* needs a
/// provisioning profile that supports it, and free Apple ID provisioning
/// often will not grant one. Rather than guess at install time, both paths
/// are implemented and the right one is chosen at runtime by asking for the
/// container and seeing whether iOS hands one over.
enum Handoff {
    static let appGroup = "group.space.aragonite.SyrinxDemo"

    /// Non-nil only when the App Groups entitlement was actually granted.
    /// `UserDefaults(suiteName:)` is no use for this: it returns an object
    /// either way and silently fails to share.
    static var containerURL: URL? {
        FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: appGroup)
    }

    static var usingSharedContainer: Bool { containerURL != nil }

    private static var file: URL? {
        containerURL?.appendingPathComponent("pending.txt")
    }

    /// Called by the app when new text has been transcribed.
    static func publish(_ text: String) {
        guard !text.isEmpty else { return }
        if let f = file {
            // Append: the keyboard may not have collected the last piece yet,
            // and dropping words is worse than a short delay.
            let existing = (try? String(contentsOf: f, encoding: .utf8)) ?? ""
            try? (existing + text).write(to: f, atomically: true, encoding: .utf8)
        } else {
            // Clipboard fallback. Visible and it clobbers whatever was there,
            // which is why it is the second choice rather than the first.
            UIPasteboard.general.string = marker + text
        }
    }

    /// Called by the keyboard. Returns text not yet inserted, and clears it so
    /// the same words are never typed twice.
    static func take() -> String? {
        if let f = file {
            guard let s = try? String(contentsOf: f, encoding: .utf8), !s.isEmpty else {
                return nil
            }
            try? "".write(to: f, atomically: true, encoding: .utf8)
            return s
        }
        return takeFromPasteboard()
    }

    /// Clipboard fallback, for when the App Group entitlement is not granted.
    ///
    /// Keyed on `changeCount` rather than on the text: comparing strings would
    /// refuse to insert the same word twice in a row, which is a thing people
    /// say. The counter changes on every write even when the contents match.
    ///
    /// The extension's own defaults are used, because by definition there is
    /// no shared container in this path.
    private static func takeFromPasteboard() -> String? {
        let pb = UIPasteboard.general
        let seen = UserDefaults.standard.integer(forKey: "pbSeen")
        guard pb.changeCount != seen else { return nil }
        UserDefaults.standard.set(pb.changeCount, forKey: "pbSeen")
        // Only take what this app put there. Anything else is the user's own
        // clipboard and must not be typed into their document.
        guard let s = pb.string, s.hasPrefix(marker) else { return nil }
        return String(s.dropFirst(marker.count))
    }

    /// Tags clipboard writes as ours. Without it the keyboard would insert
    /// whatever the user last copied, which would be alarming.
    private static let marker = "\u{200B}"  // zero-width space, invisible if pasted

    /// Whether the app should be capturing. The keyboard sets this; the app
    /// watches it, so the mic button can live on the keyboard even though the
    /// microphone cannot.
    static var wantsCapture: Bool {
        get { defaults?.bool(forKey: "wantsCapture") ?? false }
        set { defaults?.set(newValue, forKey: "wantsCapture") }
    }

    private static var defaults: UserDefaults? {
        usingSharedContainer ? UserDefaults(suiteName: appGroup) : nil
    }
}
