import Foundation
import UIKit

/// How the app passes transcript text to the keyboard extension.
///
/// A keyboard extension cannot open the microphone. That is a security
/// boundary, not an oversight, and it has not moved in iOS 26. So the app
/// captures and transcribes, and the keyboard only inserts. This is the
/// channel between them.
///
/// Two channels, because the good one is not always available. A shared
/// container needs the App Groups entitlement, and a sideloaded build gets
/// whatever the signer chose to grant. Rather than guess at install time,
/// both paths are implemented and the right one is chosen at runtime by
/// asking for the container and seeing whether iOS hands one over.
enum Handoff {
    /// What the entitlement files ask for. What is actually granted may
    /// differ, which is why nothing reads this directly.
    static let declaredGroup = "group.space.aragonite.SyrinxDemo"

    /// The group iOS actually granted, resolved once at first use.
    ///
    /// Not a constant, because a re-signer is free to rename it. AltStore and
    /// friends register groups under their own team and rewrite the
    /// entitlement to match, so the ID in the installed app is often
    /// `group.<TEAMID>.space.aragonite.SyrinxDemo` rather than what was asked
    /// for. Hardcoding the declared name means asking for a container that
    /// was never granted and silently falling back to the clipboard.
    static let appGroup: String? = {
        // The declared name first: if it survived signing, prefer it.
        for id in [declaredGroup] + provisionedGroups() {
            if FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: id) != nil {
                return id
            }
        }
        return nil
    }()

    /// The groups named in this bundle's provisioning profile.
    ///
    /// The profile is a CMS signature wrapping a plist. Verifying the
    /// signature is the system's job, and this only needs to read a value out
    /// of a file already inside our own bundle, so the plist is located by
    /// its delimiters rather than by decoding the container.
    ///
    /// `Bundle.main` is the extension's own bundle when this runs in the
    /// keyboard, which is what is wanted: each target carries its own profile.
    private static func provisionedGroups() -> [String] {
        guard let url = Bundle.main.url(forResource: "embedded", withExtension: "mobileprovision"),
              let data = try? Data(contentsOf: url),
              let start = data.range(of: Data("<?xml".utf8)),
              let end = data.range(of: Data("</plist>".utf8), options: .backwards)
        else { return [] }
        let plist = try? PropertyListSerialization.propertyList(
            from: data[start.lowerBound..<end.upperBound], format: nil)
        guard let root = plist as? [String: Any],
              let entitlements = root["Entitlements"] as? [String: Any],
              let groups = entitlements["com.apple.security.application-groups"] as? [String]
        else { return [] }
        return groups
    }

    /// Non-nil only when the App Groups entitlement was actually granted.
    /// `UserDefaults(suiteName:)` is no use for this: it returns an object
    /// either way and silently fails to share.
    static var containerURL: URL? {
        appGroup.flatMap {
            FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: $0)
        }
    }

    static var usingSharedContainer: Bool { containerURL != nil }

    /// One line naming the live channel, for the app and the keyboard to show.
    ///
    /// Which channel is in use decides what the user has to do -- the shared
    /// container lets the keyboard start capture, the clipboard does not -- so
    /// it should never have to be inferred from behaviour.
    static var channelDescription: String {
        if let g = appGroup { return "shared container (\(g))" }
        return "clipboard (no App Group granted)"
    }

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
        appGroup.flatMap { UserDefaults(suiteName: $0) }
    }
}
