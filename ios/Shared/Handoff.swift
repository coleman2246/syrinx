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
        return "loopback 127.0.0.1:\(LocalLinkProtocol.port.rawValue) (no App Group granted)"
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
            LocalLink.shared.publish(text)
        }
    }

    /// Called by the keyboard. Yields text not yet inserted, and clears it so
    /// the same words are never typed twice.
    ///
    /// Asynchronous because one of the two channels is a socket. `nil` means
    /// the app could not be reached at all, which the keyboard reports
    /// differently from "reached, nothing new": the first needs the user to
    /// open the app, the second needs them to keep talking.
    static func take(_ completion: @escaping (String?) -> Void) {
        if let f = file {
            guard let s = try? String(contentsOf: f, encoding: .utf8), !s.isEmpty else {
                completion("")
                return
            }
            try? "".write(to: f, atomically: true, encoding: .utf8)
            completion(s)
            return
        }
        LocalLinkClient.send("TAKE", then: completion)
    }

    /// Ask the app to start or stop capturing.
    ///
    /// The keyboard cannot open the microphone, so this is how its mic button
    /// does anything at all. Reports whether the app was there to hear it.
    static func setWantsCapture(_ on: Bool, then completion: @escaping (Bool) -> Void) {
        if usingSharedContainer {
            wantsCapture = on
            completion(true)
            return
        }
        LocalLinkClient.send(on ? "START" : "STOP") { completion($0 != nil) }
    }

    /// Whether the app is capturing right now, or nil if it cannot be reached.
    static func captureState(_ completion: @escaping (Bool?) -> Void) {
        if usingSharedContainer {
            completion(wantsCapture)
            return
        }
        LocalLinkClient.send("STATE") { completion($0.map { $0 == "1" }) }
    }

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
