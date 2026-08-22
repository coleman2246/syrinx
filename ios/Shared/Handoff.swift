import Foundation

/// How the app passes transcript text to the keyboard extension.
///
/// A keyboard extension cannot open the microphone. That is a security
/// boundary, not an oversight, and it has not moved in iOS 26. So the app
/// captures and transcribes, and the keyboard only inserts.
///
/// One channel, deliberately. There used to be two -- a shared container when
/// the App Groups entitlement was granted, loopback otherwise -- and each side
/// chose for itself. A sideloader that granted the entitlement to the app but
/// not to the extension left them picking different channels and talking past
/// each other, with both sides reporting themselves healthy.
///
/// The container earned that risk only if it worked where loopback does not,
/// and it does not: a keyboard needs Full Access to reach a shared container
/// just as it does to open a socket. So loopback is the only channel and the
/// two ends cannot disagree about something they no longer decide. The App
/// Groups entitlement went with it -- see docs/ios.md.
enum Handoff {
    /// Called by the app when new text has been transcribed.
    static func publish(_ text: String) {
        guard !text.isEmpty else { return }
        LocalLink.shared.publish(text)
    }

    /// Called by the keyboard. Yields text not yet inserted and clears it, so
    /// the same words are never typed twice.
    ///
    /// `nil` means the app could not be reached, which the keyboard reports
    /// differently from "reached, nothing new": the first needs the user to
    /// open the app, the second needs them to keep talking.
    static func take(_ completion: @escaping (Frame?) -> Void) {
        LocalLinkClient.send("TAKE") { reply in
            guard let reply else {
                completion(nil)
                return
            }
            completion(Frame(
                text: reply.first ?? "",
                levels: (reply.count > 1 ? reply[1] : "")
                    .split(separator: ",").compactMap { Float($0) },
                capturing: reply.count > 2 && reply[2] == "1"))
        }
    }

    /// One poll's worth of state: what to type, and what the microphone is
    /// hearing. Together because the keyboard wants both at the same instant.
    struct Frame {
        let text: String
        let levels: [Float]
        /// What the app is actually doing, which is not always what the
        /// keyboard last asked for -- dictation can be started from the app.
        let capturing: Bool
    }

    /// Ask the app to start or stop capturing. Reports whether it was heard.
    static func setWantsCapture(_ on: Bool, then completion: @escaping (Bool) -> Void) {
        LocalLinkClient.send(on ? "START" : "STOP") { completion($0 != nil) }
    }

    /// Whether the app is capturing, or nil if it cannot be reached.
    static func captureState(_ completion: @escaping (Bool?) -> Void) {
        LocalLinkClient.send("STATE") { completion($0.map { $0.first == "1" }) }
    }

    static var channelDescription: String {
        "loopback 127.0.0.1:\(LocalLinkProtocol.port.rawValue)"
    }
}
