import SwiftUI

struct ContentView: View {
    @StateObject private var dictation = Dictation()
    @State private var showSettings = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                HStack {
                    Circle()
                        .fill(dictation.running ? .red : .secondary)
                        .frame(width: 10, height: 10)
                    Text(dictation.status).foregroundStyle(.secondary)
                    Spacer()
                }

                ScrollView {
                    Text(dictation.transcript.isEmpty
                         ? "Press Start and speak."
                         : dictation.transcript)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .foregroundStyle(dictation.transcript.isEmpty ? .secondary : .primary)
                        .textSelection(.enabled)
                }
                .frame(maxHeight: .infinity)

                if let e = dictation.error {
                    Text(e).font(.footnote).foregroundStyle(.red)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }

                HStack {
                    Button(dictation.running ? "Stop" : "Start") { dictation.toggle() }
                        .buttonStyle(.borderedProminent)
                    Button("Copy") { UIPasteboard.general.string = dictation.transcript }
                        .disabled(dictation.transcript.isEmpty)
                    Button("Clear") { dictation.clear() }
                        .disabled(dictation.transcript.isEmpty)
                    Spacer()
                }
            }
            .padding()
            .navigationTitle("Syrinx")
            .toolbar {
                Button { showSettings = true } label: { Image(systemName: "gearshape") }
            }
            .sheet(isPresented: $showSettings) {
                SettingsView(dictation: dictation)
            }
        }
    }
}

struct SettingsView: View {
    @ObservedObject var dictation: Dictation
    @Environment(\.dismiss) private var dismiss
    @State private var diagnostics = SettingsView.report()

    /// Both halves of what decides whether dictation can start: the channel
    /// the keyboard uses, and the audio session it depends on.
    static func report() -> String {
        LocalLink.shared.diagnostics
            + "\n"
            + AudioSession.diagnostics
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Server") {
                    TextField("wss://host/v1/stream", text: $dictation.serverURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    SecureField("token", text: $dictation.token)
                }
                Section("Keyboard") {
                    Text("The keyboard reaches this app over \(Handoff.channelDescription). Full Access must be on, or the extension has no network at all.")
                        .font(.footnote)
                    // Live state rather than a claim about it. "requests
                    // served" is the one that matters: zero means the keyboard
                    // has never once reached this process.
                    Text(diagnostics)
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                    Text("granted group: \(Handoff.appGroup ?? "none")")
                        .font(.caption).foregroundStyle(.secondary)
                    Toggle("Hold the microphone", isOn: Binding(
                        get: { dictation.holdMicrophone },
                        set: { dictation.setHoldMicrophone($0) }))
                    Text("Keeps the microphone open so the keyboard can start dictation without opening this app. iOS will not let a background app claim a microphone it did not already have, so the alternative is opening Syrinx each time. The indicator stays lit because the microphone really is open; nothing is sent anywhere until you start dictating.")
                        .font(.caption).foregroundStyle(.secondary)
                    Button("Refresh") { diagnostics = Self.report() }
                }
                Section {
                    // The address is used exactly as written, same as the
                    // desktop clients — no scheme or port is inferred.
                    Text("The address is used exactly as written, including the scheme and path.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Settings")
            .toolbar { Button("Done") { dismiss() } }
        }
    }
}
