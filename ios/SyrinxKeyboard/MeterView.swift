import UIKit

/// The desktop overlay's meter, on a keyboard.
///
/// Dictating into another app gives no feedback at all: text appears wherever
/// the cursor is, and when nothing appears there is no way to tell a dead
/// microphone from a quiet room from a server that stopped answering.
///
/// Two signals, because one is not enough. The bars say sound is arriving.
/// The caption says words are being recognised, which a lively meter does not
/// prove -- a meter can bounce happily while the transcript stays empty.
final class MeterView: UIView {
    /// Smoothed bar heights, so the display eases rather than flickers.
    ///
    /// Starts at a full row of zeroes rather than empty: a meter that draws
    /// nothing until the first sample arrives looks like a missing feature
    /// rather than a quiet one.
    private var shown = [Float](repeating: 0, count: 10)
    /// Where the bars are heading. Set by the network, approached by the
    /// display link.
    private var target = [Float](repeating: 0, count: 10)
    private var caption = ""

    /// Redraws are driven by the screen, not by the socket.
    ///
    /// Levels arrive about twenty times a second, and easing once per arrival
    /// means the animation itself runs at twenty frames -- visibly stepped,
    /// however good the data is. Separating the two lets the bars move every
    /// frame between samples, which is what the desktop overlay does by
    /// repainting on a timer rather than on new audio.
    private var link: CADisplayLink?

    /// How quickly a falling bar drops, as a fraction of the gap per frame.
    ///
    /// Rising is instant so a syllable registers immediately; falling is eased,
    /// because bars that snap to zero between words read as dropouts rather
    /// than as speech. Larger than the overlay's 0.28 because this updates at
    /// 10 Hz rather than 30, and the same fraction per frame would crawl.
    /// Per frame at 60 Hz, so slower per step than the overlay's 0.28 at 30.
    private let fall: Float = 0.15
    /// Rising is quick rather than instant. Instant is right when the redraw
    /// is tied to the data; here it would step at the sample rate and undo
    /// the point of interpolating at all. Five frames to full is 80 ms.
    private let rise: Float = 0.5

    // The GUI's palette, so the phone and the desktop are recognisably one
    // application rather than two things that both draw bars.
    private let recording = UIColor(red: 0xE0/255, green: 0x4A/255, blue: 0x63/255, alpha: 1)
    private let warning   = UIColor(red: 0xF7/255, green: 0x9D/255, blue: 0x3C/255, alpha: 1)
    private let success   = UIColor(red: 0x6B/255, green: 0xB7/255, blue: 0x00/255, alpha: 1)

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .clear
        isUserInteractionEnabled = false
    }

    required init?(coder: NSCoder) { nil }

    /// Run the display link only while on screen. A keyboard is dismissed far
    /// more often than it is used, and a timer ticking sixty times a second
    /// behind a hidden view is pure drain.
    override func didMoveToWindow() {
        super.didMoveToWindow()
        if window == nil {
            link?.invalidate()
            link = nil
        } else if link == nil {
            let l = CADisplayLink(target: self, selector: #selector(step))
            // .common, or the bars freeze the moment anything is being
            // touched -- which on a keyboard is most of the time.
            l.add(to: .main, forMode: .common)
            link = l
        }
    }

    func show(levels: [Float], caption: String) {
        guard !levels.isEmpty else { return }
        if target.count != levels.count {
            target = Array(repeating: 0, count: levels.count)
            shown = Array(repeating: 0, count: levels.count)
        }
        for i in target.indices { target[i] = min(max(levels[i], 0), 1) }
        if caption != self.caption {
            self.caption = caption
            setNeedsDisplay()
        }
    }

    @objc private func step() {
        var moved = false
        for i in shown.indices {
            let t = target[i]
            let next = shown[i] + (t - shown[i]) * (t >= shown[i] ? rise : fall)
            if abs(next - shown[i]) > 0.001 {
                shown[i] = next
                moved = true
            } else if shown[i] != t, abs(t - shown[i]) < 0.002 {
                // Settle exactly, so a bar does not creep for ever towards a
                // value it never reaches.
                shown[i] = t
                moved = true
            }
        }
        if moved { setNeedsDisplay() }
    }

    /// Drop every bar to nothing. A meter frozen at its last frame after the
    /// microphone closes reads as though it were still listening.
    func clear() {
        target = [Float](repeating: 0, count: max(target.count, 10))
        caption = ""
        setNeedsDisplay()
    }

    override func draw(_ rect: CGRect) {
        guard let ctx = UIGraphicsGetCurrentContext() else { return }

        let panel = bounds.insetBy(dx: 8, dy: 2)
        UIColor.tertiarySystemFill.setFill()
        UIBezierPath(roundedRect: panel, cornerRadius: 8).fill()

        let captionHeight: CGFloat = 16
        let pad: CGFloat = 6
        let bars = CGRect(x: panel.minX + pad, y: panel.minY + pad,
                          width: panel.width - pad * 2,
                          height: panel.height - pad * 2 - captionHeight)

        if !shown.isEmpty && bars.height > 0 {
            let gap: CGFloat = 3
            let n = CGFloat(shown.count)
            let w = (bars.width - gap * (n - 1)) / n
            for (i, v) in shown.enumerated() {
                // A visible floor, so an idle meter reads as "running, quiet"
                // rather than "not running at all".
                let h = max(bars.height * CGFloat(v), 2)
                let bar = CGRect(x: bars.minX + CGFloat(i) * (w + gap),
                                 y: bars.maxY - h, width: w, height: h)
                let colour = v > 0.9 ? recording : (v > 0.7 ? warning : success)
                colour.setFill()
                UIBezierPath(roundedRect: bar, cornerRadius: 1.5).fill()
            }
        }

        let text = caption.isEmpty ? "listening…" : caption
        (text as NSString).draw(
            in: CGRect(x: bars.minX, y: bars.maxY + 1, width: bars.width, height: captionHeight),
            withAttributes: [
                .font: UIFont.systemFont(ofSize: 12),
                .foregroundColor: caption.isEmpty ? UIColor.tertiaryLabel : UIColor.label,
            ])
        ctx.flush()
    }
}
