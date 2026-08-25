# Icon

A syrinx is the vocal organ a bird sings with, so the mark is a bird: a white
silhouette on a rounded square in the brand colour, `#5B5FC7`.

| File | What it is |
|---|---|
| `icon.svg` | The source of truth. Hand-authored, and the only file to edit. |
| `icon-256.png` | 256x256 RGBA. `include_bytes!`d into `syrinx-gui` as the window, taskbar and alt-tab icon. |
| `icon.ico` | 16/24/32/48/64/128, linked into `syrinx-gui.exe` by `build.rs` so the file has an icon in Explorer. |

Two rasters because the two jobs take different formats: egui wants decoded
RGBA at one size, and the Windows resource table wants an `.ico` carrying every
size at once, so that Explorer, the taskbar and alt-tab each pick their own
rather than rescaling one.

## Regenerating

Both rasters come from the SVG. After editing it:

```sh
rsvg-convert -w 256 -h 256 icon.svg -o icon-256.png
for s in 16 24 32 48 64 128; do rsvg-convert -w $s -h $s icon.svg -o /tmp/ico_$s.png; done
magick /tmp/ico_16.png /tmp/ico_24.png /tmp/ico_32.png /tmp/ico_48.png /tmp/ico_64.png /tmp/ico_128.png icon.ico
```

Then commit the results. The rasters are checked in deliberately: rendering
them at build time would put `rsvg-convert` and ImageMagick between a fresh
clone and a working `cargo build`, on every platform, to produce two files that
change about as often as the name does. `cargo test -p syrinx-gui` checks that
`icon-256.png` still decodes at the size the window expects, which is worth
running after a re-render -- the GUI logs a failure there and carries on
without an icon rather than refusing to start, so nothing else would say.
