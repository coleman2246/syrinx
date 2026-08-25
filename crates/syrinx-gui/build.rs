//! Puts the Windows icon into `syrinx-gui.exe` itself.
//!
//! The icon `main.rs` sets on the viewport only exists once the process is
//! running. Explorer, a shortcut, and a taskbar pin made before the first
//! launch all read the executable's own resource table instead, so an `.ico`
//! has to be linked into the binary to give the file an icon of its own.
//!
//! Off Windows this does nothing at all. `winresource` is a build-dependency
//! only under `cfg(windows)`, and the body below carries the same gate, so a
//! Linux build neither needs the crate nor looks for the `.ico`.
//!
//! Both of those gates read the *host*, not the target: cargo resolves a
//! `[target.'cfg(...)'.build-dependencies]` entry against the machine the
//! build script will run on, and `#[cfg]` in a build script means the same
//! thing. They therefore agree by construction, which is the point -- a gate
//! that disagreed with the manifest would be a missing-crate build failure.
//! The cost is that cross-compiling to Windows from elsewhere silently skips
//! the icon; the Windows build documented in the README is a native one.

fn main() {
    // Only the Windows branch reads it, but declaring it unconditionally keeps
    // the rule true on the platform that has one: re-render the icon and the
    // next build picks it up.
    println!("cargo:rerun-if-changed=assets/icon.ico");

    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/icon.ico");

        // A warning rather than a panic. The resource compiler is part of the
        // Windows SDK and can be absent from an otherwise working MSVC
        // toolchain; an executable that ends up wearing the default icon is
        // worth saying out loud, but it is not worth refusing to build.
        if let Err(e) = res.compile() {
            println!("cargo:warning=syrinx-gui.exe has no embedded icon: {e}");
        }
    }
}
