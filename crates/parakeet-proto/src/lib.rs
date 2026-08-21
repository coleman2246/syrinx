//! Wire protocol for parakeet-stt. One definition, shared by server and clients.
//!
//! This crate deliberately depends on nothing but `serde`, so any client -- a
//! headless typer, a GUI, a future mobile app -- can compile it cheaply. Keeping
//! the protocol in one place is what stops the server and its clients drifting:
//! a breaking change fails the build rather than failing at runtime on the far
//! side of a network.
