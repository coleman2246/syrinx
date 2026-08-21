//! Shared-token authentication.
//!
//! The service is LAN-only behind a static bearer token. That is proportionate
//! for a trusted home network; it stops casual and accidental access without
//! the cert-distribution friction TLS would add on a Windows laptop.

/// Check an `Authorization` header against the configured token.
///
/// The comparison is constant-time over equal-length inputs. The token is a LAN
/// shared secret rather than a password hash, but timing-safe comparison is
/// nearly free here, so there is no reason to leak.
pub fn check_bearer(header: Option<&str>, expected: &str) -> bool {
    // An unset token must never mean "allow everyone". Failing closed here is
    // what stops a missing config value silently opening the service up.
    if expected.is_empty() {
        return false;
    }
    let Some(h) = header else { return false };
    let Some(token) = h.strip_prefix("Bearer ") else {
        return false;
    };
    let (a, b) = (token.as_bytes(), expected.as_bytes());
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correct_token_is_accepted() {
        assert!(check_bearer(Some("Bearer s3cret"), "s3cret"));
    }

    #[test]
    fn wrong_or_missing_token_is_rejected() {
        assert!(!check_bearer(Some("Bearer nope"), "s3cret"));
        assert!(!check_bearer(None, "s3cret"));
        assert!(!check_bearer(Some("s3cret"), "s3cret")); // missing scheme
    }

    #[test]
    fn empty_configured_token_rejects_everything() {
        // Refuse to run wide open by accident.
        assert!(!check_bearer(Some("Bearer "), ""));
        assert!(!check_bearer(None, ""));
    }

    #[test]
    fn token_of_different_length_is_rejected() {
        assert!(!check_bearer(Some("Bearer s3cretlonger"), "s3cret"));
        assert!(!check_bearer(Some("Bearer s3c"), "s3cret"));
    }

    #[test]
    fn scheme_is_case_sensitive_and_requires_the_space() {
        assert!(!check_bearer(Some("bearer s3cret"), "s3cret"));
        assert!(!check_bearer(Some("Bearers3cret"), "s3cret"));
    }
}
