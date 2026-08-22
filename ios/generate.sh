#!/usr/bin/env bash
# Generate SyrinxDemo.xcodeproj from project.yml.
#
# The project is generated rather than committed: a pbxproj is thousands of
# lines of machine-managed UUIDs that cannot be reviewed and conflict on every
# branch.
set -euo pipefail
cd "$(dirname "$0")"

# XcodeGen fails on a missing configFile, and Local.xcconfig is gitignored, so
# a clean checkout needs an empty one before anything will generate.
[ -f Local.xcconfig ] || cp Local.xcconfig.example Local.xcconfig

XG="$(command -v xcodegen || echo "$HOME/xcodegen/xcodegen/bin/xcodegen")"
"$XG" generate
