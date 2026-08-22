#!/usr/bin/env bash
# Build an unsigned .ipa for sideloading.
#
# Unsigned on purpose: AltStore and SideStore re-sign with your own Apple ID
# on the device, so no certificate is needed here and no credentials are ever
# stored by the build.
set -euo pipefail
cd "$(dirname "$0")"

./generate.sh

BUILD="$PWD/build/Release-iphoneos"
xcodebuild -project SyrinxDemo.xcodeproj -target SyrinxDemo \
    -sdk iphoneos -configuration Release -arch arm64 \
    CONFIGURATION_BUILD_DIR="$BUILD" \
    CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO

# Ad-hoc sign so the entitlements are actually embedded in the binaries.
#
# With CODE_SIGNING_ALLOWED=NO, Xcode never writes the entitlements blob at
# all -- entitlements are applied at signing time. A sideloader re-signing the
# app therefore sees no request for an App Group and grants none, and the
# keyboard silently falls back to the clipboard. Signing ad-hoc here embeds
# the request; the sideloader replaces the signature but keeps it.
#
# Extension first: signing the app seals its contents, so anything signed
# afterwards invalidates it.
codesign --force --sign - \
    --entitlements SyrinxKeyboard/SyrinxKeyboard.entitlements \
    "$BUILD/SyrinxDemo.app/PlugIns/SyrinxKeyboard.appex"
codesign --force --sign - \
    --entitlements SyrinxDemo/SyrinxDemo.entitlements \
    "$BUILD/SyrinxDemo.app"

# An .ipa is a zip with the bundle under Payload/. Nothing more.
rm -rf "$PWD/build/Payload" "$PWD/build/SyrinxDemo.ipa"
mkdir -p "$PWD/build/Payload"
cp -R "$BUILD/SyrinxDemo.app" "$PWD/build/Payload/"
(cd "$PWD/build" && zip -qry SyrinxDemo.ipa Payload)
rm -rf "$PWD/build/Payload"

echo
echo "$PWD/build/SyrinxDemo.ipa"
ls -lh "$PWD/build/SyrinxDemo.ipa" | awk '{print "  " $5}'
