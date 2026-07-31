# macOS Build

Use this on the Mac where you want to test Martinus.

## Prerequisites

- Node.js and npm
- Rust with `rustup` (`aarch64-apple-darwin` and/or `x86_64-apple-darwin` target)
- Xcode Command Line Tools (`xcode-select --install`)

## Commands

From `/desktop`:

```bash
npm install
npm run macos:build
```

For a lighter compile-only check without an app/dmg bundle:

```bash
npm run macos:build:nobundle
```

The script will:

- generate `src-tauri/icons/icon.icns` if it is missing
- detect the local architecture (Apple Silicon vs Intel) and ensure the
  matching Rust target is installed
- run the Tauri macOS build, producing a `.app` and `.dmg`

Expected output folder:

```text
src-tauri/target/<arch>-apple-darwin/release/bundle
```

where `<arch>` is `aarch64` on Apple Silicon or `x86_64` on Intel Macs.

## Optional Claude Code provider

Install and authenticate the Claude Code CLI before starting Martinus. The
desktop settings drawer will show a third provider card named **Claude Code**.
Select it and refresh runtime detection; a successful CLI version check exposes
the logical model `claude-code`. Ollama remains the default provider.
