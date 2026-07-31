# Windows Build

Use this on the Windows machine where you want to test Martinus.

## Prerequisites

- Node.js and npm
- Rust with `rustup`
- Visual Studio Build Tools 2022 with the C++ workload
- Microsoft Edge WebView2 Runtime

## Commands

From `/desktop`:

```powershell
npm install
npm run windows:build
```

For a lighter compile-only check without installer bundles:

```powershell
npm run windows:build:nobundle
```

The script will:

- generate `src-tauri/icons/icon.ico` if it is missing
- ensure the Windows Rust target exists
- run the Tauri Windows build

Expected output folder:

```text
src-tauri/target/x86_64-pc-windows-msvc/release/bundle
```

## Optional Claude Code provider

Install and authenticate the Claude Code CLI before starting Martinus. The
desktop settings drawer will show a third provider card named **Claude Code**.
The runtime detector searches `PATH` for `claude`, `claude.exe`,
`claude.cmd`, or `claude.bat`. Select the provider and refresh; a successful
CLI version check exposes the logical model `claude-code`. Ollama remains the
default provider.
