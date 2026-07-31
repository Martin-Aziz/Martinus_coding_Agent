use serde::Serialize;
use std::{
    collections::BTreeSet,
    env,
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};

const OLLAMA_TAGS_URL: &str = "http://127.0.0.1:11434/api/tags";
const LMSTUDIO_MODELS_URL: &str = "http://127.0.0.1:1234/v1/models";
const ORCHESTRATOR_HEALTH_URL: &str = "http://127.0.0.1:18080/healthz";
const AI_HEALTH_URL: &str = "http://127.0.0.1:8090/healthz";

// Ports for Martinus-owned services ONLY. We do NOT include 11434 (Ollama) or
// 1234 (LM Studio) — those are the user's own runtimes; killing them would take
// down their models with the Martinus app. Ports 8080/8091/8765 remain in the
// fallback list because they were historically used by the Go orchestrator and
// legacy NiceGUI variants; if a stale process is squatting on them we'll clean
// up on stop.
const MARTINUS_OWNED_PORTS: &[u16] = &[8090, 8080, 8091, 8765, 18080];

// PIDs we spawned in this desktop-app session. `stop_services` targets these
// first so a "Stop local services" click never reaches unrelated processes.
static SPAWNED_PIDS: Mutex<Vec<u32>> = Mutex::new(Vec::new());

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProviderStatus {
    provider: String,
    installed: bool,
    reachable: bool,
    models: Vec<String>,
    message: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeStatus {
    project_root: String,
    default_workspace: String,
    ollama: ProviderStatus,
    lmstudio: ProviderStatus,
    claude: ProviderStatus,
    ai_service_reachable: bool,
    orchestrator_reachable: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ServiceStartResult {
    project_root: String,
    default_workspace: String,
    ai_service_reachable: bool,
    orchestrator_reachable: bool,
    message: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ServiceStopResult {
    ai_service_reachable: bool,
    orchestrator_reachable: bool,
    ollama_reachable: bool,
    lmstudio_reachable: bool,
    message: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ReferenceFile {
    name: String,
    path: String,
    content: String,
}

fn command_exists(binary: &str, args: &[&str]) -> bool {
    command_path(binary)
        .map(|path| {
            Command::new(path)
                .args(args)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|status| status.success())
                .unwrap_or(false)
        })
        .unwrap_or(false)
}

fn command_path(binary: &str) -> Option<PathBuf> {
    let candidate = PathBuf::from(binary);
    if candidate.is_absolute() && candidate.exists() {
        return Some(candidate);
    }

    if let Ok(path_var) = env::var("PATH") {
        for part in env::split_paths(&path_var) {
            for candidate in executable_candidates(&part, binary) {
                if candidate.exists() {
                    return Some(candidate);
                }
            }
        }
    }

    let common_dirs = [
        PathBuf::from("/opt/homebrew/bin"),
        PathBuf::from("/usr/local/bin"),
        PathBuf::from("/usr/bin"),
        PathBuf::from("/bin"),
        home_dir().join(".local").join("bin"),
        home_dir().join(".npm-global").join("bin"),
        home_dir().join("AppData").join("Roaming").join("npm"),
    ];
    for dir in common_dirs {
        for candidate in executable_candidates(&dir, binary) {
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    if binary == "ollama" {
        let app_cli = PathBuf::from("/Applications/Ollama.app/Contents/Resources/ollama");
        if app_cli.exists() {
            return Some(app_cli);
        }
    }

    if binary == "lms" {
        let app_cli = home_dir().join(".lmstudio").join("bin").join("lms");
        if app_cli.exists() {
            return Some(app_cli);
        }
    }

    None
}

fn executable_candidates(dir: &Path, binary: &str) -> Vec<PathBuf> {
    let mut candidates = vec![dir.join(binary)];
    if cfg!(windows) && Path::new(binary).extension().is_none() {
        candidates.push(dir.join(format!("{binary}.exe")));
        candidates.push(dir.join(format!("{binary}.cmd")));
        candidates.push(dir.join(format!("{binary}.bat")));
    }
    candidates
}

fn app_exists(app_name: &str) -> bool {
    let app_paths = [
        PathBuf::from(format!("/Applications/{app_name}.app")),
        home_dir().join("Applications").join(format!("{app_name}.app")),
    ];
    app_paths.iter().any(|path| path.exists())
}

fn home_dir() -> PathBuf {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

/// Suppress the console window Windows allocates for every child process.
/// Without this the app flashes a black cmd window on each runtime probe —
/// and the probes run on every Settings refresh.
fn configure_no_window(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    {
        let _ = command;
    }
}

fn spawn_command(binary: &str) -> Command {
    let mut command = Command::new(command_path(binary).unwrap_or_else(|| PathBuf::from(binary)));
    configure_no_window(&mut command);
    command
}

/// Reveal a path in the platform file manager.
/// Previously hardcoded macOS `open`/`open -R`, which simply does not exist on
/// Windows — "Open workspace" and every file button failed there.
fn reveal_in_file_manager(target: &Path) -> Result<(), String> {
    let mut command;
    #[cfg(target_os = "windows")]
    {
        command = Command::new("explorer");
        if target.is_file() {
            // explorer wants `/select,<path>` as a single argument.
            command.arg(format!("/select,{}", target.display()));
        } else {
            command.arg(target);
        }
    }
    #[cfg(target_os = "macos")]
    {
        command = Command::new("open");
        if target.is_file() {
            command.arg("-R").arg(target);
        } else {
            command.arg(target);
        }
    }
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        command = Command::new("xdg-open");
        let shown = if target.is_file() {
            target.parent().unwrap_or(target)
        } else {
            target
        };
        command.arg(shown);
    }

    configure_no_window(&mut command);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|err| format!("Could not reveal {}: {err}", target.display()))?;
    Ok(())
}

/// Open a file with the OS default handler (used to preview generated HTML).
fn open_with_default_app(target: &Path) -> Result<(), String> {
    let mut command;
    #[cfg(target_os = "windows")]
    {
        // `start` is a cmd builtin, not an executable. The empty "" is the
        // window title, which start would otherwise swallow the path into.
        command = Command::new("cmd");
        command.arg("/C").arg("start").arg("").arg(target);
    }
    #[cfg(target_os = "macos")]
    {
        command = Command::new("open");
        command.arg(target);
    }
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        command = Command::new("xdg-open");
        command.arg(target);
    }

    configure_no_window(&mut command);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|err| format!("Could not open {}: {err}", target.display()))?;
    Ok(())
}

fn command_output(binary: &str, args: &[&str]) -> Option<String> {
    spawn_command(binary)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).to_string())
}

/// `signal` follows the POSIX convention ("-TERM" / "-KILL"). On Windows there
/// are no signals: -TERM maps to a plain taskkill, -KILL adds /F.
fn terminate_pid(pid: &str, signal: &str) -> Result<(), String> {
    let mut command;
    #[cfg(target_os = "windows")]
    {
        command = Command::new("taskkill");
        command.args(["/PID", pid, "/T"]);
        if signal == "-KILL" {
            command.arg("/F");
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        command = Command::new("kill");
        command.args([signal, pid]);
    }

    configure_no_window(&mut command);
    let status = command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|err| format!("Could not send {signal} to pid {pid}: {err}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("terminating pid {pid} ({signal}) exited with status {status}"))
    }
}

fn listener_pids(port: u16) -> Vec<String> {
    #[cfg(target_os = "windows")]
    {
        // No lsof on Windows. Parse `netstat -ano` instead.
        //
        // Deliberately does NOT match the state column against "LISTENING":
        // netstat is localised, so that column reads ABHÖREN on a German
        // Windows, ÉCOUTE on French, and so on — matching the English word
        // finds nothing and "Stop local services" silently becomes a no-op.
        // A listening socket is identified structurally instead: its local
        // address ends in :<port> and it has no peer (remote address :0).
        let suffix = format!(":{port}");
        let mut pids: Vec<String> = Vec::new();
        if let Some(output) = command_output("netstat", &["-ano", "-p", "TCP"]) {
            for line in output.lines() {
                let fields: Vec<&str> = line.split_whitespace().collect();
                if fields.len() < 5 || !fields[0].eq_ignore_ascii_case("TCP") {
                    continue;
                }
                if !fields[1].ends_with(&suffix) || !fields[2].ends_with(":0") {
                    continue;
                }
                let pid = fields[4].trim();
                // PID 0 is the System Idle Process — never a target.
                if pid.chars().all(|c| c.is_ascii_digit())
                    && !pid.is_empty()
                    && pid != "0"
                    && !pids.iter().any(|p| p == pid)
                {
                    pids.push(pid.to_string());
                }
            }
        }
        pids
    }
    #[cfg(not(target_os = "windows"))]
    {
        command_output("lsof", &["-ti", &format!("tcp:{port}")])
            .map(|output| {
                output
                    .lines()
                    .map(str::trim)
                    .filter(|line| !line.is_empty())
                    .map(ToOwned::to_owned)
                    .collect()
            })
            .unwrap_or_default()
    }
}

fn stop_listener_port(port: u16, killed: &mut BTreeSet<String>) -> Result<(), String> {
    let first_pass = listener_pids(port);
    for pid in &first_pass {
        terminate_pid(pid, "-TERM")?;
        killed.insert(pid.clone());
    }
    if !first_pass.is_empty() {
        std::thread::sleep(std::time::Duration::from_millis(300));
    }

    let second_pass = listener_pids(port);
    for pid in &second_pass {
        terminate_pid(pid, "-KILL")?;
        killed.insert(pid.clone());
    }
    Ok(())
}

fn project_root() -> Result<PathBuf, String> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| "Could not resolve the thesis project root from the desktop app".to_string())
}

// Compute the path to the default workspace for THIS running app session.
// Uses the process start time so every UI refresh in the same session resolves
// to the same directory (previously each refresh created a brand-new
// timestamped folder — see the 101 empty workspaces the audit found).
fn desktop_session_stamp() -> u64 {
    static STAMP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *STAMP.get_or_init(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    })
}

fn default_workspace_path(root: &Path) -> PathBuf {
    root.join("evaluation_runs")
        .join(format!("desktop_demo_workspace_{}", desktop_session_stamp()))
}

// Lazy variant: returns the path without creating the directory. Used by
// status polling / UI refresh so cursor-hovering the app doesn't spam empty
// folders under evaluation_runs/.
fn default_workspace_lazy(root: &Path) -> PathBuf {
    default_workspace_path(root)
}

// Eager variant: creates the directory on first use. Called only when the user
// actually kicks off a task (Run) or explicitly starts local services.
fn default_workspace_ensure(root: &Path) -> Result<PathBuf, String> {
    let workspace = default_workspace_path(root);
    fs::create_dir_all(&workspace).map_err(|err| err.to_string())?;
    Ok(workspace)
}

fn ensure_workspace_path(path: &Path) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err("The workspace path must be absolute.".to_string());
    }
    if path.exists() && !path.is_dir() {
        return Err(format!("The workspace path is not a folder: {}", path.display()));
    }
    fs::create_dir_all(path)
        .map_err(|err| format!("Could not create workspace {}: {err}", path.display()))?;
    path.canonicalize()
        .map(strip_extended_length_prefix)
        .map_err(|err| format!("Could not resolve workspace {}: {err}", path.display()))
}

/// Drop the Windows extended-length prefix that `canonicalize()` always adds.
///
/// On Windows `canonicalize()` returns `\\?\F:\path`, and that string is what
/// the UI displays and what gets handed to the Python orchestrator as
/// `workspacePath`. It is ugly in the workspace field, and the `\\?\` form
/// disables the normal path parsing that many tools (and several stdlib
/// helpers on the Python side) assume, so keep the plain `F:\path` form.
/// UNC paths (`\\?\UNC\server\share`) are left untouched — stripping the
/// prefix there would produce a path that no longer resolves.
fn strip_extended_length_prefix(path: PathBuf) -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Some(text) = path.to_str() {
            if let Some(rest) = text.strip_prefix(r"\\?\") {
                let bytes = rest.as_bytes();
                let drive_qualified = bytes.len() >= 2
                    && bytes[0].is_ascii_alphabetic()
                    && bytes[1] == b':';
                if drive_qualified && !rest.starts_with("UNC\\") {
                    return PathBuf::from(rest);
                }
            }
        }
    }
    path
}

// Called at app startup: prune empty `desktop_demo_workspace_*` directories
// left behind by the previous no-lazy behavior so the eval_runs tree stays
// legible. Never touches non-empty directories.
fn cleanup_empty_workspaces(root: &Path) {
    let base = root.join("evaluation_runs");
    let Ok(entries) = fs::read_dir(&base) else { return; };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.starts_with("desktop_demo_workspace_") {
            continue;
        }
        let is_empty = fs::read_dir(&path)
            .map(|mut it| it.next().is_none())
            .unwrap_or(false);
        if is_empty {
            let _ = fs::remove_dir(&path);
        }
    }
}

fn lmstudio_local_models() -> Vec<String> {
    let mut roots = vec![
        home_dir().join(".lmstudio").join("models"),
        home_dir()
            .join("Library")
            .join("Application Support")
            .join("LM Studio")
            .join("models"),
    ];

    let pointer = home_dir().join(".lmstudio-home-pointer");
    if let Ok(path) = fs::read_to_string(pointer) {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            roots.push(PathBuf::from(trimmed).join("models"));
        }
    }

    let mut models = Vec::new();
    for root in roots {
        let Ok(publishers) = fs::read_dir(root) else {
            continue;
        };
        for publisher in publishers.flatten().filter(|entry| entry.path().is_dir()) {
            let Ok(entries) = fs::read_dir(publisher.path()) else {
                continue;
            };
            for model_dir in entries.flatten().filter(|entry| entry.path().is_dir()) {
                let model_path = model_dir.path();
                let has_model_file = fs::read_dir(&model_path)
                    .ok()
                    .map(|files| {
                        files.flatten().any(|file| {
                            file.path()
                                .extension()
                                .and_then(|ext| ext.to_str())
                                .map(|ext| matches!(ext, "gguf" | "safetensors"))
                                .unwrap_or(false)
                        })
                    })
                    .unwrap_or(false);
                if has_model_file {
                    let publisher_name = publisher.file_name().to_string_lossy().to_string();
                    let model_name = model_dir.file_name().to_string_lossy().to_string();
                    models.push(format!("{publisher_name}/{model_name}"));
                }
            }
        }
    }

    models.sort();
    models.dedup();
    models
}

fn http_ok(url: &str) -> bool {
    ureq::get(url)
        .timeout(std::time::Duration::from_secs(2))
        .call()
        .map(|response| (200..300).contains(&response.status()))
        .unwrap_or(false)
}

fn ollama_status() -> ProviderStatus {
    let installed = command_exists("ollama", &["--version"]);
    let mut status = ProviderStatus {
        provider: "ollama".to_string(),
        installed,
        reachable: false,
        models: vec![],
        message: if installed {
                "Ollama is installed, but the local server is not running. Click Start local services or open Ollama and refresh.".to_string()
        } else {
                "Ollama was not found on this computer. Install Ollama, then refresh.".to_string()
        },
    };

    match ureq::get(OLLAMA_TAGS_URL)
        .timeout(std::time::Duration::from_secs(3))
        .call()
    {
        Ok(response) => {
            status.reachable = true;
            match response.into_json::<serde_json::Value>() {
                Ok(data) => {
                    status.models = data["models"]
                        .as_array()
                        .map(|models| {
                            models
                                .iter()
                                .filter_map(|model| model["name"].as_str().map(ToOwned::to_owned))
                                .collect()
                        })
                        .unwrap_or_default();
                    status.message = if status.models.is_empty() {
                        "Ollama is running, but no local models were returned.".to_string()
                    } else {
                        format!("Ollama is ready with {} local model(s).", status.models.len())
                    };
                }
                Err(err) => {
                    status.message = format!("Ollama responded, but the model list could not be parsed: {err}");
                }
            }
        }
        Err(err) => {
            if installed {
                if let Some(list_output) = command_output("ollama", &["list"]) {
                    let models: Vec<String> = list_output
                        .lines()
                        .skip(1)
                        .filter_map(|line| line.split_whitespace().next().map(ToOwned::to_owned))
                        .filter(|name| !name.is_empty())
                        .collect();
                    if !models.is_empty() {
                        status.models = models;
                    }
                }
            }
            status.message = if installed {
                format!("Ollama is installed, but the local server is not running. Click Start local services or open Ollama and refresh. ({err})")
            } else {
                "Ollama was not found on this Mac. Install Ollama or use LM Studio.".to_string()
            };
        }
    }
    status
}

fn lmstudio_status() -> ProviderStatus {
    let installed = app_exists("LM Studio")
        || home_dir().join(".lmstudio").exists()
        || home_dir()
            .join("Library")
            .join("Application Support")
            .join("LM Studio")
            .exists();
    let local_models = lmstudio_local_models();
    let mut status = ProviderStatus {
        provider: "lmstudio".to_string(),
        installed,
        reachable: false,
        models: local_models.clone(),
        message: if installed {
            if local_models.is_empty() {
                "LM Studio is installed, but no downloaded models were found. Start the LM Studio local server on port 1234, then refresh.".to_string()
            } else {
                format!("LM Studio is installed with {} downloaded model(s). Start the local server on port 1234 to run them.", local_models.len())
            }
        } else {
            "LM Studio was not found on this Mac.".to_string()
        },
    };

    match ureq::get(LMSTUDIO_MODELS_URL)
        .timeout(std::time::Duration::from_secs(3))
        .call()
    {
        Ok(response) => {
            status.installed = true;
            status.reachable = true;
            match response.into_json::<serde_json::Value>() {
                Ok(data) => {
                    status.models = data["data"]
                        .as_array()
                        .map(|models| {
                            models
                                .iter()
                                .filter_map(|model| model["id"].as_str().map(ToOwned::to_owned))
                                .collect()
                        })
                        .unwrap_or_default();
                    if status.models.is_empty() {
                        status.models = local_models.clone();
                    }
                    status.message = if status.models.is_empty() {
                        "LM Studio server is running, but no loaded or downloaded models were found.".to_string()
                    } else {
                        format!("LM Studio is ready with {} model(s).", status.models.len())
                    };
                }
                Err(err) => {
                    status.message =
                        format!("LM Studio responded, but the model list could not be parsed: {err}");
                }
            }
        }
        Err(err) => {
            status.message = if installed {
                if status.models.is_empty() {
                    format!("LM Studio is installed, but its local server is not reachable on port 1234. Start the server in LM Studio, then refresh. ({err})")
                } else {
                    format!("LM Studio is installed with {} downloaded model(s), but the local server is not reachable on port 1234. Start the server in LM Studio, then refresh. ({err})", status.models.len())
                }
            } else {
                format!("LM Studio is not reachable and was not found on this Mac: {err}")
            };
        }
    }
    status
}

fn claude_status() -> ProviderStatus {
    let version = command_output("claude", &["--version"]);
    let installed = version.is_some();
    ProviderStatus {
        provider: "claude".to_string(),
        installed,
        reachable: installed,
        models: if installed {
            vec!["claude-code".to_string()]
        } else {
            vec![]
        },
        message: match version {
            Some(output) if !output.trim().is_empty() => {
                format!("Claude Code is ready ({}).", output.trim())
            }
            Some(_) => "Claude Code is installed and ready.".to_string(),
            None => "Claude Code CLI was not found. Install Claude Code, authenticate it, then refresh.".to_string(),
        },
    }
}

fn python_binary() -> String {
    if command_exists("python3", &["--version"]) {
        "python3".to_string()
    } else {
        "python".to_string()
    }
}

// Escape a filesystem path for safe interpolation inside an AppleScript
// double-quoted string. Rejects control characters (newline / carriage return
// / tab / NUL) so an attacker-supplied path cannot break out of the quoted
// literal or inject additional AppleScript statements.
fn escape_applescript_path(path: &str) -> Result<String, String> {
    if path.chars().any(|c| (c as u32) < 0x20) {
        return Err("Path contains a control character and is not usable in a folder picker.".to_string());
    }
    Ok(path.replace('\\', "\\\\").replace('"', "\\\""))
}

#[tauri::command]
async fn get_runtime_status() -> Result<RuntimeStatus, String> {
    // Tauri runs a synchronous `fn` command ON THE MAIN THREAD, which is the
    // same thread that pumps the window event loop — the window cannot process
    // clicks while it runs. This probe spawns several child processes and makes
    // up to four HTTP calls with 2s timeouts each, so on a machine with no
    // local runtime up it froze the UI for seconds at a time. It fires on mount
    // and on every provider change. Push it to the blocking pool instead.
    run_blocking(get_runtime_status_impl).await
}

/// Run a blocking probe off the UI thread and flatten the join error.
async fn run_blocking<T, F>(work: F) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, String> + Send + 'static,
{
    tauri::async_runtime::spawn_blocking(work)
        .await
        .map_err(|err| format!("Background task failed: {err}"))?
}

fn get_runtime_status_impl() -> Result<RuntimeStatus, String> {
    let root = project_root()?;
    // LAZY: do NOT create the directory here. UI refreshes hit this on mount
    // and after every Start/Stop; creating a new empty timestamped workspace
    // per refresh is how we ended up with 100+ empty folders.
    let workspace = default_workspace_lazy(&root);
    Ok(RuntimeStatus {
        project_root: root.display().to_string(),
        default_workspace: workspace.display().to_string(),
        ollama: ollama_status(),
        lmstudio: lmstudio_status(),
        claude: claude_status(),
        ai_service_reachable: http_ok(AI_HEALTH_URL),
        orchestrator_reachable: http_ok(ORCHESTRATOR_HEALTH_URL),
    })
}

#[tauri::command]
fn ensure_workspace(path: String) -> Result<String, String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err("Choose a workspace folder before running a task.".to_string());
    }
    ensure_workspace_path(Path::new(trimmed)).map(|resolved| resolved.display().to_string())
}

fn record_spawned(child_id: u32) {
    if let Ok(mut list) = SPAWNED_PIDS.lock() {
        list.push(child_id);
    }
}

#[tauri::command]
async fn start_services() -> Result<ServiceStartResult, String> {
    // Contains two 2-second sleeps plus health checks; on the main thread that
    // is a hard multi-second UI freeze on every "Start local services" click.
    run_blocking(start_services_impl).await
}

fn start_services_impl() -> Result<ServiceStartResult, String> {
    let root = project_root()?;
    // Eagerly ensure the workspace ONLY when the user actually starts services.
    let workspace = default_workspace_ensure(&root)?;
    let python = python_binary();

    if command_exists("ollama", &["--version"]) && !http_ok(OLLAMA_TAGS_URL) {
        // We intentionally do NOT record this PID: Ollama is the user's runtime,
        // not ours. Stop-services must never terminate it.
        spawn_command("ollama")
            .arg("serve")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|err| format!("Could not start Ollama with `ollama serve`: {err}"))?;
        std::thread::sleep(std::time::Duration::from_secs(2));
    }

    if (app_exists("LM Studio") || command_path("lms").is_some() || home_dir().join(".lmstudio").exists())
        && !http_ok(LMSTUDIO_MODELS_URL)
    {
        if command_path("lms").is_some() {
            // Same rationale — LM Studio's own server is not Martinus-owned.
            spawn_command("lms")
                .args(["server", "start"])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .map_err(|err| format!("Could not start LM Studio server with `lms server start`: {err}"))?;
        } else {
            // Launching the GUI app by name is macOS-only. Elsewhere the `lms`
            // CLI above is the supported path; if it is absent we let the user
            // start LM Studio themselves rather than failing the whole
            // "Start local services" click.
            #[cfg(target_os = "macos")]
            {
                let mut command = Command::new("open");
                command.args(["-a", "LM Studio"]);
                configure_no_window(&mut command);
                command
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()
                    .map_err(|err| format!("Could not open LM Studio: {err}"))?;
            }
        }
    }

    if !http_ok(AI_HEALTH_URL) {
        let child = spawn_command(&python)
            .args(["-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8090"])
            .current_dir(root.join("ai-service"))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|err| format!("Could not start AI service with {python}: {err}"))?;
        record_spawned(child.id());
    }

    if !http_ok(ORCHESTRATOR_HEALTH_URL) {
        let mut command = spawn_command(&python);
        command
            .args([
                "-m",
                "uvicorn",
                "agent.server:app",
                "--host",
                "127.0.0.1",
                "--port",
                "18080",
            ])
            .current_dir(&root)
            .env("AI_SERVICE_URL", "http://127.0.0.1:8090")
            .env("MARTINUS_ENABLE_CLAUDE_ADAPTER", "1")
            .env(
                "ORCHESTRATOR_ALLOWED_ORIGINS",
                "http://127.0.0.1:1420,http://localhost:1420,tauri://localhost,http://tauri.localhost",
            )
            // Deliberately do NOT set OLLAMA_MODEL — runtime detection picks the
            // first loaded model, and the desktop UI passes an explicit
            // `model` per request. Hard-coding a specific model here (previously
            // "gemma4:latest", which does not exist) would 404 every request
            // that omitted the model field.
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        if let Some(claude_binary) = command_path("claude") {
            command.env("MARTINUS_CLAUDE_BINARY", claude_binary);
        }
        let child = command
            .spawn()
            .map_err(|err| format!("Could not start orchestrator with {python}: {err}"))?;
        record_spawned(child.id());
    }

    std::thread::sleep(std::time::Duration::from_secs(2));
    let ai_service_reachable = http_ok(AI_HEALTH_URL);
    let orchestrator_reachable = http_ok(ORCHESTRATOR_HEALTH_URL);
    let message = match (ai_service_reachable, orchestrator_reachable) {
        (true, true) => "Local Martinus services are running.".to_string(),
        (true, false) => "AI service started, but the orchestrator is not reachable yet.".to_string(),
        (false, true) => "Orchestrator is reachable, but the AI memory service is not reachable yet.".to_string(),
        (false, false) => {
            "Services were launched, but health checks are still failing. Check Python dependencies and ports 8090/18080.".to_string()
        }
    };

    Ok(ServiceStartResult {
        project_root: root.display().to_string(),
        default_workspace: workspace.display().to_string(),
        ai_service_reachable,
        orchestrator_reachable,
        message,
    })
}

#[tauri::command]
async fn stop_services() -> Result<ServiceStopResult, String> {
    run_blocking(stop_services_impl).await
}

fn stop_services_impl() -> Result<ServiceStopResult, String> {
    let mut killed = BTreeSet::new();

    // Step 1: SIGTERM each PID we personally spawned this session. This is the
    // safe path — no risk of touching an unrelated user process.
    let spawned: Vec<u32> = SPAWNED_PIDS
        .lock()
        .map(|mut list| {
            let v = list.clone();
            list.clear();
            v
        })
        .unwrap_or_default();
    for pid in &spawned {
        let pid_str = pid.to_string();
        // Best-effort: ignore errors — process may already have exited.
        let _ = terminate_pid(&pid_str, "-TERM");
        killed.insert(pid_str);
    }
    if !spawned.is_empty() {
        std::thread::sleep(std::time::Duration::from_millis(400));
        for pid in &spawned {
            let _ = terminate_pid(&pid.to_string(), "-KILL");
        }
    }

    // Step 2: cleanup pass for Martinus-owned ports that MIGHT still be squatted
    // by a stale run from a previous session. IMPORTANT: this list does NOT
    // include 11434 (Ollama) or 1234 (LM Studio) — those are the user's own
    // runtimes and stopping Martinus must never take them down.
    for &port in MARTINUS_OWNED_PORTS {
        stop_listener_port(port, &mut killed)?;
    }

    std::thread::sleep(std::time::Duration::from_millis(400));

    let ai_service_reachable = http_ok(AI_HEALTH_URL);
    let orchestrator_reachable = http_ok(ORCHESTRATOR_HEALTH_URL);
    let ollama_reachable = http_ok(OLLAMA_TAGS_URL);
    let lmstudio_reachable = http_ok(LMSTUDIO_MODELS_URL);
    let message = if killed.is_empty() {
        "No matching local Martinus services were listening.".to_string()
    } else {
        format!(
            "Stopped {} Martinus process(es). Ollama and LM Studio were left running.",
            killed.len()
        )
    };

    Ok(ServiceStopResult {
        ai_service_reachable,
        orchestrator_reachable,
        ollama_reachable,
        lmstudio_reachable,
        message,
    })
}

#[tauri::command]
fn reveal_path(path: String) -> Result<(), String> {
    let target = PathBuf::from(path);
    if !target.exists() {
        return Err(format!("Path does not exist: {}", target.display()));
    }

    reveal_in_file_manager(&target)
}

#[tauri::command]
fn preview_workspace(path: String) -> Result<(), String> {
    let workspace = ensure_workspace_path(Path::new(path.trim()))?;
    let index = workspace.join("index.html");
    if !index.is_file() {
        return Err(format!(
            "No index.html was found in {}. Preview is available for generated browser projects.",
            workspace.display()
        ));
    }
    open_with_default_app(&index)
}

#[tauri::command]
async fn unload_ollama_model(model: String) -> Result<String, String> {
    run_blocking(move || unload_ollama_model_impl(model)).await
}

fn unload_ollama_model_impl(model: String) -> Result<String, String> {
    let selected = model.trim();
    if selected.is_empty() {
        return Err("Select an Ollama model before unloading it.".to_string());
    }
    let status = spawn_command("ollama")
        .args(["stop", selected])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .status()
        .map_err(|err| format!("Could not ask Ollama to unload {selected}: {err}"))?;
    if !status.success() {
        return Err(format!("Ollama could not unload {selected}."));
    }
    Ok(format!("Requested Ollama to unload {selected} and release its memory."))
}

#[tauri::command]
fn choose_workspace(current: String) -> Result<Option<String>, String> {
    let requested_location = if current.trim().is_empty() {
        project_root()?
    } else {
        PathBuf::from(current)
    };
    let default_location = if requested_location.exists() {
        requested_location
    } else {
        home_dir()
    };
    let script = format!(
        "try\nset selectedFolder to choose folder with prompt \"Choose where Martinus should save generated files\" default location POSIX file \"{}\"\nreturn POSIX path of selectedFolder\non error number -128\nreturn \"\"\nend try",
        escape_applescript_path(&default_location.display().to_string())?
    );

    let output = Command::new("osascript")
        .arg("-e")
        .arg(script)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| format!("Could not open the folder picker: {err}"))?;

    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }

    let selected = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if selected.is_empty() {
        Ok(None)
    } else {
        Ok(Some(selected.trim_end_matches('/').to_string()))
    }
}

#[tauri::command]
fn choose_reference_file(current: String) -> Result<Option<ReferenceFile>, String> {
    let requested_location = if current.trim().is_empty() {
        home_dir()
    } else {
        PathBuf::from(current)
    };
    let default_location = if requested_location.exists() {
        requested_location
    } else {
        home_dir()
    };
    let script = format!(
        "try\nset selectedFile to choose file with prompt \"Choose a spec or reference file for Martinus\" default location POSIX file \"{}\"\nreturn POSIX path of selectedFile\non error number -128\nreturn \"\"\nend try",
        escape_applescript_path(&default_location.display().to_string())?
    );

    let output = Command::new("osascript")
        .arg("-e")
        .arg(script)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| format!("Could not open the file picker: {err}"))?;

    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }

    let selected = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if selected.is_empty() {
        return Ok(None);
    }

    let path = PathBuf::from(&selected);
    let metadata = fs::metadata(&path).map_err(|err| format!("Could not read {}: {err}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!("Selected path is not a file: {}", path.display()));
    }
    if metadata.len() > 1_000_000 {
        return Err("Reference file is too large. Please attach a text spec under 1 MB.".to_string());
    }

    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("Could not read {} as text: {err}", path.display()))?;
    let content: String = raw.chars().take(30_000).collect();
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| "reference-file".to_string());

    Ok(Some(ReferenceFile {
        name,
        path: path.display().to_string(),
        content,
    }))
}

pub fn run() {
    // Best-effort startup cleanup: remove empty desktop_demo_workspace_* folders
    // left behind by prior runs. Never touches non-empty directories.
    if let Ok(root) = project_root() {
        cleanup_empty_workspaces(&root);
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            get_runtime_status,
            ensure_workspace,
            start_services,
            stop_services,
            reveal_path,
            preview_workspace,
            unload_ollama_model,
            choose_workspace,
            choose_reference_file
        ])
        .run(tauri::generate_context!())
        .expect("error while running Martinus desktop app");
}
