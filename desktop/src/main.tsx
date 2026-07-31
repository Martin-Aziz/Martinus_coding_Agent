import React from "react";
import ReactDOM from "react-dom/client";
import { invoke } from "@tauri-apps/api/core";
import {
  Check,
  CheckCircle2,
  ChevronDown,
  Circle,
  Clock3,
  FolderOpen,
  Loader2,
  Paperclip,
  Play,
  RefreshCw,
  Settings,
  Sparkles,
  X,
  XCircle
} from "lucide-react";
import "./styles.css";

type ProviderId = "ollama" | "lmstudio" | "claude";

type ProviderStatus = {
  provider: ProviderId;
  installed: boolean;
  reachable: boolean;
  models: string[];
  message: string;
};

type RuntimeStatus = {
  projectRoot: string;
  defaultWorkspace: string;
  ollama: ProviderStatus;
  lmstudio: ProviderStatus;
  claude: ProviderStatus;
  aiServiceReachable: boolean;
  orchestratorReachable: boolean;
};

type ServiceStartResult = {
  projectRoot: string;
  defaultWorkspace: string;
  aiServiceReachable: boolean;
  orchestratorReachable: boolean;
  message: string;
};

type ServiceStopResult = {
  aiServiceReachable: boolean;
  orchestratorReachable: boolean;
  ollamaReachable: boolean;
  lmstudioReachable: boolean;
  message: string;
};

type WorkspaceEntry = {
  name: string;
  path: string;
  is_dir: boolean;
  size: number;
};

type ReferenceFile = {
  name: string;
  path: string;
  content: string;
};

type PhaseState = "pending" | "active" | "completed" | "failed" | "skipped";
type HardwareProfile = "desktop" | "low" | "medium" | "high";

type PhaseCard = {
  phase: string;
  output: string;
  status: "streaming" | "completed" | "failed";
  attempts?: number;
  model?: string;
  promptVersion?: string;
  retrievedCount?: number;
  startedAt: number;
  duration?: number;
  filesWritten?: string[];
};

type VerificationCheck = {
  file: string;
  ok: boolean;
  kind: string;
  error?: string;
  detail?: string;
};

type VerificationSummary = {
  passed: boolean;
  generatedFiles: string[];
  requestedFiles: string[];
  missingFiles: string[];
  checks: VerificationCheck[];
};

const API = "http://127.0.0.1:18080";
const phases = ["Discovery", "Clarification", "Architecture", "Implementation", "Review", "Verification"];
const profileLabels: Record<HardwareProfile, string> = {
  desktop: "Fast · 2 phases",
  low: "Verified · 3 phases",
  medium: "Full · 6 phases",
  high: "Advanced · 6 phases"
};

const phaseColors: Record<string, string> = {
  Discovery: "#2563eb",
  Clarification: "#7c3aed",
  Architecture: "#4f46e5",
  Implementation: "#059669",
  Review: "#d97706",
  Verification: "#dc2626"
};

const profilePhases: Record<HardwareProfile, string[]> = {
  desktop: ["Discovery", "Implementation"],
  low: ["Discovery", "Implementation", "Verification"],
  medium: phases,
  high: phases
};

const statusLabels: Record<string, string> = {
  idle: "Ready",
  starting: "Starting",
  live: "Running",
  done: "Done",
  error: "Error"
};

const defaultTask = `Create a tiny but beautiful browser calculator project with a polished glass-style HTML/CSS/JavaScript UI.

Write the output as files using this exact format:
<file path="README.md">...</file>
<file path="index.html">...</file>
<file path="styles.css">...</file>
<file path="app.js">...</file>

Keep the implementation small, complete, and easy for a professor to inspect, but make the visible design demo-worthy with real calculator interactions, responsive layout, hover/focus states, and polished visual details.`;

function isBrowserAppTask(taskText: string) {
  const lowered = taskText.toLowerCase();
  return [
    "html",
    "css",
    "javascript",
    "website",
    "web app",
    "browser",
    "landing page",
    "calculator"
  ].some((keyword) => lowered.includes(keyword));
}

function describeNetworkError(err: unknown, target: string) {
  const message = String(err);
  if (
    message.includes("TypeError: Load failed") ||
    message.includes("Failed to fetch") ||
    message.includes("NetworkError") ||
    message.includes("Connection refused") ||
    message.includes("os error 61") ||
    message.includes("os error 111")
  ) {
    return `${target} is not reachable yet. Start local services in Settings, wait a few seconds, and try again.`;
  }
  // Strip low-level "os error NN" noise that leaks through some backends —
  // never surface a raw errno to a non-technical user.
  const cleaned = message.replace(/\s*\(os error \d+\)\s*/gi, "");
  return cleaned;
}

async function waitForServer(url: string, attempts = 12, delayMs = 500) {
  for (let index = 0; index < attempts; index += 1) {
    try {
      const response = await fetch(url, { method: "GET" });
      if (response.ok) return true;
    } catch {
      // Retry until the local server finishes starting.
    }
    await new Promise((resolve) => setTimeout(resolve, delayMs));
  }
  return false;
}

function isTauriPreviewError(err: unknown) {
  return String(err).includes("Cannot read properties of undefined") && String(err).includes("invoke");
}

function providerTone(provider: ProviderStatus | undefined) {
  if (!provider) return "down";
  if (provider.reachable && provider.models.length > 0) return "healthy";
  if (provider.installed || provider.reachable) return "degraded";
  return "down";
}

function ProviderCard({
  title,
  provider,
  selected,
  onSelect
}: {
  title: string;
  provider: ProviderStatus;
  selected: boolean;
  onSelect: () => void;
}) {
  const tone = providerTone(provider);
  const Icon = tone === "healthy" ? CheckCircle2 : tone === "degraded" ? Sparkles : XCircle;
  return (
    <button className={`provider-card ${selected ? "selected" : ""}`} onClick={onSelect}>
      <span className={`provider-status ${tone}`}>
        <Icon size={16} />
      </span>
      <span className="provider-copy">
        <strong>{title}</strong>
        <small>{provider.message}</small>
      </span>
      <span className="pill">{provider.models.length}</span>
    </button>
  );
}

function PhaseStepper({ states, profile }: { states: Record<string, PhaseState>; profile: HardwareProfile }) {
  const activePhases = profilePhases[profile];
  return (
    <div className="stepper">
      {phases.map((phase, index) => {
        const included = activePhases.includes(phase);
        // Not in the active profile → render as "skipped" so the user never
        // reads "the app got stuck on Clarification" when Clarification was
        // never scheduled to run.
        const state = !included ? ("skipped" as PhaseState) : states[phase] ?? "pending";
        return (
          <React.Fragment key={phase}>
            {index > 0 && <span className={`step-line ${state === "completed" ? "completed" : ""}`} />}
            <span className="step-item" title={!included ? `${phase} is skipped by ${profileLabels[profile]}.` : phase}>
              <span className={`step-dot ${state}`}>
                {state === "completed" && <Check size={8} strokeWidth={4} />}
                {state === "failed" && <X size={8} strokeWidth={4} />}
              </span>
              <span className={`step-label ${state}`}>{phase}</span>
            </span>
          </React.Fragment>
        );
      })}
    </div>
  );
}

function MartinusLogo() {
  return (
    <span className="brand-lockup">
      <span className="logo-mark" aria-hidden="true">
        <span>M</span>
      </span>
      <span className="brand-copy">
        <strong>Martinus</strong>
        <small>local coding agent</small>
      </span>
    </span>
  );
}

function PhaseCardView({ card }: { card: PhaseCard }) {
  const [open, setOpen] = React.useState(false);
  const [now, setNow] = React.useState(() => performance.now());
  const color = phaseColors[card.phase] ?? "#64748b";
  React.useEffect(() => {
    if (card.status !== "streaming") return;
    const timer = window.setInterval(() => setNow(performance.now()), 1000);
    return () => window.clearInterval(timer);
  }, [card.status]);
  const duration = card.duration
    ? `${card.duration.toFixed(1)}s`
    : card.status === "streaming"
      ? `${Math.max(0, Math.round((now - card.startedAt) / 1000))}s elapsed`
      : "";
  const filesWritten = card.filesWritten ?? [];
  const phaseSummary = card.status === "streaming"
    ? "The local model is working. Some providers return their text only when the phase finishes."
    : filesWritten.length > 0
      ? `Phase completed and wrote ${filesWritten.length} file${filesWritten.length === 1 ? "" : "s"}: ${filesWritten.join(", ")}.`
      : "Phase completed. The local provider returned no narrative text for this phase.";

  return (
    <article className="phase-card" style={{ borderLeftColor: color }}>
      <button className="phase-header" onClick={() => setOpen((value) => !value)}>
        <span className="phase-dot-solid" style={{ background: color }} />
        {card.status === "streaming" ? <span className="spinner" /> : <Check size={14} />}
        <strong style={{ color }}>{card.phase}</strong>
        {card.status === "streaming" ? (
          <span className="phase-meta">
            <span className="pill running-pill"><Clock3 size={12} /> Running</span>
            <span className="pill">{duration}</span>
          </span>
        ) : (
          <span className="phase-meta">
            {duration && <span className="pill">{duration}</span>}
            {card.attempts ? <span className="pill">{card.attempts} attempt{card.attempts === 1 ? "" : "s"}</span> : null}
            {card.model ? <span className="pill">{card.model}</span> : null}
          </span>
        )}
        <ChevronDown className={open ? "chevron open" : "chevron"} size={14} />
      </button>
      {open && (
        <>
          <div className="phase-activity">
            <strong>Agent activity</strong>
            <span>{phaseSummary}</span>
            {filesWritten.length > 0 && (
              <span className="phase-files">Files written: {filesWritten.join(", ")}</span>
            )}
          </div>
          {card.output && <pre className="phase-body">{card.output}</pre>}
          <footer className="phase-actions">
            {card.output && <button onClick={() => navigator.clipboard.writeText(card.output)}>Copy output</button>}
            {card.retrievedCount ? <span className="pill">retrieved {card.retrievedCount}</span> : null}
            {card.promptVersion ? <span className="pill">{card.promptVersion}</span> : null}
          </footer>
        </>
      )}
    </article>
  );
}

function loadSavedDefaults(): { provider?: ProviderId; model?: string; workspace?: string; hardwareProfile?: HardwareProfile } {
  try {
    const raw = localStorage.getItem("martinus:defaults");
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return typeof parsed === "object" && parsed ? parsed : {};
  } catch {
    return {};
  }
}

function App() {
  const savedDefaults = React.useMemo(loadSavedDefaults, []);
  const [status, setStatus] = React.useState<RuntimeStatus | null>(null);
  const [settingsOpen, setSettingsOpen] = React.useState(false);
  const [provider, setProvider] = React.useState<ProviderId>(savedDefaults.provider ?? "ollama");
  const [model, setModel] = React.useState(savedDefaults.model ?? "");
  const [workspace, setWorkspace] = React.useState(savedDefaults.workspace ?? "");
  const [task, setTask] = React.useState(defaultTask);
  const [sessionId, setSessionId] = React.useState(() => `desktop-${Date.now()}`);
  const [taskHistory, setTaskHistory] = React.useState<string[]>([]);
  const [referenceFiles, setReferenceFiles] = React.useState<ReferenceFile[]>([]);
  const [hardwareProfile, setHardwareProfile] = React.useState<HardwareProfile>(savedDefaults.hardwareProfile ?? "desktop");
  const [phaseStates, setPhaseStates] = React.useState<Record<string, PhaseState>>({});
  const [phaseCards, setPhaseCards] = React.useState<PhaseCard[]>([]);
  const [verificationSummary, setVerificationSummary] = React.useState<VerificationSummary | null>(null);
  const [files, setFiles] = React.useState<WorkspaceEntry[]>([]);
  // Generated files and the "Preview app" button belong to a *finished* run.
  // Gating on this instead of files.length stops the previous run's app from
  // being offered for preview while a new run is still mid-pipeline.
  const [showResults, setShowResults] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [statusText, setStatusText] = React.useState("idle");
  const [serviceMessage, setServiceMessage] = React.useState("");
  const [error, setError] = React.useState("");
  // AbortController for the in-flight task stream. Held in a ref so the Cancel
  // button (which lives outside the runTask closure) can reach it.
  const runAbortRef = React.useRef<AbortController | null>(null);
  const [savedNotice, setSavedNotice] = React.useState("");

  // Streamed tokens are buffered here and flushed on an interval instead of
  // calling setPhaseCards per token. A local model emits tokens far faster than
  // a webview can paint; re-rendering per token saturated the main thread and
  // made clicks feel dropped. macOS WKWebView absorbed it, WebView2 did not.
  const tokenBufferRef = React.useRef<Map<string, string>>(new Map());
  const flushTimerRef = React.useRef<number | null>(null);

  const flushTokens = React.useCallback(() => {
    const buffer = tokenBufferRef.current;
    if (buffer.size === 0) return;
    const pending = new Map(buffer);
    buffer.clear();
    setPhaseCards((current) => {
      let changed = false;
      const next = current.map((card) => {
        if (card.status !== "streaming") return card;
        const addition = pending.get(card.phase);
        if (!addition) return card;
        changed = true;
        return { ...card, output: card.output + addition };
      });
      return changed ? next : current;
    });
  }, []);

  const stopTokenFlush = React.useCallback(() => {
    if (flushTimerRef.current !== null) {
      window.clearInterval(flushTimerRef.current);
      flushTimerRef.current = null;
    }
    flushTokens();
  }, [flushTokens]);

  // Never leave the interval running if the window closes mid-run.
  React.useEffect(() => () => {
    if (flushTimerRef.current !== null) window.clearInterval(flushTimerRef.current);
  }, []);

  const selectedProvider = status?.[provider];
  const models = selectedProvider?.models ?? [];
  const lineBudget = hardwareProfile === "desktop" ? 900 : 2000;
  const displayStatus = statusLabels[statusText] ?? statusText;
  const healthTone =
    selectedProvider?.reachable && models.length > 0 ? "healthy" :
    selectedProvider?.installed ? "degraded" :
    "down";

  const refresh = React.useCallback(async () => {
    setError("");
    try {
      const next = await invoke<RuntimeStatus>("get_runtime_status");
      setStatus(next);
      setWorkspace((current) => current || next.defaultWorkspace);
      const currentProvider = next[provider];
      const preferredProvider: ProviderId =
        currentProvider.reachable || currentProvider.installed || currentProvider.models.length > 0 ? provider :
        next.ollama.reachable || next.ollama.installed || next.ollama.models.length > 0 ? "ollama" :
        next.lmstudio.reachable || next.lmstudio.installed || next.lmstudio.models.length > 0 ? "lmstudio" :
        next.claude.reachable || next.claude.installed || next.claude.models.length > 0 ? "claude" :
        provider;
      setProvider(preferredProvider);
      const preferredModels = next[preferredProvider].models;
      const firstModel = preferredModels[0] ?? "";
      setModel((current) => preferredModels.includes(current) ? current : firstModel);
    } catch (err) {
      if (isTauriPreviewError(err)) {
        setStatus((current) => current ?? {
          projectRoot: "",
          defaultWorkspace: "",
          aiServiceReachable: false,
          orchestratorReachable: false,
          ollama: {
            provider: "ollama",
            installed: false,
            reachable: false,
            models: [],
            message: "Open the packaged Martinus app to detect local runtimes."
          },
          lmstudio: {
            provider: "lmstudio",
            installed: false,
            reachable: false,
            models: [],
            message: "Open the packaged Martinus app to detect local runtimes."
          },
          claude: {
            provider: "claude",
            installed: false,
            reachable: false,
            models: [],
            message: "Open the packaged Martinus app to detect Claude Code."
          }
        });
        return;
      }
      setError(String(err));
    }
  }, [provider]);

  React.useEffect(() => {
    refresh();
  }, [refresh]);

  React.useEffect(() => {
    // Guard on `status`: before the first runtime probe resolves there is no
    // model list to validate against, and `models` is a fresh empty array on
    // every render. Without this guard the effect fired on mount, found the
    // saved model "missing" from an empty list, and reset it to "" — which is
    // why "Save as defaults" never appeared to stick. refresh() then filled in
    // the provider's first model instead of the one the user chose.
    if (!status) return;
    if (!models.includes(model)) {
      setModel(models[0] ?? "");
    }
  }, [status, models, model]);

  async function startLocalServices() {
    setBusy(true);
    setError("");
    setStatusText("starting");
    try {
      const result = await invoke<ServiceStartResult>("start_services");
      setServiceMessage(result.message);
      setWorkspace((current) => current || result.defaultWorkspace);
      await waitForServer(`${API}/healthz`);
      await refresh();
      setStatusText("idle");
    } catch (err) {
      setError(String(err));
      setStatusText("error");
    } finally {
      setBusy(false);
    }
  }

  async function stopLocalServices() {
    setBusy(true);
    setError("");
    try {
      const result = await invoke<ServiceStopResult>("stop_services");
      setServiceMessage(result.message);
      await refresh();
      setStatusText("idle");
    } catch (err) {
      setError(String(err));
      setStatusText("error");
    } finally {
      setBusy(false);
    }
  }

  // Returns the directory listing WITHOUT publishing it to the results panel.
  // The pre-run prompt-context pass needs the listing but must not make the
  // previous run's output visible again mid-run.
  async function fetchWorkspaceFiles(targetWorkspace = workspace): Promise<WorkspaceEntry[]> {
    if (!targetWorkspace) return [];
    const url = new URL(`${API}/v1/workspace/list`);
    url.searchParams.set("workspacePath", targetWorkspace);
    url.searchParams.set("path", ".");
    const response = await fetch(url).catch((err) => {
      throw new Error(describeNetworkError(err, "The local workspace browser"));
    });
    if (!response.ok) throw new Error(await response.text());
    return await response.json();
  }

  async function loadFiles(targetWorkspace = workspace) {
    setFiles(await fetchWorkspaceFiles(targetWorkspace));
  }

  function absoluteWorkspacePath(entry?: WorkspaceEntry) {
    const base = workspace.replace(/\/+$/, "");
    return entry ? `${base}/${entry.path}` : base;
  }

  async function revealInFinder(path: string) {
    try {
      await invoke("reveal_path", { path });
    } catch (err) {
      setError(String(err));
    }
  }

  async function previewGeneratedApp() {
    try {
      await invoke("preview_workspace", { path: workspace });
    } catch (err) {
      setError(String(err));
    }
  }

  async function unloadOllamaModel() {
    if (!model) return;
    try {
      const message = await invoke<string>("unload_ollama_model", { model });
      setServiceMessage(message);
      await refresh();
    } catch (err) {
      setError(String(err));
    }
  }

  async function chooseWorkspace() {
    try {
      const selected = await invoke<string | null>("choose_workspace", { current: workspace });
      if (selected) {
        setWorkspace(selected);
        setFiles([]);
      }
    } catch (err) {
      setError(String(err));
    }
  }

  function resetRunState() {
    setPhaseCards([]);
    setVerificationSummary(null);
    setFiles([]);
    setShowResults(false);
    setPhaseStates(Object.fromEntries(phases.map((phase) => [phase, "pending"])));
  }

  function showPreparingPhase() {
    const firstPhase = profilePhases[hardwareProfile][0] ?? "Discovery";
    setPhaseStates((current) => ({ ...current, [firstPhase]: "active" }));
    setPhaseCards([
      {
        phase: firstPhase,
        output: "Preparing local services and opening the task stream...",
        status: "streaming",
        startedAt: performance.now()
      }
    ]);
  }

  function buildTaskPayload(currentTask: string, currentWorkspace = workspace, knownFiles: WorkspaceEntry[] = files) {
    const previousRequests = taskHistory
      .map((item, index) => `${index + 1}. ${item}`)
      .join("\n\n");
    const generatedFiles = knownFiles
      .map((file) => `- ${file.path}${file.is_dir ? "/" : ""}`)
      .join("\n");
    const attachedReferences = referenceFiles
      .map((file) => `--- ${file.name} (${file.path}) ---\n${file.content}`)
      .join("\n\n");

    return [
      previousRequests ? `Previous requests in this Martinus session:\n${previousRequests}` : "",
      currentWorkspace ? `Current workspace:\n${currentWorkspace}` : "",
      generatedFiles ? `Known generated files in the workspace:\n${generatedFiles}` : "",
      attachedReferences ? `Attached reference files/specs:\n${attachedReferences}` : "",
      `Current request:\n${currentTask}`
    ].filter(Boolean).join("\n\n");
  }

  async function chooseReferenceFile() {
    try {
      const selected = await invoke<ReferenceFile | null>("choose_reference_file", { current: workspace });
      if (!selected) return;
      setReferenceFiles((current) => {
        if (current.some((file) => file.path === selected.path)) return current;
        return [...current, selected];
      });
    } catch (err) {
      setError(String(err));
    }
  }

  function removeReferenceFile(path: string) {
    setReferenceFiles((current) => current.filter((file) => file.path !== path));
  }

  function startNewSession() {
    setSessionId(`desktop-${Date.now()}`);
    setTask(defaultTask);
    setTaskHistory([]);
    setReferenceFiles([]);
    setHardwareProfile("desktop");
    setPhaseStates({});
    setPhaseCards([]);
    setVerificationSummary(null);
    setFiles([]);
    setShowResults(false);
    setError("");
    setStatusText("idle");
  }

  async function runTask() {
    const currentTask = task.trim();
    const browserTask = isBrowserAppTask(currentTask);
    if (!currentTask) {
      setError("Describe what you want to build first.");
      return;
    }
    setBusy(true);
    setError("");
    setStatusText("live");
    resetRunState();
    showPreparingPhase();

    // The status endpoint deliberately reports the default workspace lazily so
    // opening/refreshing the app does not create empty folders. Before a real
    // run, materialize that folder even when the local services were already
    // running. Without this step, the desktop app could submit a nonexistent
    // workspace and receive HTTP 400 while the direct API path still worked.
    let runWorkspace = workspace;
    try {
      if (!runWorkspace) {
        const runtime = await invoke<RuntimeStatus>("get_runtime_status");
        runWorkspace = runtime.defaultWorkspace;
      }
      runWorkspace = await invoke<string>("ensure_workspace", { path: runWorkspace });
      setWorkspace(runWorkspace);
    } catch (err) {
      const msg = describeNetworkError(err, "The workspace");
      setPhaseStates((current) => {
        const next = { ...current };
        for (const [phase, state] of Object.entries(current)) {
          if (state === "active") next[phase] = "failed";
        }
        return next;
      });
      setPhaseCards((current) => current.map((card) => card.status === "streaming" ? {
        ...card,
        status: "failed" as const,
        output: msg
      } : card));
      setError(msg);
      setStatusText("error");
      setBusy(false);
      return;
    }

    // Pre-run: read the file list so buildTaskPayload sees the actual disk
    // state instead of stale React state from a previous run. Swallow errors
    // — a missing workspace here is not fatal for the task itself.
    // Deliberately kept in a local, NOT pushed into `files`: publishing it here
    // re-displayed the previous run's output (and its "Preview app" button)
    // seconds after resetRunState() had just cleared it.
    let contextFiles: WorkspaceEntry[] = [];
    if (runWorkspace) {
      try {
        contextFiles = await fetchWorkspaceFiles(runWorkspace);
      } catch {
        /* ignore — the task run will surface real errors */
      }
    }

    // Fresh AbortController for this run. Any previous Cancel button reference
    // is overwritten.
    const abort = new AbortController();
    runAbortRef.current = abort;

    // Paint buffered tokens ~8x/second. Fast enough to read as live streaming,
    // slow enough to leave the main thread free for input.
    tokenBufferRef.current.clear();
    if (flushTimerRef.current === null) {
      flushTimerRef.current = window.setInterval(flushTokens, 120);
    }

    try {
      let activeModel = model;
      let activeProviderStatus = selectedProvider;

      if (!status?.orchestratorReachable || !selectedProvider?.reachable) {
        setStatusText("starting");
        setPhaseCards((current) => current.map((card, index) => index === 0 ? {
          ...card,
          output: "Starting Martinus services, local runtime, and task server..."
        } : card));
        const result = await invoke<ServiceStartResult>("start_services");
        setServiceMessage(result.message);
        setWorkspace((current) => current || result.defaultWorkspace);
        const ready = await waitForServer(`${API}/healthz`);
        if (!ready) {
          throw new Error("The local Martinus task server is still starting. Wait a few seconds and run the request again.");
        }
        const next = await invoke<RuntimeStatus>("get_runtime_status");
        setStatus(next);
        activeProviderStatus = next[provider];
        if (!activeModel || !activeProviderStatus.models.includes(activeModel)) {
          activeModel = activeProviderStatus.models[0] ?? "";
          setModel(activeModel);
        }
        setStatusText("live");
      }

      if (!activeModel) {
        const providerName = provider === "lmstudio" ? "LM Studio" : provider === "claude" ? "Claude Code" : "Ollama";
        throw new Error(`${providerName} is selected, but no local model was detected. Open Settings, install or start the runtime, then Refresh.`);
      }
      if (!activeProviderStatus?.reachable) {
        const providerName = provider === "lmstudio" ? "LM Studio" : provider === "claude" ? "Claude Code" : "Ollama";
        throw new Error(`${providerName} is selected, but it is not reachable. Install or start the runtime, then Refresh.`);
      }

      const taskPayload = buildTaskPayload(currentTask, runWorkspace, contextFiles);
      const response = await fetch(`${API}/v1/tasks/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sessionId,
          task: taskPayload,
          workspacePath: runWorkspace,
          goals: [
            "Create inspectable local files",
            "Keep output small but visually polished",
            "Produce demo-worthy UI for frontend tasks",
            "Include complete HTML, CSS, and JavaScript when the task asks for a browser app",
            "Use explicit file tags",
            "Preserve session context for follow-up edits",
            ...(browserTask ? [
              "Finish only when every explicitly requested browser file exists in the workspace",
              "Recover any missing HTML, CSS, or JavaScript file before completion"
            ] : [])
          ],
          provider,
          model: activeModel,
          hardwareProfile,
          maxLines: lineBudget
        }),
        signal: abort.signal
      }).catch((err) => {
        if (err?.name === "AbortError") {
          throw new Error("Run cancelled.");
        }
        throw new Error(describeNetworkError(err, "The local Martinus task server"));
      });

      if (!response.ok || !response.body) {
        throw new Error(await response.text());
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";
        for (const raw of events) {
          const line = raw.split("\n").find((part) => part.startsWith("data: "));
          if (!line) continue;
          const event = JSON.parse(line.slice(6));

          if (event.type === "phase_started") {
            const phase = event.phase ?? "Agent";
            // This phase's card is about to be reset to empty output; drop any
            // buffered text so it cannot be appended after the reset.
            tokenBufferRef.current.delete(phase);
            setPhaseStates((current) => ({ ...current, [phase]: "active" }));
            setPhaseCards((current) => {
              const index = current.findIndex((card) => card.phase === phase && card.status === "streaming");
              if (index >= 0) {
                const next = [...current];
                next[index] = { ...next[index], output: "", startedAt: performance.now() };
                return next;
              }
              return [
                ...current,
                { phase, output: "", status: "streaming", startedAt: performance.now() }
              ];
            });
          }

          if (event.type === "token") {
            // Buffer only — the interval started below paints these in batches.
            const phase = event.phase ?? "Agent";
            const buffer = tokenBufferRef.current;
            buffer.set(phase, (buffer.get(phase) ?? "") + (event.token ?? ""));
          }

          if (event.type === "files_written") {
            const phase = event.phase ?? "Agent";
            const written = Array.isArray(event.files) ? event.files.filter((item: unknown) => typeof item === "string") : [];
            setPhaseCards((current) => {
              const next = [...current];
              const index = next.findIndex((card) => card.phase === phase);
              if (index >= 0) {
                next[index] = {
                  ...next[index],
                  filesWritten: Array.from(new Set([...(next[index].filesWritten ?? []), ...written]))
                };
              } else {
                next.push({ phase, output: "", status: "completed", startedAt: performance.now(), filesWritten: written });
              }
              return next;
            });
          }

          if (event.type === "phase_completed") {
            const phase = event.phase ?? "Agent";
            // Drain buffered tokens before the card leaves "streaming" state —
            // the flush only targets streaming cards, so anything still pending
            // would otherwise be stranded.
            flushTokens();
            setPhaseStates((current) => ({ ...current, [phase]: "completed" }));
            setPhaseCards((current) => {
              const next = [...current];
              const index = next.findIndex((card) => card.phase === phase && card.status === "streaming");
              const currentCard = index >= 0 ? next[index] : { phase, output: "", status: "streaming" as const, startedAt: performance.now() };
              const completed = {
                ...currentCard,
                status: "completed" as const,
                output: event.output ?? currentCard.output,
                attempts: event.attempts,
                model: event.model,
                promptVersion: event.promptVersion,
                retrievedCount: event.retrievedCount,
                duration: (performance.now() - currentCard.startedAt) / 1000
              };
              if (index >= 0) next[index] = completed;
              else next.push(completed);
              return next;
            });
          }

          if (event.type === "phase_failed") {
            const phase = event.phase ?? "Agent";
            flushTokens();
            setPhaseStates((current) => ({ ...current, [phase]: "failed" }));
            setPhaseCards((current) => [
              ...current,
              { phase, status: "failed", output: event.error ?? "Phase failed.", startedAt: performance.now() }
            ]);
          }

          if (event.type === "error") {
            // Mark any currently-streaming phase as failed BEFORE the throw so
            // the UI does not leave a spinner running under the error banner.
            flushTokens();
            setPhaseStates((current) => {
              const next = { ...current };
              for (const [k, v] of Object.entries(current)) {
                if (v === "active") next[k] = "failed";
              }
              return next;
            });
            setPhaseCards((current) => current.map((c) => c.status === "streaming" ? {
              ...c,
              status: "failed" as const,
              output: c.output || (event.message ?? "Agent execution failed.")
            } : c));
            throw new Error(event.message ?? "Agent execution failed.");
          }

          if (event.type === "verification_completed") {
            setVerificationSummary({
              passed: Boolean(event.passed),
              generatedFiles: Array.isArray(event.generatedFiles) ? event.generatedFiles : [],
              requestedFiles: Array.isArray(event.requestedFiles) ? event.requestedFiles : [],
              missingFiles: Array.isArray(event.missingFiles) ? event.missingFiles : [],
              checks: Array.isArray(event.checks) ? event.checks : []
            });
          }

          if (event.type === "run_completed") {
            if (event.verification) {
              setVerificationSummary(event.verification as VerificationSummary);
            }
            setStatusText("done");
          }
        }
      }
      setTaskHistory((current) => [...current, currentTask]);
      setTask("");
    } catch (err) {
      const msg = String(err);
      setPhaseStates((current) => {
        const next = { ...current };
        for (const [phase, state] of Object.entries(current)) {
          if (state === "active") next[phase] = "failed";
        }
        return next;
      });
      setPhaseCards((current) => current.map((card) => card.status === "streaming" ? {
        ...card,
        status: "failed" as const,
        output: card.output || msg.replace(/^Error:\s*/, "")
      } : card));
      setError(msg);
      setStatusText("error");
    } finally {
      stopTokenFlush();
      // Always refresh files at the end — even on failure/cancel, partial output
      // may have been written and hiding it wastes time for the professor.
      try {
        if (runWorkspace) {
          await loadFiles(runWorkspace);
          setShowResults(true);
        }
      } catch {
        /* ignore — file listing is best-effort */
      }
      runAbortRef.current = null;
      setBusy(false);
      setTimeout(() => setStatusText((s) => (s === "error" ? "error" : "idle")), 1400);
    }
  }

  function cancelRun() {
    const ctrl = runAbortRef.current;
    if (ctrl) ctrl.abort();
  }

  const canRun =
    Boolean(task.trim()) &&
    !busy;
  const runDisabledReason =
    !task.trim() ? "Describe what you want Martinus to build first." :
    busy ? "Martinus is already running." :
    "";
  const hasPreview = files.some((file) => !file.is_dir && file.path.toLowerCase() === "index.html");

  return (
    <main className="app-shell">
      <header className="topbar">
        <h1 className="topbar-title"><MartinusLogo /></h1>
        <span className="topbar-spacer" />
        <button
          className="mode-pill"
          onClick={startNewSession}
          title="Start a fresh Martinus session and clear the current run history."
        >
          <RefreshCw size={14} />
          New session
        </button>
        <button className="icon-button" aria-label="Settings" title="Open runtime, model, workspace, and hardware profile settings." onClick={() => setSettingsOpen(true)}>
          <Settings size={18} />
        </button>
        <button className="health-pill" onClick={refresh} title="Refresh local runtime detection for Ollama, LM Studio, and available models.">
          <span className={`health-dot ${healthTone}`} />
          {selectedProvider ? `${provider} · ${model || "no model"}` : "Health"}
        </button>
        <span className={`status-badge ${statusText}`} title="Current task execution state.">
          <Circle size={8} fill="currentColor" />
          {displayStatus}
        </span>
      </header>

      <PhaseStepper states={phaseStates} profile={hardwareProfile} />
      <div className="pipeline-note">
        {hardwareProfile === "desktop"
          ? "Fast mode runs Discovery and Implementation. Automatic file and syntax checks still run after generation."
          : hardwareProfile === "low"
            ? "Verified mode adds the model-led Verification phase and automatic file and syntax checks."
            : "Full pipeline mode runs all six agent phases plus automatic file and syntax checks."}
      </div>

      <section className="chat-area">
        {error && <div className="error-banner">{error}</div>}
        {(taskHistory.length > 0 || referenceFiles.length > 0) && (
          <div className="session-card">
            <strong>Session context</strong>
            <span>{taskHistory.length} request{taskHistory.length === 1 ? "" : "s"} in this session</span>
            {referenceFiles.length > 0 && <span>{referenceFiles.length} attached spec/reference file{referenceFiles.length === 1 ? "" : "s"}</span>}
            <small>{sessionId}</small>
          </div>
        )}
        {phaseCards.length === 0 ? (
          <div className="empty-run">
            <span>{taskHistory.length > 0 ? "Ready for a follow-up" : "Ready"}</span>
            <p>{taskHistory.length > 0 ? "Ask for a change, question, or improvement. Martinus will include this session history and the selected workspace context." : "Describe what you want to build. Martinus will run the local coding pipeline and write generated files into the selected workspace."}</p>
          </div>
        ) : (
          phaseCards.map((card, index) => <PhaseCardView card={card} key={`${card.phase}-${index}`} />)
        )}

        {verificationSummary && (
          <div className={`verification-card ${verificationSummary.passed ? "passed" : "failed"}`}>
            <div className="verification-title">
              {verificationSummary.passed ? <CheckCircle2 size={18} /> : <XCircle size={18} />}
              <span>
                <strong>{verificationSummary.passed ? "Verification passed" : "Verification needs attention"}</strong>
                <small>Observable file and syntax checks—not hidden model reasoning.</small>
              </span>
            </div>
            <div className="verification-stats">
              <span>{verificationSummary.generatedFiles.length} generated file{verificationSummary.generatedFiles.length === 1 ? "" : "s"}</span>
              <span>{verificationSummary.checks.filter((check) => check.ok).length}/{verificationSummary.checks.length} executable checks passed</span>
              <span>{verificationSummary.missingFiles.length === 0 ? "No requested files missing" : `${verificationSummary.missingFiles.length} requested file(s) missing`}</span>
            </div>
            {verificationSummary.checks.length > 0 && (
              <div className="verification-checks">
                {verificationSummary.checks.map((check) => (
                  <span className={check.ok ? "ok" : "bad"} key={`${check.file}-${check.kind}`} title={check.error || check.detail || check.kind}>
                    {check.ok ? "✓" : "×"} {check.file} · {check.kind}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        {showResults && files.length > 0 && (
          <div className="summary-card">
            <div className="summary-header">
              <span>
                <strong>Generated files</strong>
                <small>{workspace}</small>
              </span>
              <button onClick={() => revealInFinder(absoluteWorkspacePath())}>
                <FolderOpen size={15} />
                Open workspace
              </button>
              {hasPreview && (
                <button onClick={previewGeneratedApp}>
                  <Play size={15} />
                  Preview app
                </button>
              )}
            </div>
            <div className="file-list">
              {files.map((file) => (
                <button key={file.path} onClick={() => revealInFinder(absoluteWorkspacePath(file))}>
                  {file.is_dir ? "Folder" : "File"}: {file.path}
                </button>
              ))}
            </div>
          </div>
        )}
      </section>

      <footer className="input-bar">
        <div className="workspace-row">
          <span>Workspace</span>
          <span className="workspace-picker">
            <input value={workspace} onChange={(event) => setWorkspace(event.target.value)} />
            <button type="button" onClick={chooseWorkspace} title="Choose the folder where generated files should be saved.">
              <FolderOpen size={15} />
              Browse
            </button>
          </span>
        </div>

        {referenceFiles.length > 0 && (
          <div className="attachment-row">
            {referenceFiles.map((file) => (
              <span className="attachment-chip" key={file.path} title={file.path}>
                <Paperclip size={13} />
                {file.name}
                <button type="button" onClick={() => removeReferenceFile(file.path)} aria-label={`Remove ${file.name}`}>
                  <X size={12} />
                </button>
              </span>
            ))}
          </div>
        )}

        <div className="task-textarea-wrap">
          <textarea
            value={task}
            onChange={(event) => setTask(event.target.value)}
            placeholder={taskHistory.length > 0 ? "Ask a follow-up, for example: make the calculator responsive, add keyboard support, or explain where the files are saved." : "Describe what you want to build...\n\nBe specific about features, tech stack, and requirements. You can attach a short spec file with the paperclip."}
          />
          <div className={`char-counter ${task.length > 1800 ? "warn" : ""}`}>{task.length} / 2048</div>
        </div>

        <div className="input-actions">
          <button className="icon-button" aria-label="Attach reference" title="Attach a short spec, requirements document, or reference file for the next request." onClick={chooseReferenceFile}>
            <Paperclip size={18} />
          </button>
          <select
            className="profile-pill"
            value={hardwareProfile}
            onChange={(event) => setHardwareProfile(event.target.value as HardwareProfile)}
            title="Choose how many pipeline phases and reasoning techniques Martinus should use."
          >
            {Object.entries(profileLabels).map(([value, label]) => (
              <option value={value} key={value}>{label}</option>
            ))}
          </select>
          <button
            className="run-button"
            disabled={!canRun}
            onClick={runTask}
            title={runDisabledReason || (taskHistory.length > 0 ? "Apply this follow-up change using the current session context." : "Run Martinus and write generated files to the selected workspace.")}
          >
            {busy ? <Loader2 className="spin" size={16} /> : <Play size={16} />}
            {taskHistory.length > 0 ? "Apply Change" : "Run"}
          </button>
          {busy && (
            <button
              className="run-button"
              onClick={cancelRun}
              title="Cancel the current Martinus run."
              style={{ marginLeft: 8 }}
            >
              <X size={16} />
              Cancel
            </button>
          )}
        </div>
      </footer>

      {settingsOpen && <button className="drawer-backdrop" aria-label="Close settings" onClick={() => setSettingsOpen(false)} />}
      <aside className={`settings-drawer ${settingsOpen ? "open" : ""}`} aria-hidden={!settingsOpen}>
        <div className="drawer-title">
          <strong>Settings</strong>
          <button className="icon-button" aria-label="Close settings" onClick={() => setSettingsOpen(false)}>
            <X size={18} />
          </button>
        </div>

        <section className="drawer-section">
          <label>Session ID</label>
          <div className="input-with-action">
            <input value={sessionId} readOnly />
            <button className="icon-button" aria-label="Start new session" onClick={startNewSession} title="Start a fresh session and clear the current context."><RefreshCw size={16} /></button>
          </div>
        </section>

        <section className="drawer-section">
          <label>Workspace Path</label>
          <div className="workspace-picker">
            <input value={workspace} onChange={(event) => setWorkspace(event.target.value)} />
            <button type="button" onClick={chooseWorkspace} title="Choose the folder where generated files should be saved.">
              <FolderOpen size={15} />
              Browse
            </button>
          </div>
          <button className="flat-button" onClick={() => revealInFinder(absoluteWorkspacePath())}>
            <FolderOpen size={16} />
            Open workspace
          </button>
        </section>

        <section className="drawer-section">
          <label>Runtime</label>
          {status && (
            <div className="provider-list">
              <ProviderCard title="Ollama" provider={status.ollama} selected={provider === "ollama"} onSelect={() => setProvider("ollama")} />
              <ProviderCard title="LM Studio" provider={status.lmstudio} selected={provider === "lmstudio"} onSelect={() => setProvider("lmstudio")} />
              <ProviderCard title="Claude Code" provider={status.claude} selected={provider === "claude"} onSelect={() => setProvider("claude")} />
            </div>
          )}
          <div className="runtime-actions">
            <button className="flat-button" onClick={startLocalServices} disabled={busy}>
              {busy ? <Loader2 className="spin" size={16} /> : <Play size={16} />}
              Start local services
            </button>
            <button className="flat-button" onClick={stopLocalServices} disabled={busy}>
              <XCircle size={16} />
              Stop local services
            </button>
            <button className="flat-button" onClick={refresh} disabled={busy}>
              <RefreshCw size={16} />
              Refresh
            </button>
          </div>
          <p className="hint">
            {selectedProvider?.message || serviceMessage || "Local model runtime can use Ollama, LM Studio, or Claude Code."}
          </p>
        </section>

        <section className="drawer-section">
          <label>Model</label>
          <select value={model} onChange={(event) => setModel(event.target.value)}>
            {models.length === 0 && <option value="">Auto (hardware profile)</option>}
            {models.map((item) => <option value={item} key={item}>{item}</option>)}
          </select>
          {provider === "ollama" && model && (
            <button className="flat-button" onClick={unloadOllamaModel} disabled={busy}>
              <XCircle size={16} />
              Unload selected Ollama model
            </button>
          )}
          <p className="hint">Unload a large Ollama model before switching runtimes to release RAM.</p>
        </section>

        <section className="drawer-section">
          <label>Hardware Profile</label>
          <select value={hardwareProfile} onChange={(event) => setHardwareProfile(event.target.value as HardwareProfile)}>
            {Object.entries(profileLabels).map(([value, label]) => (
              <option value={value} key={value}>{label}</option>
            ))}
          </select>
          <div className="profile-help">
            <strong>Profile descriptions:</strong>
            <span>Fast · 2 phases — Discovery and Implementation; best for quick local generation</span>
            <span>Verified · 3 phases — adds the model-led Verification phase</span>
            <span>Full · 6 phases — Clarification, Architecture, Review, and Verification included</span>
            <span>Advanced · 6 phases — full pipeline plus advanced reasoning techniques</span>
            <span>All modes still run deterministic file and syntax checks after generation.</span>
          </div>
        </section>

        <section className="drawer-section">
          <label>Line Budget</label>
          <input type="number" min="100" max="2000" value={lineBudget} readOnly />
        </section>

        <button
          className="save-defaults"
          onClick={() => {
            try {
              localStorage.setItem("martinus:defaults", JSON.stringify({
                provider,
                model,
                workspace,
                hardwareProfile,
              }));
              // Confirm in the drawer itself. This used to write to
              // `serviceMessage`, which only renders behind
              // `selectedProvider?.message ||` — that is effectively always
              // set, so the confirmation was never visible and saving looked
              // like it had silently failed.
              setSavedNotice(`Saved — ${provider}${model ? ` · ${model}` : ""} will be selected next launch.`);
              window.setTimeout(() => setSavedNotice(""), 4000);
            } catch (err) {
              setError(`Could not save defaults: ${String(err)}`);
            }
          }}
        >Save as defaults</button>
        {savedNotice && <p className="hint saved-notice">{savedNotice}</p>}
      </aside>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
