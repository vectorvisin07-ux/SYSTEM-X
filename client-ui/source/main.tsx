import {
  FormEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

const routes = ["chat", "catalogue", "agent", "developer", "system"] as const;
type Route = (typeof routes)[number];
type Role = "user" | "assistant";
type Message = { role: Role; content: string };
type JsonRecord = Record<string, unknown>;
type StreamResult = {
  text: string;
  started: boolean;
  deltaCount: number;
  reasoningDeltaCount: number;
  completed: boolean;
  finishReason: string | null;
  requestId: string | null;
  operationId: string | null;
};

const SYSTEM_PROMPT =
  "Give the final answer in normal assistant content. Do not leave the answer only in reasoning content.";

function routeFromHash(): Route {
  const candidate = location.hash.slice(1) as Route;
  return routes.includes(candidate) ? candidate : "chat";
}

function safeText(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function eventData(line: string): JsonRecord | null {
  if (!line.startsWith("data:")) return null;
  try {
    const value: unknown = JSON.parse(line.slice(5).trim());
    return value && typeof value === "object" ? (value as JsonRecord) : null;
  } catch {
    return null;
  }
}

function streamValue(event: JsonRecord, ...keys: string[]): string {
  for (const key of keys) {
    const value = safeText(event[key]);
    if (value) return value;
  }
  const output = event.output;
  if (output && typeof output === "object") {
    return streamValue(output as JsonRecord, ...keys);
  }
  return "";
}

async function readStream(
  response: Response,
  onDelta: (text: string) => void,
  onReasoning: () => void,
): Promise<StreamResult> {
  if (!response.body) throw new Error("The local stream is unavailable");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  let text = "";
  let started = false;
  let deltaCount = 0;
  let reasoningDeltaCount = 0;
  let completed = false;
  let finishReason: string | null = null;
  let requestId = response.headers.get("x-system-x-request-id");
  let operationId: string | null = null;
  const consume = (block: string) => {
    let eventName = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) eventName = line.slice(6).trim();
      const event = eventData(line);
      if (!event) continue;
      const currentType = safeText(event.type) || eventName;
      requestId = requestId || safeText(event.request_id);
      operationId = operationId || safeText(event.operation_id);
      if (currentType === "response.started") started = true;
      if (currentType === "response.reasoning.delta") {
        reasoningDeltaCount += 1;
        onReasoning();
      }
      if (currentType === "response.output_text.delta") {
        const delta = streamValue(event, "delta", "text", "content");
        if (delta) {
          text += delta;
          deltaCount += 1;
          onDelta(text);
        }
      }
      if (
        currentType === "response.completed" ||
        currentType === "response.incomplete"
      ) {
        completed = currentType === "response.completed";
        finishReason =
          streamValue(event, "finish_reason", "finish") || finishReason;
      }
    }
  };
  while (true) {
    const chunk = await reader.read();
    if (chunk.done) break;
    pending += decoder.decode(chunk.value, { stream: true });
    const blocks = pending.split("\n\n");
    pending = blocks.pop() || "";
    blocks.forEach(consume);
  }
  pending += decoder.decode();
  if (pending.trim()) consume(pending);
  return {
    text,
    started,
    deltaCount,
    reasoningDeltaCount,
    completed,
    finishReason,
    requestId,
    operationId,
  };
}

function App() {
  const [route, setRoute] = useState<Route>(routeFromHash);
  const [draft, setDraft] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState(false);
  const [csrf, setCsrf] = useState<string | null>(null);
  const [sessionReady, setSessionReady] = useState(false);
  const [notice, setNotice] = useState("");
  const [catalogue, setCatalogue] = useState<JsonRecord | null>(null);
  const [workspaceData, setWorkspaceData] = useState<JsonRecord | null>(null);
  const [activation, setActivation] = useState<JsonRecord | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const update = () => setRoute(routeFromHash());
    addEventListener("hashchange", update);
    return () => removeEventListener("hashchange", update);
  }, []);

  useEffect(() => {
    fetch("/ui/session/state", { credentials: "same-origin" })
      .then(async (response) => {
        if (!response.ok) throw new Error("Studio session unavailable");
        return (await response.json()) as JsonRecord;
      })
      .then((state) => {
        const token = safeText(state.csrf);
        setCsrf(token || null);
        setSessionReady(Boolean(state.authenticated && token));
      })
      .catch(() => {
        setSessionReady(false);
        setNotice("Open the Studio through the System X front door.");
      });
  }, []);

  const requestJson = async (
    path: string,
    init: RequestInit = {},
  ): Promise<JsonRecord> => {
    const headers = new Headers(init.headers);
    headers.set("Accept", "application/json");
    if (csrf) headers.set("x-studio-csrf", csrf);
    const response = await fetch(path, {
      ...init,
      credentials: "same-origin",
      headers,
    });
    if (!response.ok) throw new Error("The local session is unavailable");
    return (await response.json()) as JsonRecord;
  };

  useEffect(() => {
    if (!csrf || route === "chat") return;
    setWorkspaceData(null);
    setActivation(null);
    const path =
      route === "catalogue" ? "/ui/studio/catalogue" : "/ui/studio/" + route;
    requestJson(path)
      .then((data) => {
        if (route === "catalogue") setCatalogue(data);
        else setWorkspaceData(data);
      })
      .catch((error: unknown) =>
        setNotice(
          error instanceof Error ? error.message : "Workspace unavailable",
        ),
      );
  }, [csrf, route]);

  const send = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const prompt = draft.trim();
    if (!prompt || busy || !csrf) return;
    const nextMessages = [
      ...messages,
      { role: "user" as const, content: prompt },
    ];
    setDraft("");
    setMessages([...nextMessages, { role: "assistant", content: "" }]);
    setNotice("");
    setBusy(true);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const response = await fetch("/system/v1/chat", {
        method: "POST",
        credentials: "same-origin",
        signal: controller.signal,
        headers: {
          Accept: "text/event-stream",
          "Content-Type": "application/json",
          "x-studio-csrf": csrf,
        },
        body: JSON.stringify({
          model: "default",
          messages: [{ role: "system", content: SYSTEM_PROMPT }, ...nextMessages],
          stream: true,
          max_output_tokens: 1024,
        }),
      });
      if (!response.ok) throw new Error("The local session is unavailable");
      const result = await readStream(
        response,
        (text) =>
          setMessages((current) =>
            current.map((message, index) =>
              index === current.length - 1
                ? { ...message, content: text }
                : message,
            ),
          ),
        () => undefined,
      );
      if (!result.completed || !result.text.trim()) {
        throw new Error("The response did not contain final assistant content");
      }
      setMessages((current) =>
        current.map((message, index) =>
          index === current.length - 1
            ? { ...message, content: result.text }
            : message,
        ),
      );
    } catch (error: unknown) {
      if (error instanceof DOMException && error.name === "AbortError") {
        setMessages(nextMessages);
        setNotice("Request cancelled.");
      } else {
        setMessages((current) => current.slice(0, -1));
        setNotice(error instanceof Error ? error.message : "Request failed");
      }
    } finally {
      abortRef.current = null;
      setBusy(false);
    }
  };

  const activateCurrent = async () => {
    try {
      setActivation(
        await requestJson("/ui/studio/activation", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: "default" }),
        }),
      );
    } catch (error: unknown) {
      setNotice(
        error instanceof Error ? error.message : "Activation unavailable",
      );
    }
  };

  const title = route[0].toUpperCase() + route.slice(1);
  const modelRows = useMemo(() => {
    const rows = catalogue?.models;
    return Array.isArray(rows)
      ? rows.filter(
          (row): row is JsonRecord =>
            Boolean(row && typeof row === "object"),
        )
      : [];
  }, [catalogue]);

  const renderWorkspace = () => {
    if (route === "chat") {
      return (
        <>
          <div className="messages" aria-live="polite">
            {messages.length ? (
              messages.map((message, index) => (
                <p
                  className={"message " + message.role}
                  key={message.role + "-" + index}
                >
                  <strong>{message.role === "user" ? "You" : "System X"}</strong>
                  <span>
                    {message.content ||
                      (busy && index === messages.length - 1 ? "…" : "")}
                  </span>
                </p>
              ))
            ) : (
              <i>Start a private conversation.</i>
            )}
          </div>
          <form onSubmit={send}>
            <label htmlFor="prompt">Message</label>
            <textarea
              id="prompt"
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              disabled={busy || !sessionReady}
              aria-describedby="session-status"
            />
            <div className="form-actions">
              <button
                type="submit"
                disabled={busy || !draft.trim() || !sessionReady}
              >
                Send
              </button>
              <button
                type="button"
                onClick={() => {
                  setMessages([]);
                  setDraft("");
                  setNotice("");
                }}
              >
                New conversation
              </button>
              <button
                type="button"
                className="stop"
                onClick={() => abortRef.current?.abort()}
                disabled={!busy}
              >
                Stop
              </button>
            </div>
          </form>
        </>
      );
    }
    if (route === "catalogue") {
      return (
        <article className="panel">
          <h2>Model catalogue</h2>
          <p>Live public model metadata from the authenticated System X owner.</p>
          <div className="model-list">
            {modelRows.length ? (
              modelRows.map((model) => (
                <div className="model-card" key={safeText(model.id)}>
                  <strong>{safeText(model.id)}</strong>
                  <span>
                    {safeText(model.registration_state)} ·{" "}
                    {safeText(model.runtime_state)}
                  </span>
                  <small>
                    {Array.isArray(model.aliases)
                      ? model.aliases.join(", ")
                      : ""}
                  </small>
                </div>
              ))
            ) : (
              <i>Loading the current model catalogue…</i>
            )}
          </div>
          <button
            type="button"
            onClick={activateCurrent}
            disabled={!sessionReady}
          >
            Use current model
          </button>
          {activation && (
            <p role="status">
              Activation: {safeText(activation.classification)}; deployments
              changed: {String(activation.deployment_delta ?? 0)}.
            </p>
          )}
        </article>
      );
    }
    if (route === "agent") {
      const capabilities = workspaceData?.capabilities;
      const toolCalling =
        capabilities && typeof capabilities === "object"
          ? safeText((capabilities as JsonRecord).tool_calling)
          : "not_tested";
      return (
        <article className="panel">
          <h2>Agent workspace</h2>
          <p>Tool execution is bounded by the current model capability ceiling.</p>
          <p role="status">
            Tool calling: <strong>{toolCalling}</strong>
          </p>
          <button type="button" disabled={toolCalling !== "available"}>
            Submit synthetic tool
          </button>
          <small>
            System X does not execute shell, filesystem, repository, browser,
            MCP, or network tools from this workspace.
          </small>
        </article>
      );
    }
    if (route === "developer") {
      const protocols = Array.isArray(workspaceData?.protocols)
        ? workspaceData.protocols.map(safeText)
        : [];
      return (
        <article className="panel">
          <h2>Developer workspace</h2>
          <p>
            Authenticated protocol surfaces, restricted to this same-origin
            Studio.
          </p>
          <ul>
            {protocols.map((protocol) => (
              <li key={protocol}>{protocol}</li>
            ))}
          </ul>
          <p role="status">
            External origins: rejected · private topology: hidden · secret
            material: hidden.
          </p>
        </article>
      );
    }
    const data = workspaceData || {};
    return (
      <article className="panel">
        <h2>System workspace</h2>
        <dl>
          <dt>Installation</dt>
          <dd>{safeText(data.installation_state) || "Loading"}</dd>
          <dt>Service</dt>
          <dd>{safeText(data.service_state) || "Loading"}</dd>
          <dt>Readiness</dt>
          <dd>{safeText(data.readiness_state) || "Loading"}</dd>
          <dt>Model</dt>
          <dd>{safeText(data.model_state) || "Loading"}</dd>
          <dt>Connection</dt>
          <dd>{safeText(data.connection_state) || "Loading"}</dd>
          <dt>Default</dt>
          <dd>{safeText(data.default_alias) || "default"}</dd>
          <dt>Model children</dt>
          <dd>{String(data.model_child_count ?? "Loading")}</dd>
        </dl>
        <small>
          Private topology, credentials, paths, and direct control operations
          are not exposed.
        </small>
      </article>
    );
  };

  return (
    <div className="studio">
      <header>
        <a href="#chat" className="brand">
          SYSTEM X <span>Studio</span>
        </a>
        <nav aria-label="Primary">
          {routes.map((item) => (
            <button
              className={route === item ? "active" : ""}
              onClick={() => {
                location.hash = item;
              }}
              key={item}
            >
              {item[0].toUpperCase() + item.slice(1)}
            </button>
          ))}
        </nav>
        <span id="session-status" role="status">
          {busy
            ? "STREAMING"
            : sessionReady
              ? "READY"
              : "SESSION REQUIRED"}
        </span>
      </header>
      <main>
        <aside>
          <small>LOCAL AI WORKSPACE</small>
          <h1>{title}</h1>
          <p>One secure same-origin studio for local models.</p>
          {notice && <p role="alert">{notice}</p>}
        </aside>
        <section aria-label={route + " workspace"}>{renderWorkspace()}</section>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
