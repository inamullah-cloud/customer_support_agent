(() => {
  "use strict";

  // ---------------------------------------------------------------
  // Setup
  // ---------------------------------------------------------------
  // Same origin by default (FastAPI serves this page). Override with ?api=http://host:8000
  const API = (new URLSearchParams(location.search).get("api") || window.RELAY_API || "").replace(/\/$/, "");

  const store = {
    get(key) { try { return localStorage.getItem(key); } catch { return null; } },
    set(key, value) { try { localStorage.setItem(key, value); } catch { /* storage unavailable */ } },
    del(key) { try { localStorage.removeItem(key); } catch { /* storage unavailable */ } },
  };

  const $ = (id) => document.getElementById(id);
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  };

  const randomId = () => "user_" + Math.random().toString(36).slice(2, 8);

  const state = {
    userId: store.get("relay.user") || randomId(),
    threadId: store.get("relay.thread"),
    busy: false,
    awaiting: false,
    steps: [],
  };
  store.set("relay.user", state.userId);

  const NODE_LABELS = {
    classify: "Understood the request",
    memory_load_node: "Recalled memory",
    select_capabilities: "Chose tools",
    planning_node: "Made a plan",
    agent: "Agent step",
    approval_check: "Checked permissions",
    human_approval_node: "Waiting for approval",
    dynamic_tool_executor: "Ran tools",
    retry: "Retried",
    fallback: "Wrote a fallback reply",
    reflection_node: "Reviewed the answer",
    memory_write_node: "Updated memory",
    rejection_node: "Action rejected",
  };
  const HIDDEN_NODES = new Set(["observability_node"]);

  const SUGGESTIONS = [
    { text: "Where is my order DF1617?" },
    { text: "What is the price of the Wireless Headphones?" },
    { text: "How is the weather in Mingora?" },
    { text: "Please cancel my order DF1617", note: "Asks for your approval first" },
    { text: "I prefer SMS updates. Where is my order DF1617?", note: "The agent saves the preference to memory" },
  ];

  // ---------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------
  const titleCase = (snake) => {
    const text = String(snake || "").replace(/_/g, " ");
    return text.charAt(0).toUpperCase() + text.slice(1);
  };
  const fmtMs = (ms) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`);
  const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;

  let toastTimer;
  function toast(message) {
    const node = $("toast");
    node.textContent = message;
    node.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { node.hidden = true; }, 4000);
  }

  class ApiError extends Error {
    constructor(message, status) { super(message); this.status = status; }
  }

  async function request(path, options = {}) {
    let res;
    try {
      res = await fetch(API + path, {
        ...options,
        headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      });
    } catch {
      throw new ApiError(
        `Can't reach the API${API ? " at " + API : ""}. Start it with: uvicorn src.api.main:app`, 0);
    }
    if (!res.ok) {
      let detail = "";
      try {
        const body = await res.json();
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      } catch { /* non-JSON error body */ }
      throw new ApiError(detail || `Request failed (HTTP ${res.status}).`, res.status);
    }
    return res;
  }
  const getJSON = async (path) => (await request(path)).json();

  // Server-Sent Events over fetch (EventSource can't POST)
  async function streamSSE(path, body, onEvent) {
    const res = await request(path, { method: "POST", body: JSON.stringify(body) });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

      let end;
      while ((end = buffer.indexOf("\n\n")) !== -1) {
        const raw = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);

        let event = "message";
        const data = [];
        for (const line of raw.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data.push(line.slice(5).trim());
        }
        if (data.length) onEvent(event, JSON.parse(data.join("\n")));
      }
    }
  }

  // ---------------------------------------------------------------
  // Conversation log
  // ---------------------------------------------------------------
  const thread = $("thread");
  const log = $("log");
  let workingRow = null;

  const scrollToEnd = () => { log.scrollTop = log.scrollHeight; };
  const hideEmpty = () => { const e = $("empty"); if (e) e.hidden = true; };

  function addUser(text) {
    hideEmpty();
    thread.append(el("div", "msg user", text));
    scrollToEnd();
  }

  function chip(text, kind) {
    return el("span", "chip" + (kind ? " " + kind : ""), text);
  }

  function addChips(node, data) {
    const chips = el("div", "chips");
    if (data.category) chips.append(chip(titleCase(data.category)));
    (data.executed_tools || []).forEach((t) => chips.append(chip(t, "tool")));
    (data.memories_used || []).forEach((m) => chips.append(chip("Used memory: " + m, "memory")));
    (data.memories_saved || []).forEach((m) => chips.append(chip("Saved to memory: " + m, "memory")));
    if (data.fallback_used) chips.append(chip("Fallback reply", "warn"));
    if (data.observability) chips.append(chip(fmtMs(data.observability.total_duration_ms)));
    if (chips.childElementCount) node.append(chips);
  }

  function addAssistant(text, data) {
    const node = el("div", "msg assistant", text || "The agent did not produce an answer.");
    if (data) addChips(node, data);
    thread.append(node);
    scrollToEnd();
    return node;
  }

  function addError(message) {
    thread.append(el("div", "msg error", message));
    scrollToEnd();
  }

  function setWorking(label) {
    if (!workingRow) {
      workingRow = el("div", "msg working");
      thread.append(workingRow);
    }
    workingRow.textContent = label ? label + "…" : "Working…";
    scrollToEnd();
  }
  function clearWorking() {
    if (workingRow) workingRow.remove();
    workingRow = null;
  }

  // ---------------------------------------------------------------
  // Trail (the step-by-step view of one run)
  // ---------------------------------------------------------------
  function describe(node, update) {
    switch (node) {
      case "classify": return update.category ? titleCase(update.category) : "";
      case "memory_load_node": {
        const n = (update.memories_used || []).length;
        return n ? plural(n, "memory used", "memories used") : "Nothing relevant";
      }
      case "select_capabilities": return (update.selected_tools || []).join(", ");
      case "planning_node": return update.plan ? plural(update.plan.length, "step", "steps") : "";
      case "agent": return "response" in update ? "Wrote the answer" : "Asked to use a tool";
      case "approval_check": return update.approval_status === "pending" ? "Needs approval" : "No approval needed";
      case "dynamic_tool_executor":
        return update.error || (update.executed_tools || []).slice(-1)[0] || "";
      case "reflection_node": return update.reflection_status === "approved" ? "Approved" : "Needs correction";
      case "memory_write_node": {
        const n = (update.memories_saved || []).length;
        return n ? plural(n, "memory saved", "memories saved") : "Nothing new to save";
      }
      default: return "";
    }
  }

  function renderTrail() {
    const list = $("trail");
    list.replaceChildren();
    $("trailHint").hidden = state.steps.length > 0 || state.busy;

    for (const step of state.steps) {
      const item = el("li");
      item.dataset.status = step.status;
      item.append(el("span", "dot"));

      const row = el("div", "row");
      row.append(el("span", "label", step.label));
      if (step.ms != null) row.append(el("span", "ms", fmtMs(step.ms)));
      item.append(row);
      if (step.detail) item.append(el("div", "detail", step.detail));
      list.append(item);
    }

    if (state.busy) {
      const tail = el("li");
      tail.dataset.status = "active";
      const tailRow = el("div", "row");
      tailRow.append(el("span", "label", "Working"));
      tail.append(el("span", "dot"), tailRow);
      list.append(tail);
    }
  }

  function pushStep(node, update) {
    if (HIDDEN_NODES.has(node)) return;
    state.steps.push({
      node,
      label: NODE_LABELS[node] || titleCase(node),
      detail: describe(node, update),
      status: node === "dynamic_tool_executor" && update.error ? "failed" : "done",
      ms: null,
    });
    renderTrail();
  }

  function applyDurations(report) {
    if (!report) return;
    const lastIndex = {};
    state.steps.forEach((s, i) => { lastIndex[s.node] = i; });
    for (const [node, ms] of Object.entries(report.node_durations || {})) {
      if (lastIndex[node] != null) state.steps[lastIndex[node]].ms = ms;
    }
    renderTrail();
  }

  // ---------------------------------------------------------------
  // Running a turn (shared by "send" and "approve/reject")
  // ---------------------------------------------------------------
  function setBusyUi(busy) {
    $("send").disabled = busy || state.awaiting;
    $("input").disabled = state.awaiting;
  }

  function setBusy(busy) {
    state.busy = busy;
    $("send").disabled = busy || state.awaiting;
    $("input").disabled = state.awaiting;
    $("input").placeholder = state.awaiting
      ? "Approve or reject the action above to continue."
      : "Message the agent";
    $("app").setAttribute("aria-busy", String(busy));
    renderTrail();
  }

  async function runStream(path, body) {
    try {
      await runStreamInner(path, body);
    } catch (err) {
      // A bug in this page must never look like "the agent didn't answer".
      console.error(err);
      clearWorking();
      addError("Something went wrong in the page: " + (err && err.message ? err.message : err));
      state.busy = false;
      setBusyUi(false);
    }
  }

  async function runStreamInner(path, body) {
    setBusy(true);
    setWorking("Working");
    let final = null;
    let failure = null;

    try {
      await streamSSE(path, body, (event, data) => {
        if (event === "thread") {
          state.threadId = data.thread_id;
          store.set("relay.thread", data.thread_id);
        } else if (event === "node") {
          pushStep(data.node, data.update);
          const label = NODE_LABELS[data.node];
          if (label && !HIDDEN_NODES.has(data.node)) setWorking(label);
        } else if (event === "final") {
          final = data;
        } else if (event === "error") {
          failure = new ApiError(data.detail || "The agent failed to respond.", 500);
        }
      });
    } catch (err) {
      failure = err;
    }

    clearWorking();
    if (failure) {
      if (failure.status === 409) {
        await restoreThread(); // thread state changed under us: re-sync
      } else {
        addError(failure.message);
      }
    } else if (final) {
      handleFinal(final);
    } else {
      addError("The connection closed before the agent finished. Try again.");
    }
    setBusy(false);
    if (!$("viewHealth").hidden) loadHealth();
    $("input").focus();
  }

  function handleFinal(final) {
    state.awaiting = final.status === "awaiting_approval";

    if (state.awaiting) {
      const tools = (final.approval.pending_tool_calls || []).map((c) => c.name).join(", ");
      state.steps.push({
        node: "human_approval_node", label: NODE_LABELS.human_approval_node,
        detail: tools, status: "waiting", ms: null,
      });
      addApproval(final.approval);
    } else {
      applyDurations(final.observability);
      addAssistant(final.response, final);
      if ((final.memories_saved || []).length) {
        loadMemories(final.memories_saved);
        toast(`Saved ${plural(final.memories_saved.length, "memory", "memories")}.`);
      }
    }
  }

  async function send(text) {
    text = text.trim();
    if (!text || state.busy || state.awaiting) return;
    addUser(text);
    state.steps = [];
    await runStream("/api/chat/stream", {
      message: text, thread_id: state.threadId, user_id: state.userId,
    });
  }

  // ---------------------------------------------------------------
  // Approval slip
  // ---------------------------------------------------------------
  function addApproval(approval) {
    const slip = el("section", "slip");
    slip.setAttribute("aria-label", "Approval needed");
    slip.append(el("h3", null, "Approval needed"));
    slip.append(el("p", null, approval.message));

    for (const call of approval.pending_tool_calls || []) {
      const box = el("div", "call");
      box.append(el("b", null, call.name));
      const args = Object.entries(call.args || {});
      if (args.length) {
        const dl = el("dl");
        for (const [key, value] of args) {
          dl.append(el("dt", null, key), el("dd", null, String(value)));
        }
        box.append(dl);
      }
      slip.append(box);
    }

    const actions = el("div", "actions");
    const approve = el("button", "btn primary", "Approve");
    const reject = el("button", "btn", "Reject");
    approve.type = reject.type = "button";
    actions.append(approve, reject);
    slip.append(actions);

    const names = (approval.pending_tool_calls || []).map((c) => c.name).join(", ");
    const decide = async (approved) => {
      approve.disabled = reject.disabled = true;
      slip.className = "slip resolved";
      slip.replaceChildren(el("h3", null, `${approved ? "Approved" : "Rejected"}: ${names}`));

      const waiting = state.steps.find((s) => s.status === "waiting");
      if (waiting) { waiting.status = approved ? "done" : "failed"; waiting.detail = approved ? "Approved" : "Rejected"; }
      state.awaiting = false;
      await runStream(`/api/chat/${encodeURIComponent(state.threadId)}/approval/stream`, { approved });
    };
    approve.addEventListener("click", () => decide(true));
    reject.addEventListener("click", () => decide(false));

    thread.append(slip);
    scrollToEnd();
    renderTrail();
    setBusy(false);
    approve.focus();
  }

  // ---------------------------------------------------------------
  // Restore / reset the conversation
  // ---------------------------------------------------------------
  function resetConversation() {
    thread.querySelectorAll(".msg, .slip").forEach((n) => n.remove());
    clearWorking();
    state.steps = [];
    state.awaiting = false;
    const empty = $("empty");
    if (empty) empty.hidden = false;
    setBusy(false);
  }

  async function restoreThread() {
    resetConversation();
    if (!state.threadId) return;

    let data;
    try {
      data = await getJSON(`/api/chat/${encodeURIComponent(state.threadId)}`);
    } catch (err) {
      if (err.status === 404) {
        store.del("relay.thread");
        state.threadId = null;
      } else if (err.status === 0) {
        hideEmpty();
        addError(err.message);
      }
      return;
    }

    hideEmpty();
    let lastAssistant = null;
    for (const m of data.messages) {
      if (m.role === "user") addUser(m.content);
      else if (m.role === "assistant" && m.content) lastAssistant = addAssistant(m.content);
    }

    if (data.status === "awaiting_approval" && data.approval) {
      state.awaiting = true;
      state.steps.push({
        node: "human_approval_node", label: NODE_LABELS.human_approval_node,
        detail: (data.approval.pending_tool_calls || []).map((c) => c.name).join(", "),
        status: "waiting", ms: null,
      });
      addApproval(data.approval);
    } else if (lastAssistant) {
      addChips(lastAssistant, data);
    }
    setBusy(false);
  }

  function newChat() {
    if (state.busy) return;
    store.del("relay.thread");
    state.threadId = null;
    resetConversation();
    $("input").focus();
  }

  // ---------------------------------------------------------------
  // Memory panel
  // ---------------------------------------------------------------
  async function loadMemories(highlight = []) {
    const list = $("memoryList");
    let memories;
    try {
      memories = await getJSON(`/api/users/${encodeURIComponent(state.userId)}/memories`);
    } catch (err) {
      list.replaceChildren(el("li", "memory-empty", err.message));
      return;
    }

    if (!memories.length) {
      list.replaceChildren(el("li", "memory-empty",
        "Nothing remembered yet. Tell the agent a preference, such as “I prefer SMS updates”, or add one above."));
      return;
    }

    list.replaceChildren(...memories.map((m) => {
      const item = el("li");
      if (highlight.includes(m.content)) item.className = "fresh";
      item.append(el("span", "kind", titleCase(m.memory_type)));
      item.append(el("span", "text", m.content));
      const remove = el("button", "remove", "Remove");
      remove.type = "button";
      remove.setAttribute("aria-label", `Remove memory: ${m.content}`);
      remove.addEventListener("click", async () => {
        try {
          await request(`/api/users/${encodeURIComponent(state.userId)}/memories/${m.memory_id}`, { method: "DELETE" });
          loadMemories();
        } catch (err) { toast(err.message); }
      });
      item.append(remove);
      return item;
    }));
  }

  async function addMemory(event) {
    event.preventDefault();
    const content = $("memoryContent").value.trim();
    if (!content) return;
    try {
      await request(`/api/users/${encodeURIComponent(state.userId)}/memories`, {
        method: "POST",
        body: JSON.stringify({ memory_type: $("memoryType").value, content }),
      });
      $("memoryContent").value = "";
      loadMemories([content]);
    } catch (err) {
      toast(err.status === 409 ? "You already have that memory." : err.message);
    }
  }

  function changeUser() {
    const value = $("userId").value.trim();
    if (!value || value === state.userId) { $("userId").value = state.userId; return; }
    state.userId = value;
    store.set("relay.user", value);
    loadMemories();
    toast("Switched memory to " + value);
  }

  // ---------------------------------------------------------------
  // Health tab
  // ---------------------------------------------------------------
  function stat(label, valueNode) {
    const box = el("div", "stat");
    const dd = el("dd");
    dd.append(valueNode);
    box.append(el("dt", null, label), dd);
    return box;
  }
  const statusBadge = (status) => {
    const b = el("span", "status", titleCase(status));
    b.dataset.status = status;
    return b;
  };

  async function loadHealth() {
    const box = $("healthLive");
    try {
      const d = await getJSON("/api/observability/dashboard");
      box.replaceChildren(
        stat("Status", statusBadge(d.health.status)),
        stat("Chats measured", document.createTextNode(String(d.summary.total_runs))),
        stat("Average time", document.createTextNode(fmtMs(d.summary.average_execution_time_ms))),
        stat("Slowest step", document.createTextNode(d.health.slowest_node ? titleCase(d.health.slowest_node) : "None slow")),
      );
    } catch (err) {
      box.replaceChildren(el("p", "hint", err.message));
    }
  }

  async function runEvaluation() {
    const button = $("runEval");
    const box = $("healthEval");
    button.disabled = true;
    box.replaceChildren(el("p", "hint", "Running the test cases…"));

    try {
      const d = await (await request("/api/evaluation/run", { method: "POST" })).json();
      const frag = document.createDocumentFragment();

      const stats = el("div", "stats");
      stats.append(
        stat("Status", statusBadge(d.dashboard.health.status)),
        stat("Passed", document.createTextNode(`${d.metrics.overall_success_rate.toFixed(0)}%`)),
        stat("Average time", document.createTextNode(fmtMs(d.average_latency_ms))),
        stat("Cases", document.createTextNode(String(d.metrics.total_cases))),
      );
      frag.append(stats);

      const cases = el("ul");
      for (const r of d.results) {
        const ok = r.evaluation.overall_success;
        const li = el("li");
        li.append(el("span", ok ? "pass" : "fail", ok ? "Passed" : "Failed"));
        li.append(el("span", null, ok ? r.evaluation.query : `${r.evaluation.query} (${r.evaluation.failure_reason.replace(/_/g, " ")})`));
        cases.append(li);
      }
      frag.append(cases);

      if (d.dashboard.insights.length) {
        const insights = el("ul", "insights");
        d.dashboard.insights.forEach((t) => insights.append(el("li", null, t)));
        frag.append(insights);
      }
      box.replaceChildren(frag);
    } catch (err) {
      box.replaceChildren(el("p", "hint", err.message));
    } finally {
      button.disabled = false;
    }
  }

  // ---------------------------------------------------------------
  // Tabs, panels, composer wiring
  // ---------------------------------------------------------------
  function selectTab(name) {
    const isHealth = name === "health";
    $("tabTrail").setAttribute("aria-selected", String(!isHealth));
    $("tabHealth").setAttribute("aria-selected", String(isHealth));
    $("tabTrail").tabIndex = isHealth ? -1 : 0;
    $("tabHealth").tabIndex = isHealth ? 0 : -1;
    $("viewTrail").hidden = isHealth;
    $("viewHealth").hidden = !isHealth;
    if (isHealth) loadHealth();
  }

  function togglePanel(name) {
    const app = $("app");
    app.dataset.panel = app.dataset.panel === name ? "none" : name;
    $("toggleMemory").setAttribute("aria-expanded", String(app.dataset.panel === "memory"));
    $("toggleRun").setAttribute("aria-expanded", String(app.dataset.panel === "run"));
  }

  function autosize() {
    const input = $("input");
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 150) + "px";
  }

  function init() {
    $("userId").value = state.userId;
    $("userId").addEventListener("change", changeUser);
    $("memoryForm").addEventListener("submit", addMemory);

    const suggestions = $("suggestions");
    for (const s of SUGGESTIONS) {
      const btn = el("button", "suggestion", s.text);
      btn.type = "button";
      if (s.note) btn.append(el("small", null, s.note));
      btn.addEventListener("click", () => send(s.text));
      suggestions.append(btn);
    }

    $("composer").addEventListener("submit", (e) => {
      e.preventDefault();
      const input = $("input");
      const text = input.value;
      input.value = "";
      autosize();
      send(text);
    });
    $("input").addEventListener("input", autosize);
    $("input").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        $("composer").requestSubmit();
      }
    });

    $("newChat").addEventListener("click", newChat);
    $("toggleMemory").addEventListener("click", () => togglePanel("memory"));
    $("toggleRun").addEventListener("click", () => togglePanel("run"));
    $("tabTrail").addEventListener("click", () => selectTab("trail"));
    $("tabHealth").addEventListener("click", () => selectTab("health"));
    $("runEval").addEventListener("click", runEvaluation);

    renderTrail();
    loadMemories();
    restoreThread();
  }

  init();
})();