import { api, el } from "./lib.js";

const PAGE = 50;

const time = (iso) => new Date(iso).toLocaleString([], { dateStyle: "short", timeStyle: "medium" });
const seconds = (ms) => `${(ms / 1000).toFixed(1)} s`;

function turnBody(turn) {
  const rows = [];
  const line = (label, ...value) => rows.push(el("div", { class: "field" }, el("label", {}, label), el("div", {}, ...value)));
  line("heard", turn.text || el("span", { class: "hint" }, "silence"));
  line("answer", turn.answer ?? el("span", { class: "hint" }, "answering…"));
  for (const call of turn.tools) {
    const result = call.result == null ? "…" : call.result;
    line("tool", el("code", {}, `${call.name}(${JSON.stringify(call.args)})`), " → ",
      el("span", { class: call.failed ? "err" : "" }, result));
  }
  if (turn.error) line("error", el("span", { class: "err" }, turn.error));
  const stages = Object.entries(turn.timings).sort((a, b) => a[1] - b[1]).map(([k, v]) => `${k} ${seconds(v)}`);
  if (stages.length) line("timings", stages.join(" · "));
  const voice = [];
  if (turn.speech_ms != null) voice.push(`${seconds(turn.speech_ms)} of speech`);
  if (turn.similarity != null) voice.push(`similarity ${turn.similarity.toFixed(2)} with the closest voice`);
  if (voice.length) line("voice", voice.join(", "));
  line("dialog", el("code", {}, turn.conversation_id));
  return rows;
}

function turnItem(turn, open) {
  const mark = turn.error ? el("span", { class: "badge err", title: turn.error }, "✗")
    : turn.answer == null ? el("span", { class: "badge warn", title: "answering" }, "…") : null;
  const box = el("details", { class: "item turn" },
    el("summary", {},
      el("span", { class: "hint" }, `${time(turn.started_at)} · ${turn.device}`),
      el("span", { class: "name" }, turn.text || "(silence)"),
      mark),
    el("div", { class: "body" }, ...turnBody(turn)));
  box.open = open;
  box.dataset.id = turn.id;
  return box;
}

// The history as a read-only feed, newest first; new turns and their answers arrive over server-sent events.
export function historyView() {
  const device = el("select", {}, el("option", { value: "" }, "all devices"));
  const from = el("input", { type: "date", title: "from this day" });
  const to = el("input", { type: "date", title: "to this day" });
  const list = el("div");
  const more = el("button", { type: "button" }, "Older");
  const msg = el("span", { class: "hint" });
  let oldest = null;

  const query = () => {
    const q = new URLSearchParams({ limit: PAGE });
    if (device.value) q.set("device", device.value);
    if (from.value) q.set("since", new Date(`${from.value}T00:00`).toISOString());
    if (to.value) {
      const end = new Date(`${to.value}T00:00`);
      end.setDate(end.getDate() + 1);
      q.set("until", end.toISOString());
    }
    return q;
  };

  const matches = (turn) => {
    const q = query(), at = new Date(turn.started_at);
    return (!q.has("device") || q.get("device") === turn.device)
      && (!q.has("since") || at >= new Date(q.get("since")))
      && (!q.has("until") || at < new Date(q.get("until")));
  };

  const load = async (reset) => {
    const q = query();
    if (!reset && oldest != null) q.set("before_id", oldest);
    const turns = await api(`/api/history?${q}`);
    if (reset) list.replaceChildren();
    list.append(...turns.map(t => turnItem(t, false)));
    if (turns.length) oldest = turns[turns.length - 1].id;
    more.hidden = turns.length < PAGE;
    msg.textContent = list.children.length ? "" : "Nothing here yet.";
  };

  const refreshDevices = async () => {
    const chosen = device.value;
    device.replaceChildren(el("option", { value: "" }, "all devices"), ...(await api("/api/history/devices")).map(d => el("option", { value: d }, d)));
    device.value = chosen;
  };

  // An open turn stays open when its answer arrives.
  const receive = (turn) => {
    const shown = list.querySelector(`[data-id="${turn.id}"]`);
    if (shown) {
      shown.replaceWith(turnItem(turn, shown.open));
    } else if (matches(turn) && (!list.firstElementChild || turn.id > Number(list.firstElementChild.dataset.id))) {
      list.prepend(turnItem(turn, false));
      msg.textContent = "";
      if (![...device.options].some(o => o.value === turn.device)) refreshDevices();
    }
  };

  for (const input of [device, from, to]) input.addEventListener("change", () => { oldest = null; load(true); });
  more.addEventListener("click", () => load(false));
  refreshDevices();
  load(true);
  new EventSource("/api/history/stream").onmessage = (e) => receive(JSON.parse(e.data));

  return el("section", { class: "card" },
    el("h2", {}, "History"),
    el("p", { class: "hint" }, "What was said to the devices and what they answered. Read-only."),
    el("div", { class: "filters" }, device, from, to),
    list, msg, more);
}
