import { api, el } from "./lib.js";

const PAGE = 50;

const time = (iso) => new Date(iso).toLocaleString([], { dateStyle: "short", timeStyle: "medium" });
const seconds = (ms) => `${(ms / 1000).toFixed(1)} s`;

export const voiceName = (v) => v.label || `voice ${v.id}`;

function voiceLine(turn, me, reload) {
  if (turn.voice_id == null) return [el("span", { class: "hint" }, "not recognized")];
  const parts = [voiceName({ id: turn.voice_id, label: turn.voice_label })];
  parts.push(turn.person_name ? ` · ${turn.person_name}` : " · nobody's");
  if (turn.voice_confirmed) parts.push(" · confirmed");
  const result = el("span", { class: "result" });
  const act = (path, method) => async () => {
    try { await api(path, undefined, method); reload(); } catch (e) { result.className = "result err"; result.textContent = e.message; }
  };
  const mine = turn.person_id === me.id;
  if ((turn.person_id == null || mine) && !turn.voice_confirmed) {
    parts.push(" ", el("button", { type: "button", onclick: act(`/api/history/${turn.id}/mine`, "POST") }, "This is me"));
  }
  if (mine || me.role === "admin") {
    parts.push(" ", el("button", { type: "button", onclick: act(`/api/history/${turn.id}/voice`, "DELETE") }, "Not this voice"));
  }
  return [...parts, " ", result];
}

function turnBody(turn, me, reload) {
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
  line("voice", ...voiceLine(turn, me, reload));
  if (voice.length) line("speech", voice.join(", "));
  line("dialog", el("code", {}, turn.conversation_id));
  return rows;
}

function turnItem(turn, open, me, reload) {
  const mark = turn.error ? el("span", { class: "badge err", title: turn.error }, "✗")
    : turn.answer == null ? el("span", { class: "badge warn", title: "answering" }, "…") : null;
  const box = el("details", { class: "item turn" },
    el("summary", {},
      el("span", { class: "hint" }, `${time(turn.started_at)} · ${turn.device}${turn.person_name ? ` · ${turn.person_name}` : ""}`),
      el("span", { class: "name" }, turn.text || "(silence)"),
      mark),
    el("div", { class: "body" }, ...turnBody(turn, me, reload)));
  box.open = open;
  box.dataset.id = turn.id;
  return box;
}

// The history as a read-only feed, newest first; new turns and their answers arrive over server-sent events.
export function historyView(me) {
  const device = el("select", {}, el("option", { value: "" }, "all devices"));
  const person = el("select", {}, el("option", { value: "" }, "everyone"));
  const from = el("input", { type: "date", title: "from this day" });
  const to = el("input", { type: "date", title: "to this day" });
  const list = el("div");
  const more = el("button", { type: "button" }, "Older");
  const msg = el("span", { class: "hint" });
  let oldest = null;

  const query = () => {
    const q = new URLSearchParams({ limit: PAGE });
    if (device.value) q.set("device", device.value);
    if (person.value) q.set("person_id", person.value);
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
      && (!q.has("person_id") || Number(q.get("person_id")) === turn.person_id)
      && (!q.has("since") || at >= new Date(q.get("since")))
      && (!q.has("until") || at < new Date(q.get("until")));
  };

  const load = async (reset) => {
    const q = query();
    if (!reset && oldest != null) q.set("before_id", oldest);
    const turns = await api(`/api/history?${q}`);
    const open = new Set([...list.querySelectorAll("details[open]")].map(d => d.dataset.id));
    if (reset) list.replaceChildren();
    list.append(...turns.map(t => turnItem(t, open.has(String(t.id)), me, reloadAll)));
    if (turns.length) oldest = turns[turns.length - 1].id;
    more.hidden = turns.length < PAGE;
    msg.textContent = list.children.length ? "" : "Nothing here yet.";
  };

  // After a voice changed hands every shown turn of it may read differently.
  const reloadAll = () => { oldest = null; return load(true); };

  const refreshDevices = async () => {
    const chosen = device.value;
    device.replaceChildren(el("option", { value: "" }, "all devices"), ...(await api("/api/history/devices")).map(d => el("option", { value: d }, d)));
    device.value = chosen;
  };

  // An open turn stays open when its answer arrives.
  const receive = (turn) => {
    const shown = list.querySelector(`[data-id="${turn.id}"]`);
    if (shown) {
      shown.replaceWith(turnItem(turn, shown.open, me, reloadAll));
    } else if (matches(turn) && (!list.firstElementChild || turn.id > Number(list.firstElementChild.dataset.id))) {
      list.prepend(turnItem(turn, false, me, reloadAll));
      msg.textContent = "";
      if (![...device.options].some(o => o.value === turn.device)) refreshDevices();
    }
  };

  for (const input of [device, person, from, to]) input.addEventListener("change", reloadAll);
  const people = me.role === "admin" ? api("/api/users") : Promise.resolve([me]);
  people.then(all => person.append(...all.map(p => el("option", { value: p.id }, p.id === me.id ? `${p.name} (me)` : p.name))));
  more.addEventListener("click", () => load(false));
  refreshDevices();
  load(true);
  new EventSource("/api/history/stream").onmessage = (e) => receive(JSON.parse(e.data));

  return el("section", { class: "card" },
    el("h2", {}, "History"),
    el("p", { class: "hint" }, "What was said to the devices and what they answered. Read-only."),
    el("div", { class: "filters" }, device, person, from, to),
    list, msg, more);
}
