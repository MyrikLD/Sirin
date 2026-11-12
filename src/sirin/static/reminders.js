import { api, el } from "./lib.js";

// weekday cannot be combined with dateStyle, so the parts are named one by one.
const when = (iso) => new Date(iso).toLocaleString([], { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

function reminderRow(reminder, me, reload) {
  const result = el("span", { class: "result" });
  const repeat = reminder.schedule
    ? `repeats ${reminder.schedule}${reminder.until ? ` until ${when(reminder.until)}` : ""}`
    : "once";
  const canCancel = me.role === "admin" || reminder.person_id === me.id;
  const cancel = canCancel ? el("button", { type: "button", class: "remove" }, "Cancel") : null;
  cancel?.addEventListener("click", async () => {
    try { await api(`/api/reminders/${reminder.id}`, undefined, "DELETE"); reload(); } catch (e) { result.className = "result err"; result.textContent = e.message; }
  });
  return el("div", { class: "item reminder" },
    el("div", { class: "when" }, el("strong", {}, when(reminder.due)), el("span", { class: "hint" }, ` · ${repeat} · set by ${reminder.person_name || "an unknown voice"}`)),
    el("div", { class: "what" }, el("span", {}, reminder.prompt), cancel, result));
}

// Pending reminders, soonest first; they are set by voice, here they can only be looked at and cancelled.
export function remindersView(me) {
  const list = el("div");
  const msg = el("p", { class: "hint" });
  const reload = async () => {
    const reminders = await api("/api/reminders");
    list.replaceChildren(...reminders.map(r => reminderRow(r, me, reload)));
    msg.textContent = reminders.length ? "" : "No reminders pending. Ask the assistant, for example: remind me in 10 minutes to turn off the stove.";
  };
  reload();
  // A recurring reminder moves to its next time and a one-off one disappears once announced.
  setInterval(() => { if (!document.getElementById("reminders-view").hidden) reload(); }, 30000);
  return el("section", { class: "card" },
    el("h2", {}, "Reminders"),
    el("p", { class: "hint" }, "What the assistant will announce. The text is the note it wrote to itself; at the time it carries it out."),
    list, msg);
}
