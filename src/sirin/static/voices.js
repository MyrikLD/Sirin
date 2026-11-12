import { api, el } from "./lib.js";
import { voiceName } from "./history.js";

const day = (iso) => (iso ? new Date(iso).toLocaleDateString() : "never");

function voiceItem(voice, people, me, reload) {
  const result = el("span", { class: "result" });
  const run = (promise, done) => {
    result.className = "result"; result.textContent = "…";
    promise.then(() => { result.className = "result ok"; result.textContent = done; return reload(); })
      .catch((e) => { result.className = "result err"; result.textContent = e.message; });
  };

  const controls = [];
  if (me.role === "admin") {
    const person = el("select", {}, el("option", { value: "" }, "nobody"), ...people.map(p => el("option", { value: p.id }, p.name)));
    person.value = voice.person_id ?? "";
    person.addEventListener("change", () => run(
      api(`/api/voices/${voice.id}`, person.value ? { person_id: Number(person.value) } : { unlink: true }, "PATCH"), "saved"));
    controls.push(el("form", {}, el("span", { class: "hint" }, "belongs to"), person));
  } else {
    controls.push(el("form", {}, el("button", { type: "button", onclick: () => run(api(`/api/voices/${voice.id}`, { unlink: true }, "PATCH"), "unlinked") }, "Not my voice")));
  }

  const remove = el("button", { type: "button", class: "remove" }, "delete");
  remove.addEventListener("click", (e) => {
    e.preventDefault();
    run(api(`/api/voices/${voice.id}`, undefined, "DELETE"), "deleted");
  });
  const facts = `${voice.turns} turn(s), last ${day(voice.last_heard)}${voice.isolates ? "" : ", no voiceprint yet"}`;
  return el("details", { class: "item person" },
    el("summary", {}, el("span", { class: "name" }, voiceName(voice.id)),
      el("span", { class: "hint" }, `${voice.person_name || "nobody"} · ${facts}`), remove),
    el("div", { class: "body" }, ...controls, result));
}

// Voices heard by the devices. A member sees and manages their own; the admin sees all of them.
export function voicesView(me) {
  const list = el("div");
  const msg = el("p", { class: "hint" });
  const reload = async () => {
    const [voices, people] = await Promise.all([api("/api/voices"), me.role === "admin" ? api("/api/users") : []]);
    list.replaceChildren(...voices.map(v => voiceItem(v, people, me, reload)));
    msg.textContent = voices.length ? "" : me.role === "admin"
      ? "No voices yet: one appears for every speaker the devices hear."
      : "No voice is yours yet: open a turn you said in History and press This is me.";
  };
  reload();
  return el("section", { class: "card" },
    el("h2", {}, "Voices"),
    el("p", { class: "hint" }, "Every new speaker becomes a voice. Once a voice belongs to a person, the assistant knows who speaks and, as soon as a turn of 3 seconds or more gave it a voiceprint, mutes other speech. A person can have several voices; each is recognized as them."),
    list, msg);
}
