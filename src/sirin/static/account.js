import { api, el } from "./lib.js";

const status = () => el("span", { class: "result" });

function showResult(node, promise, done) {
  node.className = "result"; node.textContent = "…";
  return promise
    .then((r) => { node.className = "result ok"; node.textContent = done; return r; })
    .catch((e) => { node.className = "result err"; node.textContent = e.message; throw e; });
}

// Name, password change and logout of whoever is logged in.
export function accountBar(me) {
  const result = status();
  const current = el("input", { type: "password", placeholder: "current password", autocomplete: "current-password", required: "" });
  const fresh = el("input", { type: "password", placeholder: "new password", autocomplete: "new-password", minlength: "8", required: "" });
  const form = el("form", {}, current, fresh, el("button", {}, "Change"), result);
  form.hidden = true;
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    showResult(result, api("/api/auth/password", { current: current.value, new: fresh.value }), "changed")
      .then(() => { current.value = fresh.value = ""; }, () => {});
  });
  const logout = async () => {
    await api("/api/auth/logout", {});
    location.href = "/login.html";
  };
  return el("div", { class: "account" },
    el("span", {}, `${me.name} (${me.login}, ${me.role})`),
    el("button", { type: "button", onclick: () => { form.hidden = !form.hidden; } }, "Password"),
    el("button", { type: "button", onclick: logout }, "Log out"),
    form,
  );
}

const roleSelect = (value) => {
  const select = el("select", {}, ...["member", "admin"].map(r => el("option", { value: r }, r)));
  select.value = value;
  return select;
};

function personItem(person, me, reload) {
  const name = el("input", { value: person.name, required: "" });
  const role = roleSelect(person.role);
  const password = el("input", { type: "password", placeholder: "new password (optional)", autocomplete: "new-password", minlength: "8" });
  const result = status();
  const form = el("form", {}, name, role, password, el("button", {}, "Save"), result);
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const change = { name: name.value, role: role.value, ...(password.value ? { password: password.value } : {}) };
    showResult(result, api(`/api/users/${person.id}`, change, "PATCH"), "saved").then(() => { password.value = ""; }, () => {});
  });
  const remove = el("button", { type: "button", class: "remove" }, "delete");
  remove.addEventListener("click", (e) => {
    e.preventDefault();
    showResult(result, api(`/api/users/${person.id}`, undefined, "DELETE"), "deleted")
      .then(() => (person.id === me.id ? (location.href = "/login.html") : reload()), () => {});
  });
  const you = person.id === me.id ? " (you)" : "";
  return el("details", { class: "item person" },
    el("summary", {}, el("span", { class: "name" }, `${person.login}${you}`), el("span", { class: "hint" }, `${person.name}, ${person.role}`), remove),
    el("div", { class: "body" }, form));
}

// Accounts of the people who log in; the admin manages them here.
export function peopleCard(me) {
  const list = el("div");
  const reload = async () => {
    const people = await api("/api/users");
    list.replaceChildren(...people.map(p => personItem(p, me, reload)));
  };
  const login = el("input", { placeholder: "login", pattern: "[A-Za-z0-9._\\-]+", required: "" });
  const name = el("input", { placeholder: "name the assistant uses", required: "" });
  const password = el("input", { type: "password", placeholder: "password", autocomplete: "new-password", minlength: "8", required: "" });
  const role = roleSelect("member");
  const result = status();
  const add = el("form", {}, login, name, password, role, el("button", {}, "Add"), result);
  add.addEventListener("submit", (e) => {
    e.preventDefault();
    showResult(result, api("/api/users", { login: login.value, name: name.value, password: password.value, role: role.value }), "added")
      .then(() => { add.reset(); return reload(); }, () => {});
  });
  reload();
  return el("section", { class: "card" },
    el("h2", {}, "People"),
    el("p", { class: "hint" }, "Accounts for this page. A member can log in and change their own password; only an admin edits settings."),
    list, add);
}
