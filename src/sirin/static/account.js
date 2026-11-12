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
  const fresh = el("input", { type: "password", placeholder: "new password", autocomplete: "new-password", required: "" });
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

const field = (label, input, hint) =>
  el("div", { class: "field" }, el("label", {}, label), input, hint ? el("p", { class: "hint" }, hint) : null);

// The fields of a person; values() gives what was filled in, without an empty password.
function personForm(person, submit, extra = []) {
  const login = person ? null : el("input", { pattern: "[A-Za-z0-9._\\-]+", required: "", autocomplete: "off" });
  const name = el("input", { value: person?.name || "", required: "" });
  const role = roleSelect(person?.role || "member");
  const password = el("input", { type: "password", autocomplete: "new-password", ...(person ? {} : { required: "" }) });
  const result = el("span", { class: "result" });
  const form = el("form", {},
    login && field("login", login, "Letters, digits, . _ -"),
    field("name", name, "What the assistant calls this person"),
    field("role", role, "An admin edits settings and people; a member sees their history, voices and tools"),
    field(person ? "new password" : "password", password, person ? "Empty keeps the password; a new one logs the person out everywhere" : null),
    el("div", { class: "probe" }, el("button", { class: "primary" }, person ? "Save" : "Add"), ...extra, result));
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const values = { name: name.value, role: role.value, ...(login ? { login: login.value } : {}), ...(password.value ? { password: password.value } : {}) };
    result.className = "result"; result.textContent = "…";
    try {
      await submit(values);
      password.value = "";
      result.className = "result ok"; result.textContent = "saved";
    } catch (err) { result.className = "result err"; result.textContent = err.message; }
  });
  return { form, result };
}

function personItem(person, me, reload) {
  const remove = el("button", { type: "button", class: "remove" }, "Delete");
  const { form, result } = personForm(person, async (values) => {
    await api(`/api/users/${person.id}`, values, "PATCH");
    await reload();
  }, [remove]);
  remove.addEventListener("click", async () => {
    try {
      await api(`/api/users/${person.id}`, undefined, "DELETE");
      if (person.id === me.id) location.href = "/login.html"; else await reload();
    } catch (err) { result.className = "result err"; result.textContent = err.message; }
  });
  return el("details", { class: "item" },
    el("summary", {},
      el("span", { class: "name text" }, person.name),
      el("span", { class: "hint" }, `${person.login}${person.id === me.id ? " · you" : ""}`),
      el("span", { class: `badge ${person.role === "admin" ? "warn" : "hint"}` }, person.role)),
    el("div", { class: "body" }, form));
}

// Accounts of the people who log in; the admin manages them here.
export function peopleView(me) {
  const list = el("div");
  const reload = async () => {
    const people = await api("/api/users");
    list.replaceChildren(...people.map(p => personItem(p, me, reload)));
  };
  const adding = el("details", { class: "item" }, el("summary", {}, el("span", { class: "name text" }, "+ add a person")));
  const { form } = personForm(null, async (values) => {
    await api("/api/users", values);
    form.reset();
    adding.open = false;
    await reload();
  });
  adding.append(el("div", { class: "body" }, form));
  reload();
  return el("section", { class: "card" },
    el("h2", {}, "People"),
    el("p", { class: "hint" }, "Who can log in to this page. Link a person's voice on the Voices tab or with This is me in History."),
    list, adding);
}
