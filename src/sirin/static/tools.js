import { api, el } from "./lib.js";

const MARKS = { ok: ["✓", "ok"], pending: ["…", "warn"], error: ["✗", "err"] };

function badge(status) {
  const [mark, cls] = MARKS[status.state];
  return el("span", { class: `badge ${cls}`, title: status.detail || status.state }, mark);
}

// Headers as "Name: value" lines, which is how people copy them.
const headersText = (headers) => Object.entries(headers || {}).map(([k, v]) => `${k}: ${v}`).join("\n");
const parseHeaders = (text) => Object.fromEntries(text.split("\n").map(l => l.trim()).filter(Boolean).map(l => {
  const at = l.indexOf(":");
  return at < 0 ? [l, ""] : [l.slice(0, at).trim(), l.slice(at + 1).trim()];
}));

function serverItem(server, reload) {
  const name = el("input", { value: server?.name || "", placeholder: "name, the prefix of its tools", required: "" });
  const url = el("input", { value: server?.url || "", placeholder: "https://…/mcp", required: "" });
  const headers = el("textarea", { placeholder: "Authorization: Bearer …" });
  headers.value = headersText(server?.headers);
  const oauth = el("input", { type: "checkbox" });
  oauth.checked = server?.oauth || false;
  const result = el("span", { class: "result hint" });
  const show = (ok, text) => { result.className = `result ${ok ? "ok" : "err"}`; result.textContent = text; };
  const values = () => ({ name: name.value.trim(), url: url.value.trim(), headers: parseHeaders(headers.value), oauth: oauth.checked });

  const form = el("form", { class: "server" },
    el("div", { class: "field" }, el("label", {}, "name"), name),
    el("div", { class: "field" }, el("label", {}, "url"), url),
    el("div", { class: "field" }, el("label", {}, "headers"), headers),
    el("div", { class: "field" }, el("label", {}, "oauth"), oauth));
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      if (server) await api(`/api/my/mcp-servers/${server.id}`, values(), "PUT");
      else await api("/api/my/mcp-servers", values());
      show(true, "saved; connecting…");
      setTimeout(reload, 1500);
    } catch (err) { show(false, err.message); }
  });

  const test = el("button", { type: "button" }, "Test");
  test.addEventListener("click", async () => {
    result.className = "result"; result.textContent = "testing…";
    try { const r = await api("/api/my/mcp-servers/probe", values()); show(r.ok, r.message); } catch (err) { show(false, err.message); }
  });
  const buttons = [el("button", { class: "primary" }, server ? "Save" : "Add"), test];
  if (server?.oauth) {
    const authorize = el("button", { type: "button" }, "Authorize");
    authorize.addEventListener("click", async () => {
      const tab = window.open("about:blank", "_blank");
      try {
        const r = await api(`/api/my/mcp-servers/${server.id}/authorize`, {});
        if (r.url) { tab.location = r.url; show(true, "Finish the login in the opened tab."); } else { tab?.close(); show(true, "Already authorized."); }
      } catch (err) { tab?.close(); show(false, err.message); }
    });
    buttons.push(authorize);
  }
  if (server) {
    buttons.push(el("button", { type: "button", class: "remove", onclick: async () => {
      try { await api(`/api/my/mcp-servers/${server.id}`, undefined, "DELETE"); reload(); } catch (err) { show(false, err.message); }
    } }, "delete"));
  }
  form.append(el("div", { class: "probe" }, ...buttons, result));
  if (!server) return form;
  return el("details", { class: "item" },
    el("summary", {}, el("span", { class: "name" }, server.name), badge(server.status)),
    el("div", { class: "body" }, form));
}

function overview() {
  const list = el("div");
  api("/api/mcp-servers/personal").then((servers) => {
    list.replaceChildren(...(servers.length ? servers.map(s => el("div", { class: "row" },
      el("span", { class: "name" }, `${s.owner}: ${s.name}`), badge(s.status))) : [el("p", { class: "hint" }, "Nobody has a personal server.")]));
  });
  return el("section", { class: "card" }, el("h2", {}, "Everyone's personal servers"),
    el("p", { class: "hint" }, "Names and state only; their addresses and headers stay with their owners."), list);
}

// MCP servers of the logged-in person: their tools are offered only when the assistant recognizes that person's voice.
export function toolsView(me) {
  const list = el("div");
  const reload = async () => {
    const servers = await api("/api/my/mcp-servers");
    list.replaceChildren(...servers.map(s => serverItem(s, reload)));
  };
  reload();
  const mine = el("section", { class: "card" },
    el("h2", {}, "My tools"),
    el("p", { class: "hint" }, "MCP servers only you get: the assistant offers their tools when it recognizes your voice. "
      + "A voice match is not access control: someone with a similar voice, or a recording of yours, can use them too."),
    list,
    el("details", { class: "item" }, el("summary", {}, el("span", { class: "name text" }, "+ add a server")),
      el("div", { class: "body" }, serverItem(null, reload))));
  return me.role === "admin" ? el("div", {}, mine, overview()) : mine;
}
