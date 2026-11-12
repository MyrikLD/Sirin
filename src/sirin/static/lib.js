export const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v; else if (k.startsWith("on")) node.addEventListener(k.slice(2), v); else node.setAttribute(k, v);
  }
  node.append(...children.filter(c => c != null));
  return node;
};

// GET without a body or a method; a body goes as JSON, by POST unless told otherwise.
// A 401 outside the login page means the session is gone: the browser goes to log in and the call never settles.
export async function api(path, body, method) {
  const opts = body === undefined
    ? { method: method || "GET" }
    : { method: method || "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opts);
  if (r.status === 401 && location.pathname !== "/login.html") {
    location.href = "/login.html?next=" + encodeURIComponent(location.pathname);
    return new Promise(() => {});
  }
  const answer = await r.json().catch(() => null);
  if (!r.ok) {
    const detail = Array.isArray(answer?.detail) ? answer.detail.map(d => `${d.loc.slice(1).join(".") || "value"}: ${d.msg}`).join("\n") : answer?.detail || r.statusText;
    throw new Error(detail);
  }
  return answer;
}
