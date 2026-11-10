import { Engine, highlighter } from "./engine.js";

const EXAMPLES = [
  "quiet but not boring, good for a first date",
  "lively but not loud",
  "cozy, dimly lit and intimate",
  "energetic without loud music",
  "romantic spot with candlelight and soft music",
  "not too loud",
];
const ASPECT = { noise: "Noise", energy: "Energy", space: "Space", romance: "Romance", lighting: "Lighting", music: "Music" };
const POLE = {
  quiet: "quiet", loud: "loud", lively: "lively", dull: "dull", cozy: "comfortable", cramped: "cramped",
  romantic: "romantic", unromantic: "not romantic", dim: "dim", bright: "bright",
  good_music: "pleasant music", loud_music: "music too loud",
};
const PAGE = 10;
const MAX = 50;
const SKIP_CATEGORIES = new Set(["Restaurants", "Food", "Nightlife", "Bars"]);

const $ = (id) => document.getElementById(id);
const form = $("search"), input = $("q"), status = $("status"), list = $("results"), more = $("more"), understood = $("understood");

let engine = null;
let marks = {};
let current = { results: [], shown: 0 };

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const signed = (x) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(2);

function setStatus(text, error = false) {
  status.textContent = text;
  status.classList.toggle("error", error);
}

function renderExamples() {
  const box = $("examples");
  for (const q of EXAMPLES) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = q;
    b.addEventListener("click", () => { input.value = q; run(q); });
    box.append(b);
  }
}

function meter(score) {
  const s = Math.max(-1, Math.min(1, score));
  const left = s >= 0 ? 50 : 50 + s * 50;
  const color = s >= 0 ? "var(--support)" : "var(--contra)";
  return `<span class="meter" aria-hidden="true"><i style="left:${left}%;width:${Math.abs(s) * 50}%;background:${color}"></i></span>`;
}

function highlight(text, aspect) {
  const re = marks[aspect];
  if (!re) return esc(text);
  let out = "", last = 0;
  for (const m of text.toLowerCase().matchAll(re)) {
    out += esc(text.slice(last, m.index)) + "<mark>" + esc(text.slice(m.index, m.index + m[0].length)) + "</mark>";
    last = m.index + m[0].length;
  }
  return out + esc(text.slice(last));
}

function targetLabel(t) {
  return `${ASPECT[t.aspect]}: ${POLE[t.pole] || t.pole}`;
}

function renderUnderstood(targets) {
  understood.hidden = false;
  if (!targets.length) {
    understood.innerHTML = "No atmosphere words recognized. Try words like <b>quiet</b>, <b>lively</b>, <b>cozy</b>, <b>romantic</b>, <b>dim</b> or <b>live music</b>.";
    return;
  }
  understood.innerHTML = "Looking for" + targets.map((t) => {
    const from = t.negated ? `from “not … ${esc(t.term)}”` : `from “${esc(t.term)}”`;
    return ` <span class="target-chip"><b>${esc(targetLabel(t))}</b> <span class="from">${from}</span></span>`;
  }).join("");
}

function evidenceItem(e, aspect) {
  const sup = e.polarity > 0;
  return `<li class="${sup ? "sup" : "con"}"><span class="tag">${sup ? "Supports" : "Against"} ${signed(e.polarity)}</span>${highlight(e.text, aspect)}</li>`;
}

function renderTarget(ts, single) {
  const head = `<div class="t-head"><span class="t-label">${esc(single ? "Evidence" : targetLabel(ts.target))}</span>${meter(ts.score)}<span class="t-num">${ts.n ? `${signed(ts.score)} · ${ts.n} sentence${ts.n === 1 ? "" : "s"}` : `${signed(ts.score)} · no evidence`}</span></div>`;
  if (!ts.n) return `<div>${head}<p class="none">No retrieved review sentences cover this, so it gets the average score for restaurants that do (${signed(ts.mu)}).</p></div>`;
  const byWeight = [...ts.evidence].sort((a, b) => b.weight - a.weight);
  const sup = byWeight.filter((e) => e.polarity > 0), con = byWeight.filter((e) => e.polarity <= 0);
  const preview = [...sup.slice(0, 2), ...con.slice(0, 1)];
  const rest = byWeight.filter((e) => !preview.includes(e));
  const aspect = ts.target.aspect;
  return `<div>${head}
    <ul class="evidence">${preview.map((e) => evidenceItem(e, aspect)).join("")}</ul>
    ${rest.length ? `<ul class="evidence" hidden>${rest.map((e) => evidenceItem(e, aspect)).join("")}</ul>
    <button type="button" class="toggle" data-more="${rest.length}">Show ${rest.length} more</button>` : ""}
  </div>`;
}

function mapsUrl(r) {
  const q = `${r.name}, ${r.address}, Philadelphia, PA ${r.postal_code || ""}`;
  return `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(q)}`;
}

function renderCard(res, rank) {
  const r = res.restaurant;
  const cats = (r.categories || "").split(", ").filter((c) => !SKIP_CATEGORIES.has(c)).slice(0, 3).join(" · ");
  const bits = [r.price ? "$".repeat(r.price) : null, `★ ${r.stars} (${r.review_count.toLocaleString()})`, cats, r.address].filter(Boolean);
  const single = res.targets.length === 1;
  const li = document.createElement("li");
  li.className = "card";
  li.innerHTML = `
    <div class="card-head">
      <div class="rank">${rank}</div>
      <div>
        <h2 class="name"><a href="${mapsUrl(r)}" target="_blank" rel="noopener">${esc(r.name)}</a></h2>
        <div class="meta">${bits.map(esc).join(" · ")}</div>
      </div>
      <div class="score"><strong>${signed(res.score)}</strong><span>match</span></div>
    </div>
    <div class="targets">${res.targets.map((ts) => renderTarget(ts, single)).join("")}</div>`;
  return li;
}

function showMore() {
  const next = current.results.slice(current.shown, current.shown + PAGE);
  next.forEach((res, i) => list.append(renderCard(res, current.shown + i + 1)));
  current.shown += next.length;
  more.hidden = current.shown >= current.results.length;
}

list.addEventListener("click", (ev) => {
  const btn = ev.target.closest(".toggle");
  if (!btn) return;
  const hidden = btn.previousElementSibling;
  const open = hidden.hidden;
  hidden.hidden = !open;
  btn.textContent = open ? "Show fewer" : `Show ${btn.dataset.more} more`;
});
more.addEventListener("click", showMore);

async function run(query) {
  query = query.trim();
  if (!query) return;
  const url = new URL(location.href);
  url.searchParams.set("q", query);
  history.replaceState(null, "", url);
  list.innerHTML = "";
  more.hidden = true;
  setStatus("Searching…");
  try {
    if (!engine) await ready;
    const t0 = performance.now();
    const { targets, results, total } = await engine.rank(query, MAX);
    renderUnderstood(targets);
    current = { results, shown: 0 };
    if (!targets.length) { setStatus(""); return; }
    const ms = Math.round(performance.now() - t0);
    setStatus(total ? `${total.toLocaleString()} restaurants with matching review evidence · ${ms} ms` : "No restaurants have reviews about this.");
    showMore();
  } catch (err) {
    console.error(err);
    setStatus("Something went wrong loading the search data. Please reload the page.", true);
  }
}

form.addEventListener("submit", (ev) => { ev.preventDefault(); run(input.value); });

const load = async (path) => {
  const res = await fetch(`idx/${path}`);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
};
const ready = Engine.create(load).then((e) => {
  engine = e;
  for (const [aspect, poles] of Object.entries(e.lexicon.aspects)) marks[aspect] = highlighter([...poles.positive.terms, ...poles.negative.terms]);
});
ready.catch((err) => { console.error(err); setStatus("Could not load the search data.", true); });

renderExamples();
const initial = new URL(location.href).searchParams.get("q");
if (initial) { input.value = initial; run(initial); }
