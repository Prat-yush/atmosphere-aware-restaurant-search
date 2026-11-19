import { Engine, highlighter } from "./engine.js";

const EXAMPLES = [
  "quiet but not boring, good for a first date",
  "lively but not loud",
  "cozy, dimly lit and intimate",
  "romantic with soft music",
  "not too loud",
];
// Plain-language names for each atmosphere pole.
const WORD = {
  quiet: "quiet", loud: "loud", lively: "lively", dull: "low-key", cozy: "cozy", cramped: "cramped",
  romantic: "romantic", unromantic: "casual, not a date spot", dim: "dimly lit", bright: "bright",
  good_music: "good music", loud_music: "loud music",
};
const QUESTION = {
  quiet: "Is it quiet?", loud: "Is it loud?", lively: "Is it lively?", dull: "Is it low-key?",
  cozy: "Is it cozy?", cramped: "Is it cramped?", romantic: "Is it romantic?",
  unromantic: "Is it more casual than romantic?", dim: "Is it dimly lit?", bright: "Is it bright?",
  good_music: "Is the music good?", loud_music: "Is the music loud?",
};
const PAGE = 10;
const MAX = 50;
const GENERIC = new Set(["Restaurants", "Food", "Nightlife", "Bars", "Event Planning & Services", "Caterers"]);

const $ = (id) => document.getElementById(id);
const form = $("search"), input = $("q"), status = $("status"), list = $("results"), more = $("more"), looking = $("looking");
const reviews = $("reviews"), how = $("how");

let engine = null;
let marks = {};
let current = { targets: [], results: [], shown: 0 };

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const word = (t) => WORD[t.pole] || t.pole.replace(/_/g, " ");
const chips = (targets) => targets.map((t) => `<span class="chip">${esc(word(t))}</span>`).join(" ");

function setStatus(text, error = false) {
  status.textContent = text;
  status.classList.toggle("error", error);
}

function renderExamples() {
  const box = $("examples");
  EXAMPLES.forEach((q, i) => {
    if (i) box.insertAdjacentHTML("beforeend", '<span class="sep">/</span>');
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = q;
    b.addEventListener("click", () => { input.value = q; run(q); });
    box.append(b);
  });
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

const cuisine = (r) => (r.categories || "").split(", ").filter((c) => !GENERIC.has(c)).slice(0, 3).join(", ");
// Five SVG stars: full, half (filled from the left, via the #half-star clip in index.html), empty.
const STAR = "M12 2.5l2.94 6.1 6.56.9-4.8 4.6 1.2 6.6L12 17.6l-5.9 3.1 1.2-6.6-4.8-4.6 6.56-.9z";
const star = (kind) => `<svg class="star" viewBox="0 0 24 24"><path class="bg" d="${STAR}"/>${kind === "empty" ? "" : `<path class="fg" d="${STAR}"${kind === "half" ? ' clip-path="url(#half-star)"' : ""}/>`}</svg>`;
const stars = (s) => Array.from({ length: 5 }, (_, i) => star(s >= i + 1 ? "full" : s >= i + 0.5 ? "half" : "empty")).join("");
function mapsUrl(r) {
  const q = `${r.name}, ${r.address}, Philadelphia, PA ${r.postal_code || ""}`;
  return `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(q)}`;
}

function renderListing(res, rank) {
  const r = res.restaurant;
  const li = document.createElement("li");
  li.className = "listing";
  const meta = [
    r.price ? `<li>${"$".repeat(r.price)}</li>` : "",
    `<li><span class="stars" aria-hidden="true">${stars(r.stars)}</span>&nbsp;${r.stars} <span class="count">(${r.review_count.toLocaleString()} reviews)</span></li>`,
    `<li class="addr"><a href="${mapsUrl(r)}" target="_blank" rel="noopener">${esc(r.address)}</a></li>`,
  ].join("");
  const kind = cuisine(r);
  li.innerHTML = `
    <div class="num">${rank}</div>
    <div>
      <h2 class="name">${esc(r.name)}</h2>
      ${kind ? `<p class="cuisine">${esc(kind)}</p>` : ""}
      <ul class="meta" aria-label="Details">${meta}</ul>
      <button type="button" class="open-reviews" data-i="${rank - 1}">See what people are saying</button>
    </div>`;
  return li;
}

function quoteList(items, aspect, cls) {
  return `<ul class="quotes ${cls}">${items.map((e) => `<li>“${highlight(e.text, aspect)}”</li>`).join("")}</ul>`;
}

function openReviews(res) {
  const r = res.restaurant;
  $("reviews-title").textContent = r.name;
  $("reviews-meta").textContent = [cuisine(r), r.address].filter(Boolean).join(" · ");
  $("reviews-body").innerHTML = res.targets.map((ts) => {
    const q = QUESTION[ts.target.pole] || `Is it ${word(ts.target)}?`;
    // The 3 most on-topic sentences per question, so the agree/disagree mix reflects the evidence.
    const byWeight = [...ts.evidence].sort((a, b) => b.weight - a.weight).slice(0, 3);
    const yes = byWeight.filter((e) => e.polarity > 0), no = byWeight.filter((e) => e.polarity <= 0);
    const aspect = ts.target.aspect;
    let body;
    if (!byWeight.length) body = `<p class="nothing">None of the reviews we searched mention this either way.</p>`;
    else body = (yes.length ? `<p class="side">Reviewers who agree</p>${quoteList(yes, aspect, "yes")}` : "")
      + (no.length ? `<p class="side no">Reviewers who don’t</p>${quoteList(no, aspect, "no")}` : "");
    return `<section class="topic"><h3>${esc(q)}</h3>${body}</section>`;
  }).join("");
  reviews.showModal();
  $("reviews-body").scrollTop = 0;
}

// Short technical explanation of the pipeline.
function openHow() {
  const { k, m, top_n, k_rrf } = engine ? engine.params : { k: 10, m: 4, top_n: 1000, k_rrf: 60 };
  const step = (title, text) => `<li><div><h3>${title}</h3><p>${text}</p></div></li>`;
  $("how-body").innerHTML = `<ol class="steps">
    ${step("Query parsing", "A rule-based lexicon of 174 terms maps your words to atmosphere targets (noise, energy, space, romance, lighting, music). A nearby negation flips the target, so “not boring” becomes <i>lively</i>.")}
    ${step("Candidate sentences", "Yelp reviews of 3,897 Philadelphia restaurants are split into sentences, and only those that mention an atmosphere term are kept.")}
    ${step("Hybrid retrieval", `For each target, BM25 and dense embeddings (MiniLM + FAISS) each rank the sentences. The two rankings are merged with reciprocal rank fusion (k = ${k_rrf}), keeping the top ${top_n.toLocaleString()}.`)}
    ${step("NLI polarity", "A DeBERTa-v3 NLI model scores each sentence from −1 to +1 by whether it contradicts or supports a hypothesis like “The atmosphere was quiet.”")}
    ${step("Per-restaurant score", `Each restaurant’s top ${k} sentences are averaged, weighted by retrieval similarity. The average is then shrunk toward the overall mean (m = ${m}), so a place with only a few mentions can’t jump to the top.`)}
    ${step("Final ranking", "Restaurants are ranked by the mean of their per-target scores, and a target with no evidence counts as average. Star ratings and popularity aren’t used.")}
  </ol>`;
  how.showModal();
  $("how-body").scrollTop = 0;
}

for (const d of [reviews, how]) {
  d.addEventListener("click", (ev) => {
    if (ev.target === d || ev.target.closest("[data-close]")) d.close(); // backdrop or × button
  });
}
$("how-open").addEventListener("click", openHow);
list.addEventListener("click", (ev) => {
  const btn = ev.target.closest(".open-reviews");
  if (btn) openReviews(current.results[Number(btn.dataset.i)]);
});

function showMore() {
  const next = current.results.slice(current.shown, current.shown + PAGE);
  next.forEach((res, i) => list.append(renderListing(res, current.shown + i + 1)));
  current.shown += next.length;
  more.hidden = current.shown >= current.results.length;
}
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
    const { targets, results, total } = await engine.rank(query, MAX);
    current = { targets, results, shown: 0 };
    looking.hidden = false;
    if (!targets.length) {
      looking.innerHTML = `<b>Hmm</b>We didn’t catch an atmosphere in that. Try words like <i>quiet</i>, <i>lively</i>, <i>cozy</i>, <i>romantic</i>, <i>dimly lit</i> or <i>live music</i>.`;
      setStatus("");
      return;
    }
    looking.innerHTML = `<b>Looking for</b>${chips(targets)}`;
    if (!total) { setStatus(""); list.innerHTML = `<li class="empty">No reviews talk about this yet.</li>`; return; }
    setStatus(`Showing the best of ${total.toLocaleString()} restaurants whose reviews mention this.`);
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
