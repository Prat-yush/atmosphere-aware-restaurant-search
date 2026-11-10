// AmbiSense ranking engine for the browser: a port of src/parse.py and the "full" system of
// src/rank.py. Retrieval (BM25 + dense + RRF) is precomputed by src/export_web.py for every
// (pole, lexicon term) pair; this file parses the query and does the scoring arithmetic.

const TOKEN_RE = /[a-z0-9]+(?:'[a-z]+)?/g;
const PUNCT_RE = /[,.;:!?()]/;
const WORD = "\\p{L}\\p{N}\\p{M}_'"; // Python's \w (Unicode) plus apostrophe

export function normalizeText(text) {
  return text.replace(/’/g, "'").replace(/‘/g, "'").toLowerCase();
}

export function termKey(text) {
  return text.trim().replace(/[\s-]+/g, " ");
}

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function termToPattern(term) {
  return normalizeText(term).trim().split(/[\s-]+/).map(escapeRegex).join("[\\s-]+");
}

function compileTerms(terms) {
  const ordered = [...new Set(terms)].sort((a, b) => b.length - a.length); // longest first
  const alternation = ordered.map(termToPattern).join("|");
  return new RegExp(`(?<![${WORD}])(?:${alternation})(?![${WORD}])`, "gu");
}

// Regex that finds lexicon terms in display text (for highlighting evidence).
export function highlighter(terms) {
  return compileTerms(terms.map((t) => termKey(normalizeText(t))));
}

export class QueryParser {
  constructor(lexicon) {
    this.cues = new Set(lexicon.negation_cues);
    this.boundaries = new Set(lexicon.clause_boundaries);
    this.window = lexicon.negation_window;
    this.poles = {};
    this.termPole = new Map();
    for (const [aspect, poles] of Object.entries(lexicon.aspects)) {
      this.poles[aspect] = { positive: poles.positive.id, negative: poles.negative.id };
      for (const side of ["positive", "negative"]) {
        for (const term of poles[side].terms) this.termPole.set(termKey(normalizeText(term)), [aspect, side]);
      }
    }
    this.matcher = compileTerms([...this.termPole.keys()]);
    this.exclusion = lexicon.exclude.length ? compileTerms(lexicon.exclude) : null;
  }

  parse(query) {
    let text = normalizeText(query);
    if (this.exclusion) text = text.replace(this.exclusion, (m) => " ".repeat(m.length)); // keep offsets
    const tokens = [...text.matchAll(TOKEN_RE)].map((m) => ({ text: m[0], start: m.index, end: m.index + m[0].length }));
    const targets = [];
    const usedCues = new Set();
    let prevEnd = 0;
    for (const m of text.matchAll(this.matcher)) {
      const start = m.index;
      const end = start + m[0].length;
      let [aspect, side] = this.termPole.get(termKey(m[0]));
      const cue = this.findNegationCue(text, tokens, start, prevEnd, usedCues);
      if (cue !== null) {
        usedCues.add(cue);
        side = side === "positive" ? "negative" : "positive";
      }
      prevEnd = end;
      const pole = this.poles[aspect][side];
      if (targets.every((t) => t.aspect !== aspect || t.side !== side)) {
        targets.push({ aspect, pole, side, term: m[0], negated: cue !== null, column: `${aspect}.${pole}` });
      }
    }
    return targets;
  }

  findNegationCue(text, tokens, termStart, prevEnd, used) {
    const before = [];
    tokens.forEach((t, i) => {
      if (prevEnd <= t.start && t.end <= termStart) before.push(i);
    });
    const last = before[before.length - 1];
    const windowed = before.slice(-this.window).reverse();
    for (const i of windowed) {
      const tok = tokens[i];
      const gap = text.slice(tok.end, i === last ? termStart : tokens[i + 1].start);
      if (PUNCT_RE.test(gap) || this.boundaries.has(tok.text)) return null;
      if ((this.cues.has(tok.text) || tok.text.endsWith("n't")) && !used.has(i)) return i;
    }
    return null;
  }
}

export class Engine {
  // load(path) must resolve to parsed JSON for a path relative to docs/idx/.
  constructor(lexicon, restaurants, load) {
    this.lexicon = lexicon;
    this.params = lexicon.params;
    this.parser = new QueryParser(lexicon);
    this.load = load;
    this.cache = new Map();
    const f = restaurants.fields;
    this.restaurants = restaurants.rows.map((row) => Object.fromEntries(f.map((name, i) => [name, row[i]])));
  }

  static async create(load) {
    const [lexicon, restaurants] = await Promise.all([load("lexicon.json"), load("restaurants.json")]);
    return new Engine(lexicon, restaurants, load);
  }

  hypothesis(target) {
    return this.lexicon.aspects[target.aspect][target.side].hypothesis;
  }

  async retrieval(target) {
    const n = this.lexicon.files[`${target.column}|${termKey(target.term)}`];
    if (n === undefined) return [];
    if (!this.cache.has(n)) this.cache.set(n, this.load(`r/${n}.json`));
    return this.cache.get(n);
  }

  // Per-restaurant scores for one target, as in Ranker.score_target (full system).
  async scoreTarget(target) {
    const { k, m } = this.params;
    const groups = new Map();
    for (const [bi, sim, pol, text] of await this.retrieval(target)) {
      if (!groups.has(bi)) groups.set(bi, []);
      const ev = groups.get(bi);
      if (ev.length < k) ev.push({ sim, polarity: pol, text });
    }
    const raws = new Map();
    for (const [bi, ev] of groups) {
      const total = ev.reduce((s, e) => s + e.sim, 0);
      let raw = 0;
      for (const e of ev) {
        e.weight = e.sim / total;
        raw += e.weight * e.polarity;
      }
      raws.set(bi, raw);
    }
    const values = [...raws.values()];
    const mu = values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
    const scores = new Map();
    for (const [bi, ev] of groups) {
      const raw = raws.get(bi);
      const n = ev.length;
      scores.set(bi, { target, score: (n * raw + m * mu) / (n + m), raw, n, mu, evidence: ev });
    }
    return { scores, mu };
  }

  async rank(query, top = 20) {
    const targets = this.parser.parse(query);
    if (!targets.length) return { targets, results: [], total: 0 };
    const perTarget = await Promise.all(targets.map((t) => this.scoreTarget(t)));
    const candidates = new Set();
    for (const { scores } of perTarget) for (const bi of scores.keys()) candidates.add(bi);
    const results = [];
    for (const bi of candidates) {
      const ts = perTarget.map(({ scores, mu }, j) => scores.get(bi) || { target: targets[j], score: mu, raw: null, n: 0, mu, evidence: [] });
      const score = ts.reduce((s, t) => s + t.score, 0) / ts.length;
      const nEvidence = ts.reduce((s, t) => s + t.n, 0);
      results.push({ restaurant: this.restaurants[bi], score, nEvidence, targets: ts });
    }
    results.sort(
      (a, b) =>
        b.score - a.score ||
        b.nEvidence - a.nEvidence ||
        (a.restaurant.business_id < b.restaurant.business_id ? -1 : a.restaurant.business_id > b.restaurant.business_id ? 1 : 0),
    );
    return { targets, results: top ? results.slice(0, top) : results, total: results.length };
  }
}
