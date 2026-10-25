// Scores an app with models/milestones.json in plain JavaScript (browser or node).
// Same math as playodds/model.py. tests/test_export.py checks both give the same numbers.
//
//   const M = await (await fetch("milestones.json")).json();
//   PlayOdds.score(M, {category: "Puzzle", name: "Block Drop", earlier_apps: 3}, 12)
//
// From the command line (used by the tests):  node models/score.js milestones.json examples.json

(function (root) {
  const DEFAULTS = {category: null, name: "", earlier_apps: 0, same_day_apps: 0, apps_last_30_days: 0,
    months_since_first_app: 0, earlier_categories: 0, earlier_apps_same_category: 0, builder_apps: 0};

  function nameParts(name) {
    name = name || "";
    const words = name.split(/\s+/).filter(w => w.length > 0);
    return {name_len: Array.from(name).length, name_words: words.length,
      name_nonascii: Array.from(name).some(ch => ch.codePointAt(0) > 127) ? 1 : 0,
      name_digit: /[0-9]/.test(name) ? 1 : 0};
  }

  function encode(app, terms) {
    const a = Object.assign({}, DEFAULTS, app);
    const v = Object.assign({}, a, nameParts(a.name));
    return terms.map(t => {
      const x = v[t.input];
      if (t.kind === "onehot") return x === t.level ? 1 : 0;
      const f = Number(x);
      if (t.kind === "bin") return (f >= t.lo && f <= t.hi) ? 1 : 0;
      if (t.kind === "log1p") return Math.log1p(Math.min(Math.max(f, 0), t.cap));
      if (t.kind === "linear") return Math.min(Math.max(f, 0), t.cap) / t.scale;
      throw new Error("unknown term kind " + t.kind);
    });
  }

  // standard normal cdf, Hart (1968) as given by West (2005), about 1e-14 accurate
  function normCdf(x) {
    const z = Math.abs(x);
    let c = 0;
    if (z <= 37) {
      const e = Math.exp(-z * z / 2);
      if (z < 7.07106781186547) {
        let b = 3.52624965998911e-02 * z + 0.700383064443688;
        b = b * z + 6.37396220353165; b = b * z + 33.912866078383; b = b * z + 112.079291497871;
        b = b * z + 221.213596169931; b = b * z + 220.206867912376;
        c = e * b;
        b = 8.83883476483184e-02 * z + 1.75566716318264; b = b * z + 16.064177579207;
        b = b * z + 86.7807322029461; b = b * z + 296.564248779674; b = b * z + 637.333633378831;
        b = b * z + 793.826512519948; b = b * z + 440.413735824752;
        c = c / b;
      } else {
        let b = z + 0.65; b = z + 4 / b; b = z + 3 / b; b = z + 2 / b; b = z + 1 / b;
        c = e / b / 2.506628274631;
      }
    }
    return x > 0 ? 1 - c : c;
  }

  function invLink(eta, link) {
    if (link === "cloglog") return -Math.expm1(-Math.exp(Math.min(eta, 30)));
    if (link === "logit") return 1 / (1 + Math.exp(-eta));
    if (link === "probit") return normCdf(eta);
    throw new Error("unknown link " + link);
  }

  function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

  function prob(model, milestone, x, months, z) {
    const m = model.milestones[String(milestone)];
    const eta = m.intercept + m.log_time * Math.log(months) + dot(x, m.coef);
    const p = invLink(eta, model.link);
    if (z === undefined) return p;
    const g = [1, Math.log(months)].concat(x);
    let v = 0;
    for (let i = 0; i < g.length; i++) v += g[i] * dot(m.cov[i], g);
    const se = Math.sqrt(Math.max(v, 0));
    return {p: p, lo: invLink(eta - z * se, model.link), hi: invLink(eta + z * se, model.link)};
  }

  function searchRight(edges, s) { let i = 0; while (i < edges.length && edges[i] <= s) i++; return i; }

  function interp(x, xs, ys) {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
    let i = 1; while (xs[i] < x) i++;
    return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
  }

  function score(model, app, months, level) {
    months = months === undefined ? 12 : months; level = level || "80";
    if (!(months >= 0.5 && months <= 66)) throw new Error("months must be between 0.5 and 66, the ages the model was fit on");
    const x = encode(app, model.terms);
    const z = model.z[level];
    const out = {months: months, milestones: {}};
    let prev = null;
    for (const ms of model.milestone_list) {
      let r = prob(model, ms, x, months, z);
      if (prev) r = {p: Math.min(r.p, prev.p), lo: Math.min(r.lo, prev.lo), hi: Math.min(r.hi, prev.hi)};
      out.milestones[ms] = prev = r;
    }
    const s = dot(x, model.milestones[model.index_milestone].coef);
    const c = model.conformal[level];
    const i = Math.min(searchRight(c.edges, s), c.lo.length - 1);
    out.percentile_interval = [c.lo[i], c.hi[i]];
    const tab = model.percentile_installs[String(months)];
    if (tab) out.installs_interval = out.percentile_interval.map(q => interp(q * 100, tab.pct, tab.installs));
    out.score = s;
    return out;
  }

  const api = {encode: encode, prob: prob, score: score, invLink: invLink};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.PlayOdds = api;

  if (typeof require !== "undefined" && typeof module !== "undefined" && require.main === module) {
    const fs = require("fs");
    const model = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
    const examples = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
    const out = examples.map(e => {
      const x = encode(e.app, model.terms);
      const p = {};
      for (const ms of model.milestone_list) p[ms] = prob(model, ms, x, e.months);
      return {p: p, scored: score(model, e.app, 12, "80")};
    });
    process.stdout.write(JSON.stringify(out));
  }
})(typeof window !== "undefined" ? window : this);
