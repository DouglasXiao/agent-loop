// Same selection rules as ios/OneTip/Logic/TipPicker.swift, for the web prototype.
// Keep the two in sync: disliked never; age window (widen 3, then 6 months);
// unseen first, else least-recently-shown third; stalest category; seeded pick.
(function (root) {
  const MASK = (1n << 64n) - 1n;

  function splitMix64(seed) {
    let state = BigInt.asUintN(64, BigInt(seed));
    return function next() {
      state = (state + 0x9e3779b97f4a7c15n) & MASK;
      let z = state;
      z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK;
      z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK;
      return z ^ (z >> 31n);
    };
  }

  function pickFrom(list, rng) {
    if (!list.length) return null;
    return list[Number(rng() % BigInt(list.length))];
  }

  function fits(tip, age, slack) {
    return tip.age_min_months - slack <= age && age <= tip.age_max_months + slack;
  }

  function next(tips, { age, history, disliked, seed }) {
    const dislikedSet = new Set(disliked || []);
    const allowed = tips.filter((t) => !dislikedSet.has(t.id));
    if (!allowed.length) return null;

    let ageFit = [];
    for (const slack of [0, 3, 6]) {
      ageFit = allowed.filter((t) => fits(t, age, slack));
      if (ageFit.length) break;
    }
    if (!ageFit.length) ageFit = allowed;

    const lastShown = new Map();
    (history || []).forEach((id, i) => lastShown.set(id, i));

    let pool = ageFit.filter((t) => !lastShown.has(t.id));
    if (!pool.length) {
      const byAge = [...ageFit].sort((a, b) => (lastShown.get(a.id) ?? -1) - (lastShown.get(b.id) ?? -1));
      pool = byAge.slice(0, Math.max(1, Math.floor(byAge.length / 3)));
    }

    const categoryOf = new Map(tips.map((t) => [t.id, t.category]));
    const categoryLast = new Map();
    (history || []).forEach((id, i) => {
      if (categoryOf.has(id)) categoryLast.set(categoryOf.get(id), i);
    });

    const rng = splitMix64(seed);
    const categories = [...new Set(pool.map((t) => t.category))].sort();
    const oldest = Math.min(...categories.map((c) => categoryLast.get(c) ?? -1));
    const category = pickFrom(categories.filter((c) => (categoryLast.get(c) ?? -1) === oldest), rng);
    const candidates = pool.filter((t) => t.category === category).sort((a, b) => (a.id < b.id ? -1 : 1));
    return pickFrom(candidates, rng);
  }

  function ageInMonths(birthday, date) {
    let months = (date.getFullYear() - birthday.getFullYear()) * 12 + (date.getMonth() - birthday.getMonth());
    if (date.getDate() < birthday.getDate()) months -= 1;
    return Math.max(0, months);
  }

  // "2岁11个月零7天"
  function ageText(birthday, date) {
    const months = ageInMonths(birthday, date);
    const anchor = new Date(birthday.getFullYear(), birthday.getMonth() + months, birthday.getDate());
    const days = Math.round((startOfDay(date) - startOfDay(anchor)) / 86400000);
    const y = Math.floor(months / 12);
    const m = months % 12;
    let text = y > 0 ? `${y}岁` : "";
    if (m > 0) text += `${m}个月`;
    if (days > 0) text += (text ? "零" : "") + `${days}天`;
    return text || "今天出生";
  }

  function startOfDay(d) {
    return new Date(d.getFullYear(), d.getMonth(), d.getDate());
  }

  function dayKey(d) {
    const p = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
  }

  function daySeed(d) {
    return d.getFullYear() * 10000 + (d.getMonth() + 1) * 100 + d.getDate();
  }

  const api = { next, ageInMonths, ageText, dayKey, daySeed, splitMix64 };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.OneTipPicker = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
