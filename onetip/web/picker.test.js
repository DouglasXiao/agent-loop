// Run: node --test onetip/web/picker.test.js
const test = require("node:test");
const assert = require("node:assert");
const P = require("./picker.js");

const tip = (id, category, min = 30, max = 48) => ({ id, category, age_min_months: min, age_max_months: max });
const tips = [
  tip("emotion-001", "emotion"), tip("emotion-002", "emotion"),
  tip("play-001", "play"), tip("play-002", "play"),
  tip("sleep-001", "sleep"),
  tip("social-001", "social", 40, 54),
];

test("same seed gives same tip", () => {
  const a = P.next(tips, { age: 35, history: [], disliked: [], seed: 20261018 });
  const b = P.next(tips, { age: 35, history: [], disliked: [], seed: 20261018 });
  assert.strictEqual(a.id, b.id);
});

test("skips disliked and out-of-age tips", () => {
  for (let seed = 0; seed < 50; seed++) {
    const pick = P.next(tips, { age: 35, history: [], disliked: ["sleep-001"], seed });
    assert.notStrictEqual(pick.id, "sleep-001");
    assert.notStrictEqual(pick.id, "social-001");
  }
});

test("rotates to the stalest category", () => {
  const pick = P.next(tips, { age: 35, history: ["emotion-001", "play-001"], disliked: [], seed: 7 });
  assert.strictEqual(pick.category, "sleep");
});

test("a simulated month has no repeats while unseen tips remain, then recycles", () => {
  const history = [];
  for (let day = 0; day < 5; day++) history.push(P.next(tips, { age: 35, history, disliked: [], seed: day }).id);
  assert.strictEqual(new Set(history).size, 5);
  const sixth = P.next(tips, { age: 35, history, disliked: [], seed: 99 });
  assert.ok(history.includes(sixth.id), "recycles once all age-fitting tips are seen");
  assert.strictEqual(sixth.id, history[0], "recycles the least recently shown");
});

test("widens the age window when nothing fits", () => {
  const pick = P.next([tip("social-001", "social", 40, 54)], { age: 36, history: [], disliked: [], seed: 1 });
  assert.strictEqual(pick.id, "social-001");
});

test("age text matches the Chinese format", () => {
  const birthday = new Date(2023, 10, 11);
  const today = new Date(2026, 9, 18);
  assert.strictEqual(P.ageInMonths(birthday, today), 35);
  assert.strictEqual(P.ageText(birthday, today), "2岁11个月零7天");
  assert.strictEqual(P.ageText(new Date(2023, 9, 18), today), "3岁");
});
