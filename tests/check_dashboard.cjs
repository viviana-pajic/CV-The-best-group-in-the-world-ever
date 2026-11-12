const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
require("../dashboard/formation.js");

const specs = [
  ["stage-replay", 2424, 3250],
  ["CbpSMoPItu0", 4473, 0],
  ["YoF0XeYhPL0", 4713, 0],
  ["0BorT97NOmQ", 6278, 0],
];
const context = {window: {}};
const index = fs.readFileSync("dashboard/index.html", "utf8");
for (const [id] of specs) {
  assert.ok(index.includes(`src="assets/${id}.js"`), `Missing picker registration for ${id}`);
  vm.runInNewContext(fs.readFileSync(`dashboard/assets/${id}.js`, "utf8"), context);
}
const engine = createFormationEngine();
for (const [i, clip] of context.window.DANCE_CLIPS.entries()) {
  const [id, count, first] = specs[i];
  assert.equal(clip.id, id);
  assert.equal(clip.frames.length, count);
  assert.equal(clip.firstFrame, first);
  assert.equal(clip.fps, 25);
  assert.equal(clip.threshold, 0.30);
  assert.ok(clip.stage.every(size => size === 100));
  assert.ok(clip.stageMeters.every(size => size === 12));
  for (const asset of [clip.poster, clip.video]) assert.ok(fs.statSync(`dashboard/${asset}`).size > 0);
  let framesWithDots = 0;
  const measured = [];
  for (const [boxes, contacts, guides] of clip.frames) {
    assert.equal(new Set(boxes.map(box => box[0])).size, boxes.length);
    assert.equal(new Set(contacts.map(point => point[0])).size, contacts.length);
    if (contacts.length) framesWithDots++;
    for (const [, x, y, estimated, u, v] of contacts) {
      assert.ok(x >= 0 && x <= 100 && y >= 0 && y <= 100);
      assert.ok(estimated === 0 || estimated === 1);
      assert.ok(Number.isFinite(u) && Number.isFinite(v));
    }
    if (i > 0) {
      assert.ok(guides.length === 14 || (guides.length === 0 && contacts.length === 0));
      assert.ok(guides.every(([, a, b]) => [...a, ...b].every(Number.isFinite)));
    }
    measured.push(contacts);
  }
  const results = {};
  for (const [minimum, angle] of [[4, 10], [2, 30]]) {
    const lines = engine.analyze(measured, clip.fps, minimum, angle);
    assert.equal(lines.length, count);
    for (const [frame, rows] of lines.entries()) {
      const memberIds = new Set();
      for (const row of rows) {
        assert.ok(row.members.length >= minimum && Math.abs(row.angle) <= angle + 1e-9);
        for (const [id] of row.members) {
          assert.ok(!memberIds.has(id));
          assert.ok(measured[frame].some(point => point[0] === id && point[3] === 0));
          memberIds.add(id);
        }
      }
    }
    results[`${minimum} at ${angle}°`] = {
      framesWithLines: lines.filter(rows => rows.length > 0).length,
      initial: lines[clip.startFrame].map(row => ({dots: row.members.length, angle: +row.angle.toFixed(1),
        score: +row.score.toFixed(1)})),
    };
  }
  console.log(JSON.stringify({title: clip.title, count, framesWithDots, ...results}));
}
