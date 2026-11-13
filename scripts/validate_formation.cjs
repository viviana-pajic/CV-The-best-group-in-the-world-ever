// Exercise the dashboard's formation engine with independently defined stage positions.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
require("../dashboard/formation.js");

const fps = 25;
const frameCount = 32;
const stages = [14, 36, 60, 82];
const horizontal = (depth, id = 1) => stages.map((x, i) => [id + i, x, depth, 0]);
const tilted = stages.map((x, i) => [i + 1, x, 50 + (x - 50) * Math.tan(Math.PI / 9), 0]);
const bent = horizontal(30);
bent[3][2] += 4;
const hidden = horizontal(30);
hidden[3][3] = 1;
const referenceDepth = 30;
const translated = horizontal(referenceDepth + 4);
const engine = createFormationEngine();
const rows = [];

function evaluate(name, points, expected, {angle = 0, transient = false} = {}) {
  const input = Array.from({length: frameCount}, (_, i) => transient && (i < 10 || i >= 16) ? [] : points);
  const results = engine.analyze(input, fps, 4, angle);
  const selected = results[12];
  assert.equal(selected.length, expected.length, name);
  for (let i = 0; i < expected.length; i++) {
    assert.ok(Math.abs(selected[i].score - expected[i]) < 1e-6, name);
    assert.equal(selected[i].members.length, 4, name);
  }
  const angleDegrees = selected.map(row => +row.angle.toFixed(2));
  rows.push({case: name, angle_limit_deg: angle, expected_rows: expected.length, fitted_rows: selected.length,
    fitted_angles_deg: angleDegrees, scores_pct: selected.map(row => +row.score.toFixed(2))});
  return selected;
}

evaluate("Two straight rows", [...horizontal(28), ...horizontal(62, 5)], [100, 100]);
evaluate("One dancer displaced by 0.48 m", bent, [100 * (1 - 1 / 6)]);
assert.equal(evaluate("Twenty-degree row within the angle limit", tilted, [100], {angle: 30})[0].angle.toFixed(2), "20.00");
evaluate("Twenty-degree row outside the angle limit", tilted, [], {angle: 0});
evaluate("Fourth position is estimated", hidden, []);
evaluate("Six-frame alignment during a transition", horizontal(30), [], {transient: true});
evaluate("Straight row translated 0.48 m from its reference", translated, [100]);
rows.at(-1).mean_offset_to_reference_m = translated.reduce((sum, point) => sum + Math.abs(point[2] - referenceDepth) * .12, 0) / translated.length;
assert.equal(rows.at(-1).mean_offset_to_reference_m, .48);

const output = path.join(__dirname, "../experiments/final_formation_validation");
fs.mkdirSync(output, {recursive: true});
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify({
  evidence: "Deterministic synthetic dancer positions supplied directly to the production dashboard engine",
  stage_m: [12, 12], fps, minimum_dancers: 4, tested_frame: 12,
  cases: rows,
}, null, 2) + "\n");
console.log(`${rows.length} formation scenarios matched their reference results.`);
