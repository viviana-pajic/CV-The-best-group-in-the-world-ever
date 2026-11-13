// Verify coverage independently from the dashboard heat map implementation.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const dashboard = path.resolve(process.argv[2]);
require(path.join(dashboard, "heatmap.js"));
const context = {window: {}};
for (const name of ["stage-replay", "CbpSMoPItu0", "YoF0XeYhPL0", "0BorT97NOmQ"]) {
  const script = fs.readFileSync(path.join(dashboard, "assets", name + ".js"), "utf8");
  vm.runInNewContext(script, context);
}

function coverageAt(clip, end) {
  const squares = new Set();
  let samples = 0;
  const [width, depth] = clip.stage;
  for (let frame = 0; frame <= end; frame++) {
    for (const [, x, y, estimated] of clip.frames[frame][1]) {
      if (estimated || x < 0 || x > width || y < 0 || y > depth) continue;
      squares.add([Math.min(11, Math.floor(12 * x / width)),
                   Math.min(11, Math.floor(12 * y / depth))].join(","));
      samples++;
    }
  }
  return {squares: squares.size, samples};
}

const results = context.window.DANCE_CLIPS.map(clip => {
  const heatmap = createStageHeatmap(clip);
  const end = clip.frames.length - 1;
  const moments = [0, Math.floor(end / 2), end, clip.startFrame, 0, end];
  const coverage = moments.map(frame => {
    const expected = coverageAt(clip, frame);
    const actual = heatmap.at(frame);
    assert.equal(actual.visited, expected.squares, clip.title);
    assert.equal(actual.coverageSide ** 2, 144);
    const mass = actual.density.reduce((total, value) => total + value, 0);
    assert.ok(Math.abs(mass - expected.samples) < .01);
    return {frame, visited: actual.visited, reported_contact_samples: expected.samples};
  });
  return {title: clip.title, coverage};
});
console.log(JSON.stringify(results, null, 2));
