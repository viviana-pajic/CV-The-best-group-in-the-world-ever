function createStageHeatmap({frames, stage}) {
  const coverageSide = 12;
  const cellsPerSquare = 4;
  const side = coverageSide * cellsPerSquare;
  const [stageWidth, stageDepth] = stage;
  const counts = new Uint32Array(side * side);
  const coverage = new Uint32Array(coverageSide ** 2);
  const bins = frames.map(([, points]) => points
    .filter(([, x, y, estimated]) => !estimated && x >= 0 && x <= stageWidth && y >= 0 && y <= stageDepth)
    .map(([, x, y]) => {
      const col = Math.min(side - 1, Math.floor(x / stageWidth * side));
      const row = Math.min(side - 1, Math.floor(y / stageDepth * side));
      return row * side + col;
    }));
  const coverageIndex = Uint16Array.from(counts, (_, i) =>
    Math.floor(Math.floor(i / side) / cellsPerSquare) * coverageSide +
    Math.floor((i % side) / cellsPerSquare));
  // 0.6 m on the 12 m stage, sampled at four cells per meter.
  const sigma = 2.4;
  const radius = Math.ceil(3 * sigma);
  const weights = Array.from({length: 2 * radius + 1}, (_, i) => Math.exp(-.5 * ((i - radius) / sigma) ** 2));
  const weightSum = weights.reduce((sum, weight) => sum + weight, 0);
  const kernel = weights.map(weight => weight / weightSum);
  const neighbors = Array.from({length: side}, (_, i) => kernel.map((_, k) => {
    const cell = i + k - radius;
    if (cell < 0) return -1 - cell;
    if (cell >= side) return 2 * side - 1 - cell;
    return cell;
  }));
  const horizontal = new Float32Array(counts.length);
  const density = new Float32Array(counts.length);
  let through = -1;
  let visited = 0;

  function update(frame, delta) {
    for (const index of bins[frame]) {
      const square = coverageIndex[index];
      if (coverage[square] === 0) visited++;
      coverage[square] += delta;
      counts[index] += delta;
      if (coverage[square] === 0) visited--;
    }
  }

  function at(frame) {
    for (let i = through + 1; i <= frame; i++) update(i, 1);
    for (let i = through; i > frame; i--) update(i, -1);
    through = frame;
    let peak = 0;
    if (!visited) {
      density.fill(0);
    } else {
      for (let y = 0; y < side; y++) {
        for (let x = 0; x < side; x++) {
          let value = 0;
          for (let k = 0; k < kernel.length; k++) value += counts[y * side + neighbors[x][k]] * kernel[k];
          horizontal[y * side + x] = value;
        }
      }
      for (let y = 0; y < side; y++) {
        for (let x = 0; x < side; x++) {
          let value = 0;
          for (let k = 0; k < kernel.length; k++) value += horizontal[neighbors[y][k] * side + x] * kernel[k];
          density[y * side + x] = value;
          peak = Math.max(peak, value);
        }
      }
    }
    return {density, peak, visited, coverageSide};
  }

  return {at, side};
}

globalThis.createStageHeatmap = createStageHeatmap;
