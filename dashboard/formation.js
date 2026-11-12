// The fit works in stage units (one unit is 1% of the stage width).
// Keep this factory self-contained so the same code runs in a local-file worker.
function createFormationEngine() {
  const MIN_SPAN = 20;
  const MIN_GAP = 3.8;
  const MAX_SPREAD = 8;
  const MAX_ERROR = 2.6;
  const ZERO_SCORE_OFFSET = 6;
  const radians = Math.PI / 180;

  function median(values) {
    values.sort((a, b) => a - b);
    const middle = Math.floor(values.length / 2);
    return values.length % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
  }

  function fit(points, limit, smoothedAngle) {
    let angle = smoothedAngle ?? 0;
    if (smoothedAngle === undefined && limit > 0) {
      const slopes = [];
      for (let i = 0; i < points.length; i++) {
        for (let j = i + 1; j < points.length; j++) {
          const dx = points[j][1] - points[i][1];
          if (Math.abs(dx) >= MIN_GAP) slopes.push((points[j][2] - points[i][2]) / dx);
        }
      }
      if (slopes.length) angle = Math.max(-limit, Math.min(limit, Math.atan(median(slopes)) / radians));
    }
    const cos = Math.cos(angle * radians);
    const sin = Math.sin(angle * radians);
    const along = points.map(([, x, y]) => x * cos + y * sin);
    const across = points.map(([, x, y]) => y * cos - x * sin);
    const offset = median([...across]);
    const errors = across.map(value => value - offset);
    return {angle, cos, sin, along, across, offset, errors,
      error: errors.reduce((total, value) => total + Math.abs(value), 0) / points.length};
  }

  function candidate(points, model) {
    const xs = points.map(point => point[1]);
    return {
      points, angle: model.angle, center: (model.offset + 50 * model.sin) / model.cos,
      slope: model.sin / model.cos, left: Math.min(...xs), right: Math.max(...xs),
      ids: new Set(points.map(point => point[0])),
    };
  }

  function distance(a, b) {
    const x = (Math.max(a.left, b.left) + Math.min(a.right, b.right)) / 2;
    return Math.abs(a.center - b.center + (x - 50) * (a.slope - b.slope));
  }

  function compatible(a, b) {
    return !a.points.some(point => b.ids.has(point[0])) &&
      (Math.abs(a.angle - b.angle) > 8 || distance(a, b) >= 6);
  }

  function propose(points, minimum, limit) {
    if (points.length < minimum) return [];
    const steps = Math.ceil(limit / 3);
    const seen = new Set();
    const options = [];
    for (let step = -steps; step <= steps; step++) {
      const theta = (steps ? step * limit / steps : 0) * radians;
      const cos = Math.cos(theta);
      const sin = Math.sin(theta);
      const ordered = points.map(point => [point[2] * cos - point[1] * sin, point])
        .sort((a, b) => a[0] - b[0]);
      for (let start = 0; start <= ordered.length - minimum; start++) {
        for (let end = start + minimum - 1; end < ordered.length; end++) {
          if (ordered[end][0] - ordered[start][0] > MAX_SPREAD) break;
          const members = ordered.slice(start, end + 1).map(row => row[1]).sort((a, b) => a[0] - b[0]);
          const key = members.map(point => point[0]).join(",");
          if (seen.has(key)) continue;
          seen.add(key);
          const model = fit(members, limit);
          if (model.error > MAX_ERROR) continue;
          const along = model.along.sort((a, b) => a - b);
          const span = along.at(-1) - along[0];
          const spread = Math.max(...model.across) - Math.min(...model.across);
          if (spread > MAX_SPREAD || span < Math.max(MIN_SPAN, 3 * spread) ||
              along.some((value, i) => i > 0 && value - along[i - 1] < MIN_GAP)) continue;
          options.push({...candidate(members, model), weight: members.length - 1 + Math.min(span, 50) / 50 - .3 * model.error});
        }
      }
    }
    options.sort((a, b) => b.weight - a.weight);
    let best = [];
    let bestWeight = 0;
    for (const seed of options.slice(0, 12)) {
      const selection = [seed];
      let weight = seed.weight;
      for (const option of options) {
        if (selection.every(other => compatible(option, other))) {
          selection.push(option);
          weight += option.weight;
        }
      }
      if (weight > bestWeight) {
        best = selection;
        bestWeight = weight;
      }
    }
    return best;
  }

  function link(proposals, maxGap) {
    const episodes = [];
    let active = [];
    proposals.forEach((options, frame) => {
      active = active.filter(episode => frame - episode.observations.at(-1).frame <= maxGap);
      const matches = [];
      for (const episode of active) {
        const previous = episode.observations.at(-1);
        const gap = frame - previous.frame;
        options.forEach((option, index) => {
          const common = option.points.filter(point => previous.option.ids.has(point[0])).length;
          if (common < 2 || Math.abs(previous.option.angle - option.angle) > 9 + gap) return;
          const delta = distance(previous.option, option);
          if (delta <= 3.5 + .3 * gap) {
            matches.push({episode, index, delta, overlap: common / (option.ids.size + previous.option.ids.size - common)});
          }
        });
      }
      matches.sort((a, b) => b.overlap - a.overlap || a.delta - b.delta || b.episode.id - a.episode.id);
      const usedEpisodes = new Set();
      const usedOptions = new Set();
      for (const {episode, index} of matches) {
        if (usedEpisodes.has(episode) || usedOptions.has(index)) continue;
        episode.observations.push({frame, option: options[index]});
        usedEpisodes.add(episode);
        usedOptions.add(index);
      }
      options.forEach((option, index) => {
        if (usedOptions.has(index)) return;
        const episode = {id: episodes.length, observations: [{frame, option}]};
        episodes.push(episode);
        active.push(episode);
      });
    });
    return episodes;
  }

  function templates(episodes, frameCount, fps) {
    const result = Array.from({length: frameCount}, () => []);
    const minObservations = Math.max(3, Math.ceil(.48 * fps));
    const radius = Math.max(2, Math.round(.48 * fps));
    for (const {id, observations} of episodes) {
      const first = observations[0].frame;
      const last = observations.at(-1).frame;
      if (observations.length < minObservations || observations.length / (last - first + 1) < .65) continue;
      let start = 0;
      let end = 0;
      for (let frame = first; frame <= last; frame++) {
        while (observations[start].frame < frame - radius) start++;
        while (end < observations.length && observations[end].frame <= frame + radius) end++;
        const nearby = observations.slice(start, end);
        const votes = new Map();
        for (const {option} of nearby) {
          for (const ident of option.ids) votes.set(ident, (votes.get(ident) ?? 0) + 1);
        }
        for (const [ident, count] of votes) {
          if (count < Math.ceil(.4 * nearby.length)) votes.delete(ident);
          else votes.set(ident, count / nearby.length);
        }
        result[frame].push({id, votes, center: median(nearby.map(row => row.option.center)),
          slope: Math.tan(median(nearby.map(row => row.option.angle)) * radians)});
      }
    }
    return result;
  }

  function assign(points, rows, minimum, limit) {
    const groups = new Map(rows.map(row => [row.id, []]));
    for (const point of points) {
      const [ident, x, y] = point;
      let best;
      let highest = 0;
      let nearest = Infinity;
      for (const row of rows) {
        const vote = row.votes.get(ident) ?? 0;
        if (!vote || vote < highest) continue;
        const delta = Math.abs(y - row.center - (x - 50) * row.slope);
        if (vote > highest || delta < nearest) {
          best = row;
          highest = vote;
          nearest = delta;
        }
      }
      if (best) groups.get(best.id).push(point);
    }
    const candidates = [];
    for (const [id, members] of groups) {
      if (members.length < minimum) continue;
      const model = fit(members, limit);
      if (Math.max(...model.along) - Math.min(...model.along) >= MIN_SPAN) {
        candidates.push({...candidate(members, model), id, error: model.error});
      }
    }
    candidates.sort((a, b) => b.points.length - a.points.length || a.error - b.error);
    const distinct = [];
    for (const row of candidates) {
      if (distinct.every(other => compatible(row, other))) distinct.push(row);
    }
    return distinct.sort((a, b) => a.center - b.center);
  }

  function score(row, angle) {
    const model = fit(row.points, 0, angle);
    const {cos, sin, offset, along, errors, error} = model;
    const at = position => [position * cos - offset * sin, position * sin + offset * cos];
    return {
      id: row.id, angle, start: at(Math.min(...along)), end: at(Math.max(...along)),
      offset: error, score: Math.max(0, 100 * (1 - error / ZERO_SCORE_OFFSET)),
      members: row.points.map(([id, x, y], i) => [id, x, y, x + errors[i] * sin, y - errors[i] * cos]),
    };
  }

  function analyze(frames, fps, minimum, limit) {
    const measured = frames.map(points => points.filter(point => !point[3]));
    const proposals = measured.map(points => propose(points, minimum, limit));
    const episodes = link(proposals, Math.max(2, Math.round(.28 * fps)));
    const tracked = templates(episodes, frames.length, fps);
    const assigned = measured.map((points, frame) => assign(points, tracked[frame], minimum, limit));
    return assigned.map((rows, frame) => rows.map(row => {
      if (!limit || row.error < 1e-9) return score(row, row.angle);
      let sum = 3 * row.angle;
      let weight = 3;
      for (const direction of [-1, 1]) {
        for (let shift = 1; shift <= 2; shift++) {
          const neighbor = assigned[frame + direction * shift]?.find(other => other.id === row.id);
          if (!neighbor) break;
          sum += (3 - shift) * neighbor.angle;
          weight += 3 - shift;
        }
      }
      return score(row, Math.max(-limit, Math.min(limit, sum / weight)));
    }));
  }

  return {analyze};
}

globalThis.createFormationEngine = createFormationEngine;
