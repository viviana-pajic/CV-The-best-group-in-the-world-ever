(() => {
  "use strict";

  const byId = id => document.getElementById(id);
  const video = byId("video");
  const videoView = byId("video-view");
  const overlay = byId("video-overlay");
  const map = byId("stage-map");
  const heatmapCanvas = byId("stage-heatmap");
  const heatmapTime = byId("heatmap-time");
  const heatmapCoverage = byId("heatmap-coverage");
  const heatmapSquares = byId("heatmap-squares");
  const play = byId("play");
  const previous = byId("previous");
  const next = byId("next");
  const timeline = byId("timeline");
  const minimum = byId("minimum");
  const angle = byId("angle");
  const select = byId("clip-select");
  const status = byId("analysis-status");
  const mapDimensions = byId("map-dimensions");
  const lineRows = byId("line-rows");
  const showBoxes = byId("show-boxes");
  const showFloor = byId("show-floor");
  const showContacts = byId("show-contacts");
  const showLines = byId("show-lines");
  const mediaError = byId("media-error");
  const colors = ["#8becbf", "#c3afff", "#ffcb82", "#8dd9ff", "#ffa9c7", "#e9e58d"];
  const contactColor = "#c9dff9";
  const cache = new Map();
  const viewedFrames = new Map();
  const clips = window.DANCE_CLIPS ?? [];
  const heatmapRaster = document.createElement("canvas");
  const heatmapPixels = heatmapRaster.getContext("2d");
  const heatColors = [
    [0, 17, 28, 44], [.18, 38, 57, 131], [.42, 50, 103, 199],
    [.68, 83, 197, 187], [.86, 161, 230, 178], [1, 255, 226, 151],
  ];
  let clip;
  let index = 0;
  let pendingSeek = null;
  let analyzed = null;
  let computing = false;
  let generation = 0;
  let analysisTimer;
  let worker;
  let latestJob;
  let frameCallback;
  let lastLineMarkup = "";
  let heatmap;
  let heatmapImage;
  let heatmapFrame = -1;
  let lastHeatmapDraw = 0;

  function rangeFill(input) {
    const fill = (input.valueAsNumber - Number(input.min)) / (Number(input.max) - Number(input.min));
    input.style.setProperty("--fill", `${100 * fill}%`);
  }

  function time(seconds) {
    const hundredths = Math.round(seconds * 100);
    const pad = value => String(value).padStart(2, "0");
    return `${pad(Math.floor(hundredths / 6000))}:${pad(Math.floor(hundredths / 100) % 60)}.${pad(hundredths % 100)}`;
  }

  function surface(canvas) {
    const {width, height} = canvas.getBoundingClientRect();
    const scale = window.devicePixelRatio || 1;
    const w = Math.round(width * scale);
    const h = Math.round(height * scale);
    if (!w || !h) return null;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(w / width, 0, 0, h / height, 0, 0);
    ctx.clearRect(0, 0, width, height);
    return {ctx, width, height};
  }

  function path(ctx, a, b) {
    ctx.beginPath();
    ctx.moveTo(...a);
    ctx.lineTo(...b);
    ctx.stroke();
  }

  function drawHeatmap(force = false) {
    if (!heatmap) return;
    const now = performance.now();
    if (!force && (index === heatmapFrame || (!video.paused && now - lastHeatmapDraw < 120))) return;
    const view = surface(heatmapCanvas);
    if (!view) return;
    const {ctx, width, height} = view;
    const {density, peak, visited, coverageSide} = heatmap.at(index);
    for (let i = 0; i < density.length; i++) {
      const amount = peak ? Math.sqrt(density[i] / peak) : 0;
      let stop = 1;
      while (stop < heatColors.length - 1 && amount > heatColors[stop][0]) stop++;
      const a = heatColors[stop - 1];
      const b = heatColors[stop];
      const t = (amount - a[0]) / (b[0] - a[0]);
      const pixel = i * 4;
      for (let c = 0; c < 3; c++) heatmapImage.data[pixel + c] = a[c + 1] + t * (b[c + 1] - a[c + 1]);
      heatmapImage.data[pixel + 3] = 255;
    }
    heatmapPixels.putImageData(heatmapImage, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(heatmapRaster, 0, 0, width, height);
    ctx.lineWidth = 1;
    for (let i = 1; i < coverageSide; i++) {
      const x = width * i / coverageSide;
      const y = height * i / coverageSide;
      ctx.strokeStyle = i % 2 === 0 ? "#e8f6ff30" : "#e8f6ff16";
      path(ctx, [x, 0], [x, height]);
      path(ctx, [0, y], [width, y]);
    }
    const total = coverageSide ** 2;
    heatmapTime.textContent = `Through ${time(index / clip.fps)}`;
    heatmapCoverage.textContent = `${(100 * visited / total).toFixed(1)}%`;
    heatmapSquares.textContent = `${visited} of ${total} stage squares visited`;
    heatmapFrame = index;
    lastHeatmapDraw = now;
  }

  function point(ctx, x, y, size, color, estimated) {
    ctx.beginPath();
    ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.strokeStyle = "#0b1220";
    ctx.lineWidth = estimated ? 3 : 2;
    ctx.stroke();
    if (estimated) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    } else {
      ctx.fillStyle = color;
      ctx.fill();
    }
  }

  function paintVideo(boxes, points, guides, memberColors) {
    const view = surface(overlay);
    if (!view) return;
    const {ctx, width, height} = view;
    const sx = width / clip.resolution[0];
    const sy = height / clip.resolution[1];
    const xy = ([x, y]) => [x * sx, y * sy];

    if (showFloor.checked) {
      for (const [label, a, b] of guides) {
        const boundary = label === "mat_boundary";
        ctx.strokeStyle = boundary ? "#8dd9ffe6" : "#8dd9ff9e";
        ctx.lineWidth = boundary ? 1.45 : 1;
        ctx.setLineDash(boundary ? [] : [4, 3]);
        path(ctx, xy(a), xy(b));
      }
      ctx.setLineDash([]);
    }
    if (showBoxes.checked) {
      ctx.font = '600 10px "Cascadia Mono", Consolas, monospace';
      for (const [id, x1, y1, x2, y2, predicted] of boxes) {
        const [x, y] = xy([x1, y1]);
        const color = memberColors.get(id) ?? "#a9bced";
        ctx.globalAlpha = predicted ? .5 : .86;
        ctx.setLineDash(predicted ? [4, 3] : []);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, (x2 - x1) * sx, (y2 - y1) * sy);
        const label = String(id);
        ctx.fillStyle = "#111929e8";
        ctx.fillRect(x, y - 13, ctx.measureText(label).width + 6, 13);
        ctx.fillStyle = color;
        ctx.fillText(label, x + 3, y - 3);
      }
      ctx.globalAlpha = 1;
      ctx.setLineDash([]);
    }
    if (showContacts.checked) {
      for (const [id, , , estimated, u, v] of points) {
        point(ctx, u * sx, v * sy, estimated ? 3.6 : 3.8, memberColors.get(id) ?? contactColor, estimated);
      }
    }
  }

  function paintMap(points, lines, memberColors) {
    const view = surface(map);
    if (!view) return;
    const {ctx, width, height} = view;
    const pad = 8;
    const scale = Math.min((width - pad * 2) / clip.stage[0], (height - pad * 2) / clip.stage[1]);
    const stageWidth = clip.stage[0] * scale;
    const stageDepth = clip.stage[1] * scale;
    const left = (width - stageWidth) / 2;
    const top = (height - stageDepth) / 2;
    const xy = ([x, y]) => [left + x * scale, top + y * scale];
    const background = ctx.createLinearGradient(0, 0, width, height);
    background.addColorStop(0, "#152837");
    background.addColorStop(1, "#142033");
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "#52698072";
    ctx.lineWidth = 1;
    ctx.strokeRect(left, top, stageWidth, stageDepth);
    ctx.strokeStyle = "#58708837";
    const cell = Math.min(stageWidth, stageDepth) / 6;
    for (let i = 1; i < stageWidth / cell - 1e-6; i++) {
      const x = left + cell * i;
      path(ctx, [x, top], [x, top + stageDepth]);
    }
    for (let i = 1; i < stageDepth / cell - 1e-6; i++) {
      const y = top + cell * i;
      path(ctx, [left, y], [left + stageWidth, y]);
    }

    if (showLines.checked) {
      lines.forEach((row, i) => {
        const color = colors[i % colors.length];
        ctx.strokeStyle = color;
        ctx.globalAlpha = .53;
        ctx.lineWidth = 1;
        for (const [, x, y, u, v] of row.members) {
          path(ctx, xy([x, y]), xy([u, v]));
        }
        ctx.globalAlpha = .18;
        ctx.lineWidth = 6;
        ctx.lineCap = "round";
        path(ctx, xy(row.start), xy(row.end));
        ctx.globalAlpha = .95;
        ctx.lineWidth = 2;
        path(ctx, xy(row.start), xy(row.end));
        ctx.globalAlpha = 1;
        ctx.lineCap = "butt";
      });
    }
    for (const [id, x, y, estimated] of points) {
      point(ctx, ...xy([x, y]), estimated ? 3.8 : 4.15, memberColors.get(id) ?? contactColor, estimated);
    }
  }

  function paintMetrics(lines, totalPoints, hasStage) {
    const label = !hasStage ? "Stage alignment unavailable" : computing ? "Updating lines…" :
      `${lines.length} ${lines.length === 1 ? "line" : "lines"} · ${totalPoints} dots`;
    if (status.textContent !== label) status.textContent = label;
    status.setAttribute("aria-busy", String(computing));
    const overall = lines.reduce((sum, row) => sum + row.score, 0) / lines.length;
    byId("alignment").textContent = lines.length ? `${overall.toFixed(1)}%` : "—";
    const empty = byId("empty-lines");
    empty.textContent = hasStage ? "No lines meet these settings at this frame." :
      "The stage could not be aligned at this frame.";
    empty.hidden = (hasStage && computing) || lines.length > 0;
    const markup = lines.map((row, i) => {
      const degrees = Math.abs(row.angle) < .05 ? "0.0°" : `${row.angle > 0 ? "+" : "−"}${Math.abs(row.angle).toFixed(1)}°`;
      const offset = clip.stageMeters ? `${(row.offset * clip.stageMeters[0] / 100).toFixed(2)} m` :
        `${row.offset.toFixed(2)}% of stage width`;
      return `<tr><td><i class="line-key" style="--line-color:${colors[i % colors.length]}"></i>${i + 1}</td>` +
        `<td>${row.members.length}</td><td>${degrees}</td><td title="Mean offset: ${offset}">${row.score.toFixed(1)}%</td></tr>`;
    }).join("");
    if (markup !== lastLineMarkup) {
      lineRows.innerHTML = markup;
      lastLineMarkup = markup;
    }
  }

  function draw() {
    if (!clip) return;
    const [boxes, points, guides = clip.guides] = clip.frames[index];
    const lines = analyzed?.[index] ?? [];
    const memberColors = new Map();
    if (showLines.checked) {
      lines.forEach((row, i) => {
        for (const [id] of row.members) memberColors.set(id, colors[i % colors.length]);
      });
    }
    paintVideo(boxes, points, guides, memberColors);
    paintMap(points, lines, memberColors);
    paintMetrics(lines, points.length, guides.length > 0);
  }

  function updateTransport(target) {
    const displayed = pendingSeek ?? target;
    timeline.value = displayed;
    rangeFill(timeline);
    timeline.setAttribute("aria-valuetext", time(displayed / clip.fps));
    byId("position").textContent = time(displayed / clip.fps);
    previous.disabled = displayed === 0;
    next.disabled = displayed === clip.frames.length - 1;
  }

  function displayFrame(target, force = false) {
    if (target === index && !force) return;
    index = target;
    byId("source-frame").textContent = `Frame ${(clip.firstFrame + index).toLocaleString("en-US")}`;
    updateTransport(index);
    draw();
    drawHeatmap(force);
  }

  function frameAt(seconds) {
    return Math.max(0, Math.min(clip.frames.length - 1, Math.floor(seconds * clip.fps + .02)));
  }

  function seek(target) {
    if (!clip) return;
    const frame = Math.max(0, Math.min(clip.frames.length - 1, target));
    pendingSeek = frame;
    updateTransport(frame);
    if (video.readyState === 0) return;
    const timestamp = (frame + .1) / clip.fps;
    if (Math.abs(video.currentTime - timestamp) > .0001 || video.seeking) {
      video.currentTime = timestamp;
    } else {
      pendingSeek = null;
      displayFrame(frame, true);
    }
  }

  function step(by) {
    video.pause();
    seek((pendingSeek ?? index) + by);
  }

  function togglePlayback() {
    if (!clip || video.readyState === 0) return;
    if (!video.paused) {
      video.pause();
      return;
    }
    if (video.ended) seek(0);
    video.play().catch(() => updatePlayback());
  }

  function updatePlayback() {
    const paused = video.paused;
    byId("play-icon").setAttribute("href", paused ? "#i-play" : "#i-pause");
    play.setAttribute("aria-label", paused ? "Play" : "Pause");
  }

  function watchPlayback() {
    if (frameCallback !== undefined) return;
    const hasVideoCallback = typeof video.requestVideoFrameCallback === "function";
    function tick(_, metadata) {
      frameCallback = undefined;
      if (clip && !video.seeking && pendingSeek === null) {
        const timestamp = metadata?.mediaTime ?? video.currentTime;
        const frame = frameAt(timestamp);
        if (!video.paused || !metadata || frame === frameAt(video.currentTime)) displayFrame(frame);
      }
      if (!video.paused) watchPlayback();
    }
    frameCallback = hasVideoCallback ? video.requestVideoFrameCallback(tick) : requestAnimationFrame(tick);
  }

  function receive({request, key, result}) {
    if (request !== generation) return;
    if (cache.size >= 8) cache.delete(cache.keys().next().value);
    cache.set(key, result);
    analyzed = result;
    computing = false;
    draw();
  }

  function fallback(job) {
    setTimeout(() => {
      if (job.request !== generation) return;
      const points = clip.frames.map(frame => frame[1]);
      receive({...job, result: createFormationEngine().analyze(points, clip.fps, job.minimum, job.limit)});
    }, 20);
  }

  function newWorker() {
    const code = `${createFormationEngine.toString()}
      const engine = createFormationEngine();
      let points, fps;
      self.onmessage = ({data}) => {
        if (data.points) ({points, fps} = data);
        else self.postMessage({...data, result: engine.analyze(points, fps, data.minimum, data.limit)});
      };`;
    const url = URL.createObjectURL(new Blob([code], {type: "text/javascript"}));
    try {
      const instance = new Worker(url);
      instance.onmessage = event => receive(event.data);
      instance.onerror = event => {
        event.preventDefault();
        instance.terminate();
        if (worker === instance) {
          worker = undefined;
          if (latestJob?.request === generation) fallback(latestJob);
        }
      };
      return instance;
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  function scheduleAnalysis(delay = 130) {
    clearTimeout(analysisTimer);
    const count = minimum.valueAsNumber;
    const limit = angle.valueAsNumber;
    byId("minimum-value").value = count;
    byId("angle-value").value = limit ? `±${limit}°` : "0°";
    minimum.setAttribute("aria-valuetext", `${count} dots`);
    angle.setAttribute("aria-valuetext", limit ? `Up to ${limit} degrees from horizontal` : "Horizontal only");
    rangeFill(minimum);
    rangeFill(angle);
    if (!clip) return;
    const key = `${clip.id}/${count}/${limit}`;
    const request = ++generation;
    analyzed = cache.get(key) ?? null;
    computing = !analyzed;
    draw();
    if (!computing) return;
    analysisTimer = setTimeout(() => {
      latestJob = {request, key, minimum: count, limit};
      if (worker) worker.postMessage(latestJob);
      else fallback(latestJob);
    }, delay);
  }

  function loadClip(selected) {
    if (selected === clip) return;
    if (clip) viewedFrames.set(clip.id, pendingSeek ?? index);
    video.pause();
    worker?.terminate();
    worker = undefined;
    clip = selected;
    analyzed = null;
    heatmap = createStageHeatmap(clip);
    heatmapRaster.width = heatmapRaster.height = heatmap.side;
    heatmapImage = heatmapPixels.createImageData(heatmap.side, heatmap.side);
    heatmapCanvas.style.aspectRatio = clip.stage.join(" / ");
    mediaError.hidden = true;
    timeline.max = clip.frames.length - 1;
    videoView.style.aspectRatio = clip.resolution.join(" / ");
    map.style.aspectRatio = clip.stage.join(" / ");
    const dimensions = clip.stageMeters?.join(" × ");
    mapDimensions.textContent = dimensions ? `${dimensions} m` : "";
    mapDimensions.hidden = !dimensions;
    map.setAttribute("aria-label", dimensions ?
      `Top-down map of the ${dimensions} m stage with dancer contacts and fitted lines` :
      "Top-down map of dancer contacts and fitted lines");
    video.poster = clip.poster;
    video.src = clip.video;
    const start = viewedFrames.get(clip.id) ?? clip.startFrame;
    pendingSeek = start;
    byId("duration").textContent = time(clip.frames.length / clip.fps);
    byId("replay-meta").textContent = `${clip.fps} fps · Contact threshold ${clip.threshold.toFixed(2)}`;
    try {
      worker = newWorker();
      worker.postMessage({points: clip.frames.map(frame => frame[1]), fps: clip.fps});
    } catch {
      worker = undefined;
    }
    displayFrame(start, true);
    scheduleAnalysis(0);
  }

  play.addEventListener("click", togglePlayback);
  video.addEventListener("click", togglePlayback);
  previous.addEventListener("click", () => step(-1));
  next.addEventListener("click", () => step(1));
  timeline.addEventListener("input", () => {
    video.pause();
    seek(timeline.valueAsNumber);
  });
  video.addEventListener("loadedmetadata", () => seek(pendingSeek ?? clip.startFrame));
  video.addEventListener("seeked", () => {
    pendingSeek = null;
    displayFrame(frameAt(video.currentTime), true);
  });
  video.addEventListener("play", () => {
    updatePlayback();
    watchPlayback();
  });
  video.addEventListener("pause", () => {
    updatePlayback();
    drawHeatmap(true);
  });
  video.addEventListener("ended", () => {
    updatePlayback();
    displayFrame(clip.frames.length - 1, true);
  });
  video.addEventListener("error", () => {
    mediaError.textContent = "Couldn’t open the replay video. Keep the dashboard and its assets folder together.";
    mediaError.hidden = false;
  });
  for (const input of [showBoxes, showFloor, showContacts, showLines]) input.addEventListener("change", draw);
  for (const input of [minimum, angle]) input.addEventListener("input", () => scheduleAnalysis());
  select.addEventListener("change", () => loadClip(clips.find(item => item.id === select.value)));
  document.addEventListener("keydown", event => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey ||
        event.target.closest("input, select, button, textarea, summary, [contenteditable]")) return;
    if (event.code === "Space" && !event.repeat) {
      event.preventDefault();
      togglePlayback();
    } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      step(event.key === "ArrowLeft" ? -1 : 1);
    }
  });
  const observer = new ResizeObserver(draw);
  observer.observe(videoView);
  observer.observe(map);
  new ResizeObserver(() => drawHeatmap(true)).observe(heatmapCanvas);

  for (const item of clips) select.add(new Option(item.title, item.id));
  if (clips.length) loadClip(clips[0]);
  else {
    mediaError.textContent = "The dashboard could not find any replay assets.";
    mediaError.hidden = false;
  }
})();
