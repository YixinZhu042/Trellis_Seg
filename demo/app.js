(() => {
  "use strict";

  const VIEWER_URL = "https://unpkg.com/@google/model-viewer@4.0.0/dist/model-viewer.min.js";
  const DEFAULT_ORBIT = "35deg 70deg 105%";
  const DEFAULT_TARGET = "auto auto auto";
  const DEFAULT_FOV = "30deg";
  const BASE_URL = new URL(".", window.location.href);
  const $ = (id) => document.getElementById(id);
  const state = {
    samples: [], postById: new Map(), manifests: {}, mode: "models",
    filter: "all", selected: null, generation: 0,
    slots: {}, library: null, leader: "old", rotationFrame: null,
    rotationTime: 0, lastInteraction: 0, dragging: false,
  };

  function element(tag, className, content) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== undefined) node.textContent = content;
    return node;
  }

  function text(value, fallback = "") {
    return typeof value === "string" && value.trim() ? value.trim() : fallback;
  }

  // The manifest owns the content; it never supplies HTML or executable URLs.
  function assetURL(path) {
    if (typeof path !== "string" || !path.trim()) return null;
    const value = path.trim();
    if (/^(?:[a-z][a-z\d+.-]*:|[\\/])/i.test(value)) return null;
    try {
      const url = new URL(value, BASE_URL);
      if (url.origin !== BASE_URL.origin || !url.pathname.startsWith(BASE_URL.pathname)) return null;
      return url.href;
    } catch {
      return null;
    }
  }

  function setLink(id, path) {
    const anchor = $(id);
    const url = assetURL(path);
    anchor.hidden = !url;
    if (url) anchor.href = url;
    else anchor.removeAttribute("href");
    return url;
  }

  function describe(value, fallback) {
    if (typeof value === "string") return text(value, fallback);
    if (typeof value === "number" && Number.isFinite(value)) return String(value);
    if (Array.isArray(value)) return value.map((item) => describe(item, "")).filter(Boolean).join(" · ") || fallback;
    if (value && typeof value === "object") {
      return Object.entries(value).map(([key, item]) => {
        const description = describe(item, "");
        return description ? `${key.replace(/_/g, " ")}: ${description}` : "";
      }).filter(Boolean).join(" · ") || fallback;
    }
    return fallback;
  }

  function normalizeManifest(manifest, filename = "comparison.json") {
    if (!manifest || !Array.isArray(manifest.samples) || !manifest.samples.length) {
      throw new Error(`${filename} does not contain any samples yet.`);
    }
    const seen = new Set();
    const samples = manifest.samples.map((sample, index) => {
      if (!sample || !["string", "number"].includes(typeof sample.id)) {
        throw new Error(`Sample ${index + 1} needs a unique id.`);
      }
      const id = String(sample.id).trim();
      if (!id || seen.has(id)) throw new Error(`Sample ${index + 1} has an empty or duplicate id.`);
      if (!["original_demo", "character"].includes(sample.kind)) {
        throw new Error(`Sample ${id} has an unsupported kind.`);
      }
      seen.add(id);
      return { ...sample, id, title: text(sample.title, `Sample ${id}`), index };
    });
    return { ...manifest, samples, protocol: manifest.protocol || {}, release_date: manifest.release_date };
  }

  function showManifestStatus(title, detail, isError = false) {
    $("manifest-status").hidden = false;
    $("manifest-status").dataset.state = isError ? "error" : "loading";
    $("manifest-status-title").textContent = title;
    $("manifest-status-detail").textContent = detail;
    $("retry-manifest").hidden = !isError;
  }

  async function loadManifest() {
    if (window.location.protocol === "file:") {
      showManifestStatus("Open this demo through a local web server", "Browsers block the comparison.json request when this page is opened as a file. From the repository directory, run the command below, then open http://localhost:8000/demo/ in your browser.", true);
      $("serve-command").hidden = false;
      $("retry-manifest").hidden = true;
      return;
    }
    showManifestStatus("Loading the comparison manifests…", "Reading model reruns, postprocessed outputs, and available assets.");
    $("retry-manifest").disabled = true;
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 20000);
    try {
      const [modelResponse, postResponse] = await Promise.all([
        fetch(new URL("comparison.json", BASE_URL), { cache: "no-store", signal: controller.signal }),
        fetch(new URL("postprocess.json", BASE_URL), { cache: "no-store", signal: controller.signal }),
      ]);
      if (!modelResponse.ok) throw new Error(`comparison.json returned HTTP ${modelResponse.status}. The release manifest may still be uploading.`);
      if (!postResponse.ok) throw new Error(`postprocess.json returned HTTP ${postResponse.status}. The postprocessed assets may still be uploading.`);
      let manifest;
      let postManifest;
      try {
        manifest = normalizeManifest(await modelResponse.json(), "comparison.json");
        postManifest = normalizeManifest(await postResponse.json(), "postprocess.json");
      }
      catch (error) {
        if (error instanceof SyntaxError) throw new Error("A comparison manifest is not valid JSON. Please retry after the release is fully published.");
        throw error;
      }
      state.samples = manifest.samples;
      state.postById = new Map(postManifest.samples.map((sample) => [sample.id, sample]));
      const missing = state.samples.filter((sample) => !state.postById.has(sample.id));
      if (missing.length) throw new Error(`postprocess.json is missing ${missing.length} comparison sample(s).`);
      state.manifests = { models: manifest, post: postManifest };
      const params = new URLSearchParams(window.location.hash.slice(1));
      state.mode = params.get("mode") === "post" ? "post" : "models";
      renderProtocol();
      renderSampleList();
      $("workspace").hidden = false;
      $("mode-switch").hidden = false;
      $("manifest-status").hidden = true;
      const requested = params.get("sample");
      const first = state.samples.find((sample) => sample.id === requested) || state.samples[0];
      selectSample(first.id);
    } catch (error) {
      const detail = error.name === "AbortError" ? "The manifest request timed out. Check your connection and retry." : `${error.message} Serve this demo over HTTP and keep comparison.json and postprocess.json beside index.html.`;
      showManifestStatus("The comparison manifest is unavailable", detail, true);
    } finally {
      window.clearTimeout(timeout);
      $("retry-manifest").disabled = false;
    }
  }

  function renderProtocol() {
    const manifest = state.manifests.models;
    const postManifest = state.manifests.post;
    const protocol = manifest.protocol;
    const isPost = state.mode === "post";
    $("release-date").textContent = text(manifest.release_date, "Comparison release");
    if (isPost) {
      const postProtocol = postManifest.protocol;
      const settings = postManifest.settings || {};
      $("mode-title").textContent = "New model: raw vs postprocessed";
      $("protocol-old-title").textContent = "Raw source";
      $("protocol-new-title").textContent = "Cleanup rule";
      $("protocol-sampling-title").textContent = "Boundary support";
      $("protocol-label-title").textContent = "Information used";
      $("protocol-old").textContent = describe(postProtocol.input, "Released parts64 new-model raw face labels.");
      $("protocol-new").textContent = describe(postProtocol.method, "Geometry-only connected-component cleanup.");
      $("protocol-sampling").textContent = describe(postProtocol.component_weight, "Boundary support weighted by face area and normal agreement.");
      $("protocol-labels").textContent = `No GT, 2D condition, or label matching is used. Global threshold: ${(Number(settings.component_area_fraction || 0) * 100).toFixed(2)}% surface area; boundary dominance: ${(Number(settings.component_dominance || 0) * 100).toFixed(0)}%.`;
      $("protocol-condition-note").textContent = describe(postProtocol.selection_note, "One global setting is used for all examples.");
      $("metric-note").textContent = `${describe(postProtocol.metric_note, "Agreement is measured against the shared 2D input labels.")} Changes are postprocessed minus raw in percentage points.`;
      $("footer-manifest").href = "postprocess.json";
      $("footer-manifest").textContent = "View postprocess manifest ↗";
    } else {
      $("mode-title").textContent = "Old model vs new model";
      $("protocol-old-title").textContent = "Old checkpoint pair";
      $("protocol-new-title").textContent = "New checkpoint pair";
      $("protocol-sampling-title").textContent = "Sampling";
      $("protocol-label-title").textContent = "Condition labels";
      $("protocol-old").textContent = describe(protocol.old_model, "Legacy 20-label VAE + DiT EMA 50,000.");
      $("protocol-new").textContent = describe(protocol.new_model, "Parts64 VAE + matching DiT checkpoint pair.");
      const steps = describe(protocol.steps, "20");
      const seed = describe(protocol.seed, "42");
      $("protocol-sampling").textContent = `${steps} Euler steps · CFG 1 · seed ${seed} · no smoothing.`;
      $("protocol-labels").textContent = "Legacy condition IDs ≤ 19 are identical for both reruns. The old pair uses a 20-label VAE; the new pair uses a 64-label VAE.";
      $("protocol-condition-note").textContent = describe(protocol.condition_note, "Each old/new pair shares the same six-view 2D condition.");
      $("metric-note").textContent = describe(protocol.metric_note, "Agreement is computed against the shared 2D conditioning labels. Changes are new minus old in percentage points.");
      $("footer-manifest").href = "comparison.json";
      $("footer-manifest").textContent = "View model manifest ↗";
    }
    document.querySelectorAll("[data-mode]").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.mode === state.mode)));
    $("count-all").textContent = state.samples.length;
    $("count-original").textContent = state.samples.filter((sample) => sample.kind === "original_demo").length;
    $("count-characters").textContent = state.samples.filter((sample) => sample.kind === "character").length;
  }

  function filteredSamples() {
    return state.samples.filter((sample) => state.filter === "all" || sample.kind === state.filter);
  }

  function renderSampleList() {
    const visible = filteredSamples();
    const fragment = document.createDocumentFragment();
    visible.forEach((sample) => {
      const button = element("button", "sample-button");
      button.type = "button";
      button.dataset.id = sample.id;
      button.setAttribute("aria-pressed", String(sample.id === state.selected?.id));
      button.setAttribute("aria-controls", "comparison");
      const number = element("span", "sample-number", String(sample.index + 1).padStart(2, "0"));
      number.setAttribute("aria-hidden", "true");
      const copy = element("span", "sample-text");
      copy.append(element("span", "sample-name", sample.title), element("span", "sample-type", sample.kind === "character" ? "Character · external 2D labels" : "Original demo"));
      const current = element("span", "sample-current");
      current.setAttribute("aria-hidden", "true");
      button.append(number, copy, current);
      button.addEventListener("click", () => selectSample(sample.id));
      button.addEventListener("keydown", (event) => {
        const keys = ["ArrowDown", "ArrowUp", "ArrowRight", "ArrowLeft", "Home", "End"];
        if (!keys.includes(event.key)) return;
        event.preventDefault();
        const buttons = Array.from($("sample-list").querySelectorAll("button"));
        let index = buttons.indexOf(button);
        if (event.key === "Home") index = 0;
        else if (event.key === "End") index = buttons.length - 1;
        else index = (index + (["ArrowDown", "ArrowRight"].includes(event.key) ? 1 : -1) + buttons.length) % buttons.length;
        buttons[index].focus();
        selectSample(buttons[index].dataset.id);
      });
      fragment.append(button);
    });
    $("sample-list").replaceChildren(fragment);
    $("sample-count").textContent = `${visible.length} / ${state.samples.length}`;
    $("empty-filter").hidden = visible.length > 0;
    updateNavigation();
  }

  function updateNavigation() {
    const visible = filteredSamples();
    const index = visible.findIndex((sample) => sample.id === state.selected?.id);
    $("previous-sample").disabled = index <= 0;
    $("next-sample").disabled = index < 0 || index >= visible.length - 1;
    $("sample-position").textContent = index >= 0 ? `${index + 1} of ${visible.length}` : "";
    $("sample-list").querySelectorAll("button").forEach((button) => {
      const active = button.dataset.id === state.selected?.id;
      button.setAttribute("aria-pressed", String(active));
      if (active) button.setAttribute("aria-current", "true");
      else button.removeAttribute("aria-current");
    });
  }

  function moveSelection(delta) {
    const visible = filteredSamples();
    const index = visible.findIndex((sample) => sample.id === state.selected?.id);
    if (visible[index + delta]) selectSample(visible[index + delta].id);
  }

  function setMode(mode) {
    if (!state.manifests.models || !["models", "post"].includes(mode) || mode === state.mode) return;
    state.mode = mode;
    renderProtocol();
    if (state.selected) selectSample(state.selected.id);
  }

  function ratio(value) {
    return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1 ? value : null;
  }

  function comparisonData(sample) {
    if (state.mode === "post") {
      const post = state.postById.get(sample.id);
      return {
        leftPath: post?.raw_glb,
        rightPath: post?.post_glb,
        leftMetrics: post?.raw,
        rightMetrics: post?.post,
        comparisonImage: post?.comparison_image,
        post,
        leftName: "raw",
        rightName: "postprocessed",
      };
    }
    return {
      leftPath: sample.old_glb,
      rightPath: sample.new_glb,
      leftMetrics: sample.metrics?.old,
      rightMetrics: sample.metrics?.new,
      comparisonImage: sample.comparison_image,
      post: null,
      leftName: "old",
      rightName: "new",
    };
  }

  function renderMetrics(sample) {
    const data = comparisonData(sample);
    ["accuracy", "miou"].forEach((metric) => {
      const oldValue = ratio(data.leftMetrics?.[metric]);
      const newValue = ratio(data.rightMetrics?.[metric]);
      $(`old-${metric}`).textContent = oldValue === null ? "—" : `${(oldValue * 100).toFixed(1)}%`;
      $(`new-${metric}`).textContent = newValue === null ? "—" : `${(newValue * 100).toFixed(1)}%`;
      const change = $(`delta-${metric}`);
      if (oldValue === null || newValue === null) {
        change.textContent = "—";
        change.removeAttribute("title");
      } else {
        const delta = Math.round((newValue - oldValue) * 1000) / 10;
        change.textContent = `${delta > 0 ? "+" : delta < 0 ? "−" : ""}${Math.abs(delta).toFixed(1)}`;
        change.title = `${data.rightName} minus ${data.leftName}, in percentage points`;
      }
    });
    const post = data.post;
    $("postprocess-stats").hidden = !post;
    if (post) {
      const changedArea = ratio(post.changed_area_fraction);
      const before = Number.isInteger(post.components_before) ? post.components_before : null;
      const after = Number.isInteger(post.components_after) ? post.components_after : null;
      $("changed-area").textContent = changedArea === null ? "—" : `${(changedArea * 100).toFixed(3)}%`;
      $("component-change").textContent = before === null || after === null ? "—" : `${before.toLocaleString("en-US")} → ${after.toLocaleString("en-US")}`;
      $("component-reduction").textContent = before === null || after === null || before === 0 ? "—" : `${(((before - after) / before) * 100).toFixed(1)}%`;
    }
  }

  function renderImage(hostId, linkId, path, alt) {
    const host = $(hostId);
    const url = setLink(linkId, path);
    const message = element("p", "image-message", url ? "Loading image…" : "No image supplied for this sample.");
    message.setAttribute("role", "status");
    host.replaceChildren(message);
    if (!url) return;
    const image = element("img");
    image.alt = alt;
    image.decoding = "async";
    image.loading = "eager";
    image.hidden = true;
    // Detached images cannot replace the newly selected sample's state.
    image.addEventListener("load", () => {
      if (!host.contains(image)) return;
      image.hidden = false;
      message.remove();
    });
    image.addEventListener("error", () => {
      if (!host.contains(image)) return;
      message.textContent = "This image could not be loaded. It may still be uploading; use the full-size link to retry.";
      image.remove();
    });
    host.append(image);
    image.src = url;
  }

  function selectSample(id) {
    const sample = state.samples.find((item) => item.id === id);
    if (!sample) { updateNavigation(); return; }
    const data = comparisonData(sample);
    state.generation += 1;
    disposeViewers();
    state.selected = sample;
    state.leader = "old";
    state.dragging = false;
    $("sample-title").textContent = sample.title;
    $("sample-kind").textContent = sample.kind === "character" ? "Character" : "Original demo";
    $("sample-faces").textContent = Number.isInteger(sample.faces) && sample.faces >= 0 ? `${sample.faces.toLocaleString("en-US")} faces` : "Face count not supplied";
    const isPost = state.mode === "post";
    $("selection-announcement").textContent = `${sample.title} selected. Loading ${isPost ? "the raw and postprocessed outputs" : "the old and new reruns"}.`;
    $("old-label").textContent = isPost ? "New · raw" : "Old · rerun";
    $("old-heading").textContent = isPost ? "Parts64 raw output" : "Legacy 20-label pair";
    $("old-footer").textContent = isPost ? "Unmodified released prediction" : "DiT EMA 50,000 · 20-label VAE";
    $("download-old").textContent = isPost ? "Download raw GLB ↓" : "Download old GLB ↓";
    $("new-label").textContent = isPost ? "New · cleaned" : "New · rerun";
    $("new-heading").textContent = isPost ? "Geometry-only postprocess" : "Parts64 pair";
    $("new-footer").textContent = isPost ? "One global component + boundary rule" : "64-label VAE + matching DiT";
    $("download-new").textContent = isPost ? "Download postprocessed GLB ↓" : "Download new GLB ↓";
    $("metric-old-heading").textContent = isPost ? "Raw" : "Old rerun";
    $("metric-new-heading").textContent = isPost ? "Postprocessed" : "New rerun";
    $("metric-caption").textContent = `Agreement with the 2D input labels. Changes are ${data.rightName} minus ${data.leftName} in percentage points.`;
    setLink("download-old", data.leftPath);
    setLink("download-new", data.rightPath);
    setLink("download-archive", sample.kind === "original_demo" ? sample.archive_glb : null);
    $("archive-note").hidden = isPost;
    $("postprocess-note").hidden = !isPost;
    renderMetrics(sample);
    $("condition-provenance").textContent = sample.kind === "character"
      ? `External 2D API labels provide coarse part conditions (8 / 6 / 6 parts across the three characters). ${isPost ? "They are used only for the displayed agreement metrics, not by cleanup." : "Both reruns receive the same labels."} These are not ground-truth annotations.`
      : `${isPost ? "Cleanup does not read the legacy six-view condition; it is shown only for input-agreement metrics." : "Both reruns receive the same legacy six-view condition with IDs ≤ 19."} Parsed annotations for original demos 1–14 may be incorrect; the reported metrics measure only 2D input agreement.`;
    renderImage("condition-image-host", "open-condition", sample.condition_image, `${sample.title}: six-view 2D segmentation condition`);
    renderImage("comparison-image-host", "open-comparison", data.comparisonImage, `${sample.title}: supplied static ${isPost ? "raw/postprocessed" : "old/new"} comparison`);
    $("source-details").open = false;
    $("source-details").hidden = !assetURL(sample.source_image);
    $("source-image-host").replaceChildren();
    setLink("open-source", sample.source_image);
    const notes = Array.isArray(sample.notes) ? sample.notes.filter((note) => typeof note === "string" && note.trim()) : [];
    $("sample-notes").replaceChildren(...notes.map((note) => element("li", "", note)));
    $("sample-notes-panel").hidden = !notes.length;
    updateNavigation();
    const params = { sample: sample.id };
    if (state.mode === "post") params.mode = "post";
    const hash = `#${new URLSearchParams(params)}`;
    if (window.location.hash !== hash) window.history.replaceState(null, "", hash);
    mountViewer("old", data.leftPath, state.generation);
    mountViewer("new", data.rightPath, state.generation);
  }

  function ensureViewerLibrary() {
    if (customElements.get("model-viewer")) {
      customElements.get("model-viewer").modelCacheSize = 0;
      return Promise.resolve();
    }
    if (state.library) return state.library;
    state.library = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.type = "module";
      script.src = VIEWER_URL;
      let settled = false;
      let timeout;
      const finish = (error) => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timeout);
        script.onerror = null;
        if (error) { script.remove(); reject(error); }
        else { customElements.get("model-viewer").modelCacheSize = 0; resolve(); }
      };
      timeout = window.setTimeout(() => finish(new Error("The 3D viewer library took too long to load.")), 25000);
      script.onerror = () => finish(new Error("The pinned 3D viewer library could not be loaded from its CDN."));
      customElements.whenDefined("model-viewer").then(() => finish());
      document.head.append(script);
    }).catch((error) => { state.library = null; throw error; });
    return state.library;
  }

  function disposeSlot(slot) {
    if (!slot) return;
    slot.disposed = true;
    window.clearTimeout(slot.timeout);
    window.clearTimeout(slot.longTimeout);
    if (slot.viewer) {
      slot.viewer.removeAttribute("src");
      slot.viewer.remove();
    }
  }

  function disposeViewers() {
    Object.values(state.slots).forEach(disposeSlot);
    state.slots = {};
    stopRotation();
  }

  function setViewerStatus(side, label, status) {
    $(`${side}-state`).textContent = label;
    $(`${side}-state`).dataset.state = status;
  }

  function isCurrent(slot) {
    return !slot.disposed && state.slots[slot.side] === slot && slot.generation === state.generation;
  }

  async function mountViewer(side, path, generation) {
    disposeSlot(state.slots[side]);
    const host = $(`${side}-viewer-host`);
    const overlay = element("div", "viewer-placeholder");
    const ring = element("span", "loading-ring");
    ring.setAttribute("aria-hidden", "true");
    const heading = element("strong", "", "Preparing 3D viewer…");
    const detail = element("p", "", "Only this selected model pair will be loaded.");
    overlay.setAttribute("role", "status");
    overlay.append(ring, heading, detail);
    host.replaceChildren(overlay);
    host.setAttribute("aria-busy", "true");
    const slot = { side, generation, overlay, host, viewer: null, ready: false, disposed: false, timeout: null, longTimeout: null };
    state.slots[side] = slot;
    setViewerStatus(side, "Loading", "loading");
    const url = assetURL(path);
    const retry = () => { if (isCurrent(slot)) mountViewer(side, path, generation); };
    const fail = (message, allowRetry = true) => {
      if (!isCurrent(slot)) return;
      window.clearTimeout(slot.timeout);
      window.clearTimeout(slot.longTimeout);
      slot.ready = false;
      if (slot.viewer) { slot.viewer.removeAttribute("src"); slot.viewer.remove(); slot.viewer = null; }
      ring.hidden = true;
      heading.textContent = "3D preview unavailable";
      detail.textContent = message;
      overlay.hidden = false;
      host.setAttribute("aria-busy", "false");
      setViewerStatus(side, "Unavailable", "error");
      overlay.querySelectorAll("button, progress").forEach((node) => node.remove());
      if (allowRetry) {
        const button = element("button", "button", "Retry 3D preview");
        button.type = "button";
        button.addEventListener("click", retry);
        overlay.append(button);
      }
    };
    if (!url) { fail("No valid GLB path was supplied for this rerun.", false); return; }
    try { await ensureViewerLibrary(); }
    catch (error) { fail(`${error.message} The static comparison and GLB download are still available.`); return; }
    if (!isCurrent(slot)) return;

    const viewer = element("model-viewer");
    slot.viewer = viewer;
    const attributes = {
      id: `${side}-viewer`, alt: `${state.selected.title}, ${state.mode === "post" ? (side === "old" ? "raw new-model output" : "geometry-only postprocessed output") : `${side} model rerun`}. Drag to orbit; use arrow keys when focused.`,
      "camera-controls": "", "camera-orbit": DEFAULT_ORBIT, "camera-target": DEFAULT_TARGET,
      "field-of-view": DEFAULT_FOV, "environment-image": "neutral", "shadow-intensity": "0",
      exposure: "1", "tone-mapping": "neutral", "interaction-prompt": "none",
      loading: "eager", reveal: "auto", "touch-action": "pan-y", tabindex: "0",
    };
    Object.entries(attributes).forEach(([key, value]) => viewer.setAttribute(key, value));
    const progress = element("progress");
    progress.max = 1;
    progress.setAttribute("aria-label", `Loading ${side} model`);
    overlay.append(progress);
    heading.textContent = `Loading ${side} model…`;
    detail.textContent = "Large GLBs can take a moment to download and prepare.";
    viewer.addEventListener("progress", (event) => {
      if (!isCurrent(slot) || slot.ready) return;
      const value = event.detail?.totalProgress;
      if (typeof value === "number" && Number.isFinite(value)) {
        progress.value = value;
        setViewerStatus(side, value >= 1 ? "Preparing" : `${Math.round(value * 100)}%`, "loading");
      }
    });
    viewer.addEventListener("load", () => {
      if (!isCurrent(slot)) return;
      slot.ready = true;
      window.clearTimeout(slot.timeout);
      window.clearTimeout(slot.longTimeout);
      overlay.hidden = true;
      host.setAttribute("aria-busy", "false");
      setViewerStatus(side, "Ready", "ready");
      if (bothReady()) {
        if ($("sync-cameras").checked) syncCamera(state.leader);
        $("selection-announcement").textContent = `${state.selected.title}: both reruns are ready.`;
      }
      updateRotation();
    });
    viewer.addEventListener("error", () => fail("The GLB could not be loaded or rendered. Check the download link, then retry. The static comparison remains available."));
    viewer.addEventListener("camera-change", (event) => {
      if (!isCurrent(slot) || !slot.ready || event.detail?.source !== "user-interaction") return;
      state.lastInteraction = performance.now();
      state.leader = side;
      if ($("sync-cameras").checked) syncCamera(side);
    });
    viewer.addEventListener("pointerdown", () => { state.dragging = true; state.leader = side; });
    host.prepend(viewer);
    viewer.src = url;
    slot.timeout = window.setTimeout(() => {
      if (!isCurrent(slot) || slot.ready) return;
      heading.textContent = "Still loading this model…";
      detail.textContent = "This may be a large download. You can wait, use the GLB link, or retry this preview.";
      const button = element("button", "button", "Retry 3D preview");
      button.type = "button";
      button.addEventListener("click", retry);
      overlay.append(button);
    }, 45000);
    slot.longTimeout = window.setTimeout(() => fail("The 3D preview timed out after three minutes. Check your connection and retry, or use the GLB download."), 180000);
  }

  function bothReady() { return state.slots.old?.ready && state.slots.new?.ready; }

  function syncCamera(side) {
    if (!bothReady()) return;
    const from = state.slots[side].viewer;
    const to = state.slots[side === "old" ? "new" : "old"].viewer;
    const orbit = from.getCameraOrbit();
    const target = from.getCameraTarget();
    to.cameraOrbit = `${orbit.theta}rad ${orbit.phi}rad ${orbit.radius}m`;
    to.cameraTarget = `${target.x}m ${target.y}m ${target.z}m`;
    to.fieldOfView = `${from.getFieldOfView()}deg`;
    to.jumpCameraToGoal();
    // Programmatic camera-change events are ignored to avoid feedback loops.
  }

  function resetCamera() {
    state.leader = "old";
    state.lastInteraction = performance.now();
    Object.values(state.slots).forEach((slot) => {
      if (!slot.viewer) return;
      slot.viewer.cameraOrbit = DEFAULT_ORBIT;
      slot.viewer.cameraTarget = DEFAULT_TARGET;
      slot.viewer.fieldOfView = DEFAULT_FOV;
      slot.viewer.jumpCameraToGoal();
    });
    window.requestAnimationFrame(() => window.requestAnimationFrame(() => {
      if ($("sync-cameras").checked) syncCamera("old");
    }));
  }

  // Orbit the cameras rather than rotating either mesh, so a synced pair stays aligned.
  function rotateFrame(now) {
    state.rotationFrame = null;
    if (!$("auto-rotate").checked || document.hidden) return;
    const elapsed = state.rotationTime ? Math.min((now - state.rotationTime) / 1000, 0.05) : 0;
    state.rotationTime = now;
    if (!state.dragging && now - state.lastInteraction > 1200) {
      const slots = $("sync-cameras").checked && bothReady() ? [state.slots[state.leader]] : Object.values(state.slots);
      slots.forEach((slot) => {
        if (!slot.ready) return;
        const orbit = slot.viewer.getCameraOrbit();
        slot.viewer.cameraOrbit = `${orbit.theta + elapsed * 0.16}rad ${orbit.phi}rad ${orbit.radius}m`;
        slot.viewer.jumpCameraToGoal();
      });
      if ($("sync-cameras").checked) syncCamera(state.leader);
    }
    state.rotationFrame = window.requestAnimationFrame(rotateFrame);
  }

  function stopRotation() {
    if (state.rotationFrame !== null) window.cancelAnimationFrame(state.rotationFrame);
    state.rotationFrame = null;
    state.rotationTime = 0;
  }

  function updateRotation() {
    stopRotation();
    if ($("auto-rotate").checked && !document.hidden && Object.values(state.slots).some((slot) => slot.ready)) {
      state.rotationFrame = window.requestAnimationFrame(rotateFrame);
    }
  }

  document.querySelectorAll("[data-filter]").forEach((button) => {
    button.addEventListener("click", () => {
      state.filter = button.dataset.filter;
      document.querySelectorAll("[data-filter]").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
      renderSampleList();
      const visible = filteredSamples();
      if (visible.length && !visible.includes(state.selected)) selectSample(visible[0].id);
    });
  });
  document.querySelectorAll("[data-mode]").forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });
  $("previous-sample").addEventListener("click", () => moveSelection(-1));
  $("next-sample").addEventListener("click", () => moveSelection(1));
  $("retry-manifest").addEventListener("click", loadManifest);
  $("reset-camera").addEventListener("click", resetCamera);
  $("sync-cameras").addEventListener("change", () => { if ($("sync-cameras").checked) syncCamera(state.leader); });
  $("auto-rotate").addEventListener("change", updateRotation);
  $("source-details").addEventListener("toggle", () => {
    if ($("source-details").open && !$("source-image-host").childElementCount && state.selected) {
      renderImage("source-image-host", "open-source", state.selected.source_image, `${state.selected.title}: source reference image, not ground truth`);
    }
  });
  window.addEventListener("pointerup", () => { state.dragging = false; });
  window.addEventListener("pointercancel", () => { state.dragging = false; });
  window.addEventListener("blur", () => { state.dragging = false; });
  document.addEventListener("visibilitychange", updateRotation);
  window.addEventListener("hashchange", () => {
    const params = new URLSearchParams(window.location.hash.slice(1));
    const requestedMode = params.get("mode") === "post" ? "post" : "models";
    if (requestedMode !== state.mode) {
      state.mode = requestedMode;
      renderProtocol();
    }
    const id = params.get("sample");
    const sample = state.samples.find((item) => item.id === id);
    if (!sample) return;
    if (state.filter !== "all" && sample.kind !== state.filter) {
      state.filter = "all";
      document.querySelectorAll("[data-filter]").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.filter === "all")));
      renderSampleList();
    }
    selectSample(sample.id);
  });
  loadManifest();
})();
