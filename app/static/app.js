"use strict";
const $ = (s) => document.querySelector(s);
const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };

const STAGES = [
  { p: 6,  label: "Preparing image" },
  { p: 16, label: "Loading specialists" },
  { p: 45, label: "Detecting poles & components" },
  { p: 82, label: "Rendering annotated views" },
  { p: 94, label: "Building report" },
];

let selectedFile = null;
let pollTimer = null;
let previewURL = null;   // object URLs we must revoke to avoid leaking blobs
let thumbURL = null;

function showView(id) {
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  $("#" + id).classList.add("active");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* ---------- health / device badge ---------- */
async function loadHealth() {
  try {
    const h = await (await fetch("/api/health")).json();
    $("#deviceText").textContent = (h.device || "cpu").toUpperCase();
    const grid = $("#modelsGrid");
    grid.innerHTML = "";
    Object.entries(h.weights).forEach(([name, on]) => {
      const item = el("div", "model-item");
      item.appendChild(el("span", "pip " + (on ? "on" : "off")));
      item.appendChild(el("span", null, name));
      grid.appendChild(item);
    });
    $("#modelsCard").hidden = false;
    if (!h.ready) {
      showError(`No pole weights found under "${h.weights_dir}". Set DRONISIGHT_WEIGHTS to your trained runs/ folder and restart the server.`);
    }
  } catch (e) {
    $("#deviceText").textContent = "?";
  }
}

/* ---------- upload ---------- */
function setFile(file) {
  if (!file) return;
  selectedFile = file;
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = URL.createObjectURL(file);
  const prev = $("#dzPreview");
  prev.src = previewURL; prev.hidden = false;
  $("#dropzone").querySelector(".dz-inner").style.display = "none";
  $("#analyzeBtn").disabled = false;
  $("#clearBtn").hidden = false;
  $("#uploadError").hidden = true;
}
function clearFile() {
  selectedFile = null;
  $("#fileInput").value = "";
  if (previewURL) { URL.revokeObjectURL(previewURL); previewURL = null; }
  $("#dzPreview").hidden = true;
  $("#dropzone").querySelector(".dz-inner").style.display = "";
  $("#analyzeBtn").disabled = true;
  $("#clearBtn").hidden = true;
}
function showError(msg) {
  const b = $("#uploadError"); b.textContent = msg; b.hidden = false;
}

function wireUpload() {
  const dz = $("#dropzone"), input = $("#fileInput");
  input.addEventListener("change", () => setFile(input.files[0]));
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => { if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]); });
  $("#clearBtn").addEventListener("click", (e) => { e.preventDefault(); clearFile(); });
  $("#analyzeBtn").addEventListener("click", startAnalysis);
  $("#againBtn").addEventListener("click", () => { clearFile(); showView("view-upload"); });
}

/* ---------- analysis + polling ---------- */
async function startAnalysis() {
  if (!selectedFile) return;
  if (thumbURL) URL.revokeObjectURL(thumbURL);
  thumbURL = URL.createObjectURL(selectedFile);
  $("#loadingThumb").src = thumbURL;
  renderStageList(0);
  $("#loadingStage").textContent = "Uploading…";
  setProgress(0);
  showView("view-loading");
  try {
    const fd = new FormData();
    fd.append("file", selectedFile);
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
    const { job_id } = await res.json();
    poll(job_id);
  } catch (e) {
    failToUpload("Could not start analysis: " + e.message);
  }
}

function poll(jobId) {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    let job;
    try { job = await (await fetch(`/api/jobs/${jobId}`)).json(); }
    catch { return; }
    setProgress(job.percent || 0);
    if (job.stage) $("#loadingStage").textContent = job.stage;
    renderStageList(job.percent || 0);
    if (job.status === "done") {
      clearInterval(pollTimer);
      const report = await (await fetch(`/api/jobs/${jobId}/result`)).json();
      renderReport(report);
      showView("view-report");
    } else if (job.status === "error") {
      clearInterval(pollTimer);
      failToUpload("Analysis failed: " + (job.error || "unknown error"));
    }
  }, 600);
}

function failToUpload(msg) { showError(msg); showView("view-upload"); }
function setProgress(p) { $("#progressBar").style.width = p + "%"; $("#progressPct").textContent = p; }
function renderStageList(pct) {
  const ul = $("#stageList"); ul.innerHTML = "";
  STAGES.forEach((s, i) => {
    const next = STAGES[i + 1] ? STAGES[i + 1].p : 100;
    const li = el("li", null, s.label);
    if (pct >= next) li.classList.add("done");
    else if (pct >= s.p) li.classList.add("active");
    ul.appendChild(li);
  });
}

/* ---------- report rendering ---------- */
function chip(n, label, bad) {
  const c = el("div", "chip" + (bad ? " bad" : ""));
  c.appendChild(el("div", "n", String(n)));
  c.appendChild(el("div", "l", label));
  return c;
}

function renderReport(r) {
  $("#repTitle").textContent = r.image;
  $("#repMeta").textContent = `device: ${r.device.toUpperCase()} · ${r.summary.poles} pole(s) · ${r.summary.components} component(s)`;
  $("#dlCsv").href = r.downloads.csv || "#";
  $("#dlJson").href = r.downloads.json || "#";

  const chips = $("#summaryChips"); chips.innerHTML = "";
  chips.appendChild(chip(r.summary.poles, "Poles"));
  chips.appendChild(chip(r.summary.components, "Components"));
  chips.appendChild(chip(r.summary.attention, "Attention", r.summary.attention > 0));
  const topClass = Object.entries(r.summary.class_counts).sort((a, b) => b[1] - a[1])[0];
  if (topClass) chips.appendChild(chip(topClass[1], topClass[0]));

  // viz layer tabs
  const tabs = $("#layerTabs"); tabs.innerHTML = "";
  const order = r.layers.filter((l) => r.viz[l]);
  const defaultLayer = order.includes("all") ? "all" : order[0];
  order.forEach((layer) => {
    const t = el("div", "tab" + (layer === defaultLayer ? " active" : ""), layer);
    t.addEventListener("click", () => {
      tabs.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
      t.classList.add("active");
      $("#vizImg").src = r.viz[layer];
    });
    tabs.appendChild(t);
  });
  $("#vizImg").src = r.viz[defaultLayer] || "";

  // attention list
  const att = $("#attentionList"); att.innerHTML = "";
  if (!r.attention_items.length) {
    att.appendChild(el("div", "att-empty", "No defects flagged — all detected conditions look normal."));
  } else {
    r.attention_items.forEach((a) => {
      const it = el("div", "att-item");
      if (a.crop_url) { const im = el("img"); im.src = a.crop_url; it.appendChild(im); }
      const box = el("div");
      box.appendChild(el("div", "att-cls", a.component + (a.condition ? " · " + a.condition : "")));
      box.appendChild(el("div", "muted", "pole " + a.pole));
      it.appendChild(box);
      att.appendChild(it);
    });
  }

  // pole cards
  const pc = $("#polesContainer"); pc.innerHTML = "";
  r.poles.forEach((pole) => {
    const block = el("div", "pole-block");
    const title = el("div", "pole-title");
    title.appendChild(el("span", null, `Pole ${pole.index}`));
    title.appendChild(el("span", "muted", `conf ${pole.confidence} · ${pole.components.length} component(s)`));
    block.appendChild(title);
    const grid = el("div", "comp-grid");
    if (!pole.components.length) grid.appendChild(el("div", "muted", "No components detected on this pole."));
    pole.components.forEach((c) => {
      const card = el("div", "comp-card" + (c.attention ? " attn" : ""));
      if (c.crop_url) { const im = el("img"); im.src = c.crop_url; im.alt = c.class; card.appendChild(im); }
      const meta = el("div", "comp-meta");
      const cc = el("div", "cc"); cc.appendChild(el("span", null, c.class));
      cc.appendChild(el("span", "conf", c.confidence)); meta.appendChild(cc);
      if (!c.has_condition_family) {
        meta.appendChild(el("div", "badge neutral", "no condition family"));
      } else if (c.conditions && c.conditions.length) {
        // multi-label: show every detected condition (e.g. broken AND chip_off), green=normal/red=defect
        c.conditions.forEach((cond) =>
          meta.appendChild(el("div", "badge " + (cond.defect ? "bad" : "ok"),
            `${cond.class} ${cond.confidence}`)));
      } else {
        meta.appendChild(el("div", "badge neutral", "no condition detected"));
      }
      card.appendChild(meta);
      grid.appendChild(card);
    });
    block.appendChild(grid);
    pc.appendChild(block);
  });

  // table
  const tb = $("#reportTable").querySelector("tbody"); tb.innerHTML = "";
  let i = 1;
  r.poles.forEach((pole) => {
    if (!pole.components.length) return;
    pole.components.forEach((c) => {
      const tr = el("tr");
      const hasConds = c.conditions && c.conditions.length;
      const cells = [
        i++, pole.index, c.class, c.confidence,
        hasConds ? c.conditions.map((x) => x.class).join(", ") : (c.has_condition_family ? "none" : "—"),
        hasConds ? c.conditions.map((x) => x.confidence).join(", ") : "—",
        c.box_full.join(", "),
      ];
      cells.forEach((v) => tr.appendChild(el("td", null, String(v))));
      const st = el("td");
      const pill = el("span", "status-pill " + (c.attention ? "bad" : "ok"), c.attention ? "attention" : "ok");
      st.appendChild(pill); tr.appendChild(st);
      tb.appendChild(tr);
    });
  });
}

/* ---------- boot ---------- */
wireUpload();
loadHealth();
