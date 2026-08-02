// Vanilla JS dashboard: calls the 5 same-origin APIs (session cookie sent automatically).
let batchId = null;
let poll = null;
let busy = false;   // true while an upload/batch is in flight — blocks double-submit
const $ = (id) => document.getElementById(id);
const esc = (v) => String(v).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// in-page toast (no browser alert)
function toast(msg, kind = "info") {
  let t = document.getElementById("toast");
  if (!t) { t = document.createElement("div"); t.id = "toast"; document.body.appendChild(t); }
  t.textContent = msg;
  t.className = "toast show " + kind;
  clearTimeout(t._timer);
  t._timer = setTimeout(() => { t.className = "toast"; }, 3500);
}
const FIELDS = [
  "emotional_tone", "emotional_intensity", "background_noise_present", "background_noise_type",
  "background_noise_severity", "audio_quality", "speaker_overlap_present",
  "long_silence_present", "confidence",
];
// Provenance of each field: model = LLM-predicted (paid) · server = local DSP ($0) ·
// hybrid = LLM's value when it reports one, else the DSP fallback · blend = combined signal.
const FIELD_SRC = {
  emotional_tone: "model", emotional_intensity: "model",
  background_noise_present: "server", background_noise_severity: "server",
  audio_quality: "server", long_silence_present: "server",
  background_noise_type: "hybrid", speaker_overlap_present: "hybrid",
  confidence: "blend",
};

function setBusy(on) {
  busy = on;
  const b = $("uploadBtn");
  b.disabled = on;
  b.textContent = on ? "Processing…" : "Upload & process";
}

$("uploadBtn").addEventListener("click", async () => {
  if (busy) return;                       // guard against double-clicks / double-submit
  const zipFiles = $("zip").files;
  const folderFiles = $("folder").files;
  const chosen = zipFiles.length ? zipFiles : folderFiles;
  if (!chosen.length) { toast("Choose a ZIP file or a folder first.", "error"); return; }
  setBusy(true);
  const fd = new FormData();
  for (const f of chosen) fd.append("files", f, f.name);  // ZIP = 1 file; folder = many
  // provider is server-configured (primary + fallback); no per-batch selection needed

  let r;
  try {
    r = await fetch("/api/batches", { method: "POST", body: fd });
  } catch (e) { toast("upload failed", "error"); setBusy(false); return; }
  if (r.status === 401) { location.href = "/"; return; }
  const data = await r.json();
  if (!r.ok) {
    const m = (data.error && data.error.message) || "upload failed";
    $("validation").innerHTML = '<p class="error">' + esc(m) + "</p>";
    toast(m, "error");
    setBusy(false);
    return;
  }
  batchId = data.batch_id;
  $("zip").value = ""; $("folder").value = "";  // clear selection so a re-click can't re-upload it
  const v = data.validation;
  let msg = "Accepted " + data.total + " file(s).";
  if (v.unmatched_files.length) msg += " Unmatched: " + esc(v.unmatched_files.join(", ")) + ".";
  if (v.missing_audio.length) msg += " Missing audio: " + esc(v.missing_audio.join(", ")) + ".";
  $("validation").innerHTML = '<p class="ok">' + msg + "</p>";
  toast("Accepted " + data.total + " file(s) — processing…", "ok");

  $("progressSec").hidden = false;
  $("resultsSec").hidden = true;          // fresh view for the new batch
  if (poll) clearInterval(poll);          // never leak a previous poll (that would wipe the compare panel)
  poll = setInterval(updateStatus, 1200);
  updateStatus();
});

async function updateStatus() {
  const r = await fetch("/api/batches/" + batchId + "/status");
  if (!r.ok) return;
  const s = await r.json();
  const done = s.done + s.failed;
  $("fill").style.width = (s.total ? (100 * done / s.total) : 0) + "%";
  const served = s.providers || [];
  const fb = (s.provider && served.length && !served.includes(s.provider))
    ? " · ⚠ fallback → " + (s.models || []).join(", ") : "";
  $("progressText").textContent =
    done + " / " + s.total + " processed" + (s.failed ? " (" + s.failed + " failed)" : "") +
    (s.cost_usd ? " · " + fmtCost(s.cost_usd) +
      " (" + Number(s.cost_per_min || 0).toFixed(6) + "/min)" : "") + fb;
  if (s.status === "done") {
    clearInterval(poll); poll = null;   // stop AND null it so nothing stray keeps re-rendering
    setBusy(false);
    const allFail = s.done === 0 && s.failed > 0;
    toast(allFail ? "Batch failed — all " + s.failed + " file(s) errored"
      : "Done — " + s.done + " processed" + (s.failed ? ", " + s.failed + " failed" : ""),
      s.failed ? "error" : "ok");
    showResults();
    loadHistory();
  }
}

async function loadHistory() {
  const r = await fetch("/api/batches");
  if (!r.ok) return;
  const d = await r.json();
  if (!d.batches.length) { $("historyList").innerHTML = '<p class="muted">No batches yet.</p>'; return; }
  let h = "<div class='tablewrap'><table><tr><th>When</th><th>Status</th><th>Files</th><th>Provider</th><th>Cost</th><th></th></tr>";
  for (const b of d.batches) {
    const when = new Date(b.created_at).toLocaleString();
    const st = b.status + (b.failed ? " (" + b.failed + " failed)" : "");
    const models = b.models || [];
    // show "requested → served" when a fallback model (not from the requested family) actually ran
    const prov = (models.length && !models.some((m) => m.startsWith(b.provider)))
      ? esc(b.provider) + " → " + esc(models.join(", ")) : esc(b.provider);
    h += "<tr><td>" + esc(when) + "</td><td>" + esc(st) + "</td><td>" + b.done + "/" + b.total +
      "</td><td>" + prov + "</td><td>" + fmtCost(b.cost_usd) +
      "</td><td><button class='btn-sm' data-bid='" + esc(b.batch_id) +
      "'>view</button></td></tr>";
  }
  $("historyList").innerHTML = h + "</table></div>";
  $("historyList").querySelectorAll("button[data-bid]").forEach((btn) =>
    btn.addEventListener("click", () => viewBatch(btn.getAttribute("data-bid"))));
}

async function viewBatch(bid) {
  batchId = bid;
  const s = await (await fetch("/api/batches/" + bid + "/status")).json();
  if (s.status === "done") {
    $("progressSec").hidden = true;
    showResults();
  } else {
    $("resultsSec").hidden = true;
    $("progressSec").hidden = false;
    clearInterval(poll);
    poll = setInterval(updateStatus, 1200);
    updateStatus();
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

const fmtCost = (c) => (c ? "$" + Number(c).toFixed(6) : "$0");

// Failures get their OWN panel with a human reason — never crammed into the 9 field columns.
function cleanErr(e) {
  let s = String(e || "failed")
    .replace(/\[in#\d+ @ 0x[0-9a-f]+\]/gi, "")        // ffmpeg internal handles
    .replace(/(file )?storage\/\S+/gi, "the file")    // internal storage path
    .replace(/^decode:\s*/i, "").replace(/\s+/g, " ").trim();
  if (/invalid data|protocol not found|decode failed|moov|not.*(audio|supported)/i.test(s))
    return "couldn’t decode — the file may be corrupt, truncated, or not valid audio";
  if (/timeout|timed out/i.test(s)) return "timed out while processing";
  return s.slice(0, 140);
}

function renderFailures(failed) {
  const el = $("failPanel");
  if (!el) return;
  if (!failed.length) { el.innerHTML = ""; el.hidden = true; return; }
  el.hidden = false;
  let h = "<b class='error'>✕ " + failed.length + " file(s) failed to process</b><ul class='faillist'>";
  for (const f of failed) h += "<li><b>" + esc(f.name) + "</b> — " + esc(cleanErr(f.error)) + "</li>";
  el.innerHTML = h + "</ul>";
}

async function showResults() {
  const r = await fetch("/api/batches/" + batchId + "/results");
  const d = await r.json();
  let h = "<tr><th>file</th><th class='src-meta'>model</th><th class='src-meta'>audio&nbsp;tok</th>" +
    "<th class='src-meta'>cost</th>" +
    FIELDS.map((c) => "<th class='src-" + FIELD_SRC[c] + "'>" + c + "</th>").join("") + "</tr>";
  let totCost = 0, totSec = 0, totTok = 0, okRows = 0; const models = new Set(); const failed = [];
  for (const row of d.results) {
    const rj = row.result_json;
    if (!rj) { failed.push(row); continue; }  // failures go to a SEPARATE panel, not the field columns
    okRows += 1;
    const cost = row.cost_usd || 0, tok = row.audio_tokens || 0;
    totCost += cost; totSec += row.audio_seconds || 0; totTok += tok;
    if (row.model) models.add(row.model);
    const cc = "<td class='src-meta'>" + esc(row.model || "—") + "</td><td class='src-meta'>" + tok +
      "</td><td class='src-meta'>" + fmtCost(cost) + "</td>";
    h += "<tr><td>" + esc(row.name) + "</td>" + cc +
      FIELDS.map((c) => "<td class='src-" + FIELD_SRC[c] + "'>" + esc(rj[c]) + "</td>").join("") + "</tr>";
  }
  $("resultsTable").innerHTML = okRows ? h : "";  // no header when there are no successful rows
  renderFailures(failed);
  const st = await (await fetch("/api/batches/" + batchId + "/status")).json();
  renderCost(totCost, totSec, totTok, st);
  const allFailed = (st.done || 0) === 0 && (st.failed || 0) > 0;
  // on a total failure there's nothing meaningful to download or compare
  $("dlCsv").hidden = allFailed;
  $("dlJson").hidden = allFailed;
  $("compareBtn").hidden = allFailed || !st.has_labels;  // labels only, and only if something ran
  $("comparePanel").innerHTML = "";
  $("dlCsv").href = "/api/batches/" + batchId + "/download?fmt=csv";
  $("dlJson").href = "/api/batches/" + batchId + "/download?fmt=json";
  $("resultsSec").hidden = false;
}

// Compare our predictions to labels.csv ground truth: exact match + LLM-semantic (synonym-aware).
$("compareBtn").addEventListener("click", async () => {
  const btn = $("compareBtn"), old = btn.textContent;
  btn.disabled = true; btn.textContent = "Comparing…";
  try {
    const d = await (await fetch("/api/batches/" + batchId + "/compare", { method: "POST" })).json();
    renderCompare(d);
  } catch (e) { toast("compare failed", "error"); }
  btn.disabled = false; btn.textContent = old;
});

function renderCompare(d) {
  const el = $("comparePanel");
  if (!d.n) { el.innerHTML = "<p class='muted'>No labeled files to compare.</p>"; return; }
  const sem = d.semantic;
  let h = "<div class='cost'><b>Predictions vs labels</b> (" + d.n + " labeled file" +
    (d.n > 1 ? "s" : "") + ")";
  if (sem && sem.overall_pct != null) {
    h += " — semantic agreement <b class='" + (sem.overall_pct >= 70 ? "ok" : "error") + "'>" +
      sem.overall_pct + "%</b>";
    if (sem.summary) h += "<div class='basis'>" + esc(sem.summary) + "</div>";
  }
  h += "</div><div class='tablewrap'><table><tr><th>field</th><th>exact match</th>" +
    "<th>semantic</th><th>note</th></tr>";
  for (const f of FIELDS) {
    const ex = (d.exact || {})[f];
    const sf = sem && sem.fields ? sem.fields[f] : null;
    const exTxt = ex && ex.pct != null ? ex.match + "/" + ex.total + " (" + ex.pct + "%)" : "—";
    const semTxt = sf && sf.agree_pct != null ? sf.agree_pct + "%" : "—";
    h += "<tr class='src-" + FIELD_SRC[f] + "'><td>" + f + "</td><td>" + exTxt + "</td><td>" +
      semTxt + "</td><td>" + (sf && sf.note ? esc(sf.note) : "") + "</td></tr>";
  }
  h += "</table></div>";
  h += sem
    ? "<p class='basis'>semantic judged by " + esc(d.semantic_model || "llm") +
      " — synonyms/near-equivalents count as agreement, so it's fairer than exact match on tone &amp; noise_type.</p>"
    : "<p class='basis'>semantic comparison unavailable — showing exact match only.</p>";
  el.innerHTML = h;
}

// real emotion spend: tokens billed × actual per-token pricing, vs the $0.003 ceiling.
// Also flags provider fallback (requested vs served) and cites the rate basis for the cost.
function renderCost(cost, sec, tok, st) {
  const el = $("costSummary");
  if (!el) return;
  const done = st.done || 0, failed = st.failed || 0, total = st.total || 0;
  // Nothing was analyzed → this is a FAILURE, not a cost readout. Never fake a green "under ceiling".
  if (done === 0 && failed > 0) {
    el.className = "cost failbox";
    el.innerHTML = "<b class='error'>✕ Batch failed — all " + failed + " file(s) errored.</b>" +
      "<div class='basis'>Nothing was analyzed, so there is no cost or result. " +
      "See the per-file reason in the red rows below.</div>";
    return;
  }
  el.className = "cost";
  const perMin = sec > 0 ? cost / (sec / 60) : 0;
  const ok = perMin <= 0.003;
  const mins = sec / 60;
  const models = st.models || [], served = st.providers || [], requested = st.provider;
  // headline = the unit the ceiling is measured on: $ per AUDIO-MINUTE (a rate, not a per-call total)
  let html = "<span class='big " + (ok ? "ok" : "error") + "'>$" + perMin.toFixed(6) +
    " / audio-min</span> " + (ok ? "✓ under" : "⚠ over") + " the $0.003 ceiling" +
    "<div class='basis'>= total <b>" + fmtCost(cost) + "</b> ÷ <b>" + mins.toFixed(2) +
    " audio-min</b> (" + tok + " audio tokens). The ceiling is a per-minute RATE — total $ scales with " +
    "call length, but longer calls amortize the fixed prompt (lower $/min); very short clips cost more/min.</div>";
  if (failed > 0) {  // some succeeded, some failed → warn about the skipped files (cost is for the rest)
    html += "<div class='fallback'>⚠ " + failed + " of " + total +
      " file(s) failed and were skipped — cost above is for the " + done + " that succeeded; " +
      "see the red rows below.</div>";
  }
  if (requested && served.length && !served.includes(requested)) {
    html += "<div class='fallback'>⚠ <b>" + esc(requested) + "</b> unavailable — served by <b>" +
      esc(served.join(", ")) + "</b> (" + esc(models.join(", ")) + ") via automatic fallback.</div>";
  }
  const basis = models.map((m) => {
    const r = (st.rates || {})[m] || {};
    return esc(m) + " — audio $" + (r.audio_in || 0) + "/1M · text-in $" + (r.text_in || 0) +
      "/1M · text-out $" + (r.text_out || 0) + "/1M";
  }).join(" · ");
  html += "<div class='basis'>rates (tokens × per-1M): " + (basis || "n/a") +
    " — vendor pricing, Aug 2026.</div>";
  el.innerHTML = html;
}

// load batch history on page open
loadHistory();
