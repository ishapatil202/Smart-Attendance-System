"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function api(path, opts = {}) {
  const res = await fetch(path, {
    method: opts.method || "GET",
    headers: opts.body ? { "Content-Type": "application/json" } : {},
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  let data = {};
  try { data = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

let toastTimer;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 3500);
}

/* ------------------------------------------------------------ model status */
let modelReady = false;
async function pollHealth() {
  const el = $("#modelStatus");
  try {
    const h = await api("/api/health");
    $("#retention").textContent = h.retention_hours;
    el.className = "status " + h.status;
    if (h.status === "ready") {
      modelReady = true;
      $(".label", el).textContent = `Model ready · ${h.students} enrolled`;
      el.title = `${h.model}, match threshold ${h.threshold}`;
      updateButtons();
      return;
    }
    $(".label", el).textContent = h.status === "error" ? "Model failed to load" : "Loading FaceNet model…";
    if (h.error) { el.title = h.error; return; }
  } catch {
    el.className = "status loading";
    $(".label", el).textContent = "Waking up server…";
  }
  setTimeout(pollHealth, 2500);
}
async function refreshCount() {
  try {
    const h = await api("/api/health");
    if (h.status === "ready") $("#modelStatus .label").textContent = `Model ready · ${h.students} enrolled`;
  } catch {}
}

/* ------------------------------------------------------------------ camera */
const cam = $("#cam");
const video = $("video", cam);
let stream = null;

async function startCamera() {
  if (stream) return true;
  if (!navigator.mediaDevices?.getUserMedia) {
    toast("This browser can't open a camera. Use the photo upload instead.");
    return false;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }, audio: false });
  } catch (e) {
    toast(e.name === "NotAllowedError" ? "Camera permission was blocked. Allow it from the address bar." : "Couldn't open a camera. Use the photo upload instead.");
    return false;
  }
  video.srcObject = stream;
  await video.play().catch(() => {});
  cam.classList.add("live");
  $("#camHint").textContent = "Face the camera in good light.";
  updateButtons();
  return true;
}
$("#camStart").addEventListener("click", startCamera);

function grabFrame() {
  if (!video.videoWidth) return null;
  const scale = Math.min(1, 640 / video.videoWidth);
  const c = document.createElement("canvas");
  c.width = Math.round(video.videoWidth * scale);
  c.height = Math.round(video.videoHeight * scale);
  c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.85);
}
function flash() {
  let f = $(".flash", cam);
  if (!f) { f = document.createElement("div"); f.className = "flash"; cam.appendChild(f); }
  f.style.opacity = ".55";
  setTimeout(() => (f.style.opacity = "0"), 90);
}
function updateButtons() {
  $("#scanBtn").disabled = !(stream && modelReady) || busy;
  $("#enrollBtn").disabled = !(stream && modelReady) || busy;
  $("#fileBtn").classList.toggle("disabled", !modelReady || busy);
  $("#fileInput").disabled = !modelReady || busy;
}
let busy = false;
function setBusy(b) { busy = b; updateButtons(); }

/* -------------------------------------------------------- take attendance */
function drawBoxes(frame, faces) {
  const c = $("canvas.overlay", cam);
  c.width = frame.width;
  c.height = frame.height;
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  const colors = { marked: "#22c55e", already_marked: "#38bdf8", unknown: "#f59e0b" };
  ctx.font = "600 15px 'Instrument Sans', sans-serif";
  for (const f of faces) {
    const [x, y, w, h] = f.box;
    const mx = frame.width - x - w; // the preview is mirrored
    const col = colors[f.status] || "#fff";
    ctx.lineWidth = 3;
    ctx.strokeStyle = col;
    ctx.strokeRect(mx, y, w, h);
    const label = `${f.name || "Unknown"} · ${Math.round(f.confidence * 100)}%`;
    const tw = ctx.measureText(label).width + 12;
    const ty = y >= 24 ? y - 24 : y + h;
    ctx.fillStyle = col;
    ctx.fillRect(mx - 1.5, ty, tw, 24);
    ctx.fillStyle = "#06201d";
    ctx.fillText(label, mx + 4.5, ty + 17);
  }
  clearTimeout(cam._clear);
  cam._clear = setTimeout(() => ctx.clearRect(0, 0, c.width, c.height), 4000);
}

$("#scanBtn").addEventListener("click", async () => {
  const frame = grabFrame();
  if (!frame) return;
  setBusy(true);
  flash();
  try {
    const r = await api("/api/recognize", { method: "POST", body: { frame } });
    drawBoxes(r.frame, r.faces);
    const list = $("#results");
    if (!r.faces.length) {
      list.innerHTML = `<li><span class="muted">No face found. Move closer and face the light.</span><span class="sub">${r.latency_ms} ms</span></li>`;
    } else {
      list.innerHTML = r.faces.map((f) => {
        const chip = f.status === "marked" ? `<span class="chip ok">Marked present</span>`
          : f.status === "already_marked" ? `<span class="chip neutral">Already marked today</span>`
          : `<span class="chip warn">Not recognised</span>`;
        const who = f.name ? `<strong>${esc(f.name)}</strong>` : `<span class="muted">Unknown face. Enroll first?</span>`;
        return `<li><span>${who}<br><span class="sub">${Math.round(f.confidence * 100)}% similarity · ${r.latency_ms} ms</span></span>${chip}</li>`;
      }).join("");
      const marked = r.faces.filter((f) => f.status === "marked").map((f) => f.name);
      if (marked.length) {
        toast(`Welcome, ${marked.join(", ")}! Attendance recorded.`);
        loadAttendance(marked);
      }
    }
    loadPerf();
  } catch (e) {
    toast(e.message);
  } finally {
    setBusy(false);
  }
});

/* ------------------------------------------------------------------ enroll */
const msg = $("#enrollMsg");
function showMsg(text, kind) { msg.className = "form-msg " + (kind || ""); msg.textContent = text; }

async function submitEnroll(photos, mirrored) {
  const form = $("#enrollForm");
  if (!form.reportValidity()) return;
  const name = $("#nameInput").value;
  $("#thumbs").innerHTML = photos.map((p) => `<img src="${p}" alt="" class="${mirrored ? "mirror" : ""}">`).join("");
  showMsg("Creating FaceNet embeddings…");
  setBusy(true);
  try {
    const r = await api("/api/enroll", { method: "POST", body: { name, photos } });
    showMsg(`Enrolled ${r.name} (${r.embeddings_stored} embeddings stored). Now press “Take attendance”.`, "ok");
    form.reset();
    refreshCount();
  } catch (err) {
    showMsg(err.message, "err");
  } finally {
    setBusy(false);
  }
}

$("#enrollForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!$("#enrollForm").reportValidity()) return;
  if (!stream && !(await startCamera())) return;
  setBusy(true);
  const photos = [];
  const btn = $("#enrollBtn");
  for (let i = 0; i < 3; i++) {
    btn.textContent = `Capturing ${i + 1}/3…`;
    await sleep(i ? 450 : 200);
    const f = grabFrame();
    if (f) { photos.push(f); flash(); }
  }
  btn.textContent = "Capture from webcam & enroll";
  setBusy(false);
  if (photos.length) await submitEnroll(photos, true);
});

$("#fileInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;
  if (!$("#enrollForm").reportValidity()) return;
  const dataUrl = await new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, 800 / Math.max(img.width, img.height));
      const c = document.createElement("canvas");
      c.width = Math.round(img.width * scale);
      c.height = Math.round(img.height * scale);
      c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
      URL.revokeObjectURL(img.src);
      resolve(c.toDataURL("image/jpeg", 0.88));
    };
    img.onerror = () => reject(new Error("That file isn't an image the browser can read."));
    img.src = URL.createObjectURL(file);
  }).catch((err) => { showMsg(err.message, "err"); return null; });
  if (dataUrl) await submitEnroll([dataUrl], false);
});

/* -------------------------------------------------------------- attendance */
let filter = "today";
async function loadAttendance(highlight = []) {
  const q = filter === "today" ? "?date=today" : "";
  $("#exportBtn").href = "/api/attendance/export" + q;
  try {
    const { records } = await api("/api/attendance" + q);
    $("#rows").innerHTML = records.length
      ? records.map((r) => `
        <tr class="${highlight.includes(r.name) && !r.is_sample ? "fresh" : ""}">
          <td>${esc(r.name)}</td><td>${esc(r.date)}</td><td>${esc(r.time)}</td>
          <td>${Math.round(r.confidence * 100)}%</td>
          <td>${r.is_sample ? '<span class="tag">sample</span>' : '<span class="chip ok">live</span>'}</td>
        </tr>`).join("")
      : `<tr><td colspan="5" class="muted">No records ${filter === "today" ? "today" : "yet"}.</td></tr>`;
  } catch (e) { toast(e.message); }
}
$$(".seg button").forEach((b) => b.addEventListener("click", () => {
  filter = b.dataset.filter;
  $$(".seg button").forEach((x) => x.classList.toggle("on", x === b));
  loadAttendance();
}));

async function loadPerf() {
  try {
    const m = await api("/api/metrics");
    const s = m.requests_with_face;
    $("#perf").textContent = s ? `Median recognition: ${s.p50_ms} ms on CPU (${s.count} requests)` : "";
  } catch {}
}

/* -------------------------------------------------------------------- boot */
pollHealth();
updateButtons();
loadAttendance();
loadPerf();
