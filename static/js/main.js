// ─── Config ─────────────────────────────────────────────────────────────────
const API_BASE = "http://localhost:5000"; // Đổi URL nếu deploy server khác

const CLASS_COLORS = {
  glioma: "#ff6b6b",
  meningioma: "#ffa94d",
  pituitary: "#cc99ff",
  notumor: "#69db7c",
};
const CLASS_VI = {
  glioma: "U thần kinh đệm",
  meningioma: "U màng não",
  pituitary: "U tuyến yên",
  notumor: "Không có u",
};

let currentFile = null;
let currentGradcamData = {};

// ─── File Handling ───────────────────────────────────────────────────────────
const dropZone = document.getElementById("dropZone");
const imgWrapper = document.getElementById("imgWrapper");
const previewImg = document.getElementById("previewImg");
const btnRow = document.getElementById("btnRow");
const analyzeBtn = document.getElementById("analyzeBtn");
const scanOverlay = document.getElementById("scanOverlay");
const errorBox = document.getElementById("errorBox");
const imgBadge = document.getElementById("imgBadge");

if (dropZone) {
  ["dragover", "dragenter"].forEach((e) =>
    dropZone.addEventListener(e, (ev) => {
      ev.preventDefault();
      dropZone.classList.add("drag-over");
    }),
  );
  ["dragleave", "drop"].forEach((e) =>
    dropZone.addEventListener(e, () => dropZone.classList.remove("drag-over")),
  );
  dropZone.addEventListener("drop", (ev) => {
    ev.preventDefault();
    const f = ev.dataTransfer.files[0];
    if (f) handleFile(f);
  });
}

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showError("Vui lòng chọn file ảnh hợp lệ (JPG, PNG...)");
    return;
  }
  clearError();
  resetResults();
  currentFile = file;

  const url = URL.createObjectURL(file);
  previewImg.src = url;
  dropZone.style.display = "none";
  imgWrapper.classList.add("visible");
  imgBadge.style.display = "none";
  btnRow.style.display = "flex";
}

function resetAll() {
  currentFile = null;
  previewImg.src = "";
  dropZone.style.display = "";
  imgWrapper.classList.remove("visible");
  btnRow.style.display = "none";
  document.getElementById("fileInput").value = "";
  resetResults();
  clearError();
}

function resetResults() {
  document.getElementById("resultBody").innerHTML = `
  <div class="result-empty">
    <span><i class="fa-solid fa-chart-column"></i></span>
    <p>Tải ảnh MRI và nhấn<br/><strong style="color:var(--accent)">Phân tích MRI</strong><br/>để xem kết quả</p>
  </div>`;
  const gradcamCard = document.getElementById("gradcamCard");
  if (gradcamCard) gradcamCard.style.display = "none";
  const modelCompareCard = document.getElementById("modelCompareCard");
  if (modelCompareCard) modelCompareCard.style.display = "none";
  imgBadge.style.display = "none";
  scanOverlay.classList.remove("active");
}

// ─── Analyze ─────────────────────────────────────────────────────────────────
async function analyze() {
  if (!currentFile) return;
  setLoading(true);

  const formData = new FormData();
  formData.append("image", currentFile);

  try {
    const resp = await fetch(`${API_BASE}/predict?gradcam_model=both`, {
      method: "POST",
      body: formData,
    });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.error || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    renderResults(data);
  } catch (e) {
    showError(
      `Lỗi kết nối API: ${e.message}. Đảm bảo server Flask đang chạy tại ${API_BASE}`,
    );
    setLoading(false);
  }
}

function setLoading(on) {
  analyzeBtn.disabled = on;
  if (on) {
    scanOverlay.classList.add("active");
    analyzeBtn.innerHTML = `<span style="animation:spin 1s linear infinite;display:inline-block">⟳</span> Đang phân tích...`;
    document.getElementById("resultBody").innerHTML = `
    <div class="loading-state">
      <div class="spinner"><i class="fa-solid fa-brain"></i></div>
      <p>Đang phân tích MRI...</p>
      <small>EfficientNet-V2-S</small>
      <div class="dots"><span>●</span><span>●</span><span>●</span></div>
    </div>`;
  } else {
    scanOverlay.classList.remove("active");
    analyzeBtn.innerHTML = `<i class="fa-solid fa-microscope"></i> Phân tích MRI`;
  }
}

// ─── Render Results ───────────────────────────────────────────────────────────
function renderResults(data) {
  setLoading(false);
  const p = data.prediction;
  const cls = p.class;
  const color = CLASS_COLORS[cls] || "#cdd6f4";
  const sev = data.severity;

  // Badge trên ảnh
  imgBadge.style.display = "block";
  imgBadge.className = `img-badge ${p.has_tumor ? "positive" : "negative"}`;
  imgBadge.textContent = p.has_tumor
    ? "⚠ PHÁT HIỆN BẤT THƯỜNG"
    : "✓ BÌNH THƯỜNG";

  // ── Diagnosis block
  let html = `
  <div class="diagnosis bg-${cls}" style="border:1px solid ${color}25">
    <div class="diagnosis-header">
      <div>
        <div class="diagnosis-label">KẾT QUẢ CHẨN ĐOÁN</div>
        <div class="diagnosis-name c-${cls}">${p.class_vi}</div>
        <div class="diagnosis-sub">${cls.toUpperCase()} · ${p.confidence}% tin cậy</div>
      </div>
      <div class="severity-badge sev-${sev.level}">
        <div class="s-label">MỨC ĐỘ</div>
        <div class="s-value">${sev.label}</div>
      </div>
    </div>
  </div>`;

  // ── Confidence bars
  html += `
  <div class="conf-section">
    <div class="conf-title">ĐỘ TIN CẬY PHÂN LOẠI · ENSEMBLE</div>`;
  const sorted = Object.entries(data.probabilities).sort(
    (a, b) => b[1].score_pct - a[1].score_pct,
  );
  for (const [cn, info] of sorted) {
    const c = CLASS_COLORS[cn] || "#cdd6f4";
    html += `
    <div class="conf-row">
      <div class="conf-meta">
        <span>${info.label_vi}</span>
        <strong style="color:${c}">${info.score_pct}%</strong>
      </div>
      <div class="conf-track">
        <div class="conf-fill" id="fill-${cn}" style="background:${c}; box-shadow:0 0 6px ${c}60"></div>
      </div>
    </div>`;
  }
  html += `</div>`;

  // ── Clinical notes
  html += `
  <div class="note-box">
    <div class="note-title">KHUYẾN NGHỊ</div>
    <p>💡 ${data.recommendation}</p>
  </div>`;

  document.getElementById("resultBody").innerHTML = html;

  // Animate bars
  requestAnimationFrame(() => {
    for (const [cn, info] of sorted) {
      const el = document.getElementById(`fill-${cn}`);
      if (el) el.style.width = `${info.score_pct}%`;
    }
  });

  // ── Grad-CAM
  currentGradcamData = data.gradcam || {};
  renderGradcam();

  // ── Per-model comparison
  if (data.per_model && Object.keys(data.per_model).length > 1) {
    renderModelCompare(data.per_model);
  }
}

function renderGradcam() {
  const keys = Object.keys(currentGradcamData).filter(
    (k) => currentGradcamData[k],
  );
  if (!keys.length) return;

  const card = document.getElementById("gradcamCard");
  const tabs = document.getElementById("gradcamTabs");
  const img = document.getElementById("gradcamImg");

  card.style.display = "block";
  tabs.innerHTML = "";
  keys.forEach((k, i) => {
    const btn = document.createElement("button");
    btn.className = `gtab ${i === 0 ? "active" : ""}`;
    const modelLabels = {
      resnet50: "ResNet50",
      efficientnet: "EfficientNet-B0",
      convnext_small: "ConvNeXt-Small",
      efficientnet_v2_s: "EfficientNet-V2-S",
      swin_t: "Swin-Transformer-T",
      swin_b: "Swin-Transformer-Base",
    };
    btn.textContent = modelLabels[k] || k;
    btn.onclick = () => {
      document
        .querySelectorAll(".gtab")
        .forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      img.src = currentGradcamData[k];
    };
    tabs.appendChild(btn);
  });
  img.src = currentGradcamData[keys[0]];
}

function renderModelCompare(perModel) {
  const card = document.getElementById("modelCompareCard");
  const container = document.getElementById("modelCompare");
  card.style.display = "block";
  container.innerHTML = Object.entries(perModel)
    .map(([name, scores]) => {
      const modelLabels = {
        resnet50: "ResNet50",
        efficientnet: "EfficientNet-B0",
        convnext_small: "ConvNeXt-Small",
        efficientnet_v2_s: "EfficientNet-V2-S",
        swin_t: "Swin-Transformer-T",
        swin_b: "Swin-Transformer-Base",
      };
      const displayName = modelLabels[name] || name;
      const rows = Object.entries(scores)
        .sort((a, b) => b[1] - a[1])
        .map(
          ([cn, pct]) => `
      <div class="model-row">
        <span>${CLASS_VI[cn]}</span>
        <strong style="color:${CLASS_COLORS[cn]}">${pct}%</strong>
      </div>`,
        )
        .join("");
      return `
    <div class="model-card">
      <div class="model-name">${displayName.toUpperCase()}</div>
      ${rows}
    </div>`;
    })
    .join("");
}

// ─── Error handling ──────────────────────────────────────────────────────────
function showError(msg) {
  errorBox.textContent = "⚠ " + msg;
  errorBox.classList.add("visible");
  setLoading(false);
}
function clearError() {
  errorBox.classList.remove("visible");
}
