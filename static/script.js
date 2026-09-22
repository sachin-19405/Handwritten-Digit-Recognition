const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");
const resultBox = document.getElementById("resultBox");
const topPreds = document.getElementById("topPreds");

const tabDraw = document.getElementById("tabDraw");
const tabUpload = document.getElementById("tabUpload");
const drawPane = document.getElementById("drawPane");
const uploadPane = document.getElementById("uploadPane");

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const uploadPreviewWrap = document.getElementById("uploadPreviewWrap");
const uploadPreview = document.getElementById("uploadPreview");
const dropZoneText = document.getElementById("dropZoneText");
const uploadClearBtn = document.getElementById("uploadClearBtn");
const uploadPredictBtn = document.getElementById("uploadPredictBtn");

let selectedFile = null;

function resetResult(message) {
  resultBox.className = "result-box empty";
  resultBox.innerHTML = `<span class="placeholder">${message}</span>`;
  topPreds.innerHTML = "";
}

function showPrediction(data) {
  resultBox.className = "result-box filled";
  resultBox.innerHTML = `
    <div class="result-digit">${data.prediction}</div>
    <div class="result-confidence">${(data.confidence * 100).toFixed(1)}% confidence</div>
  `;
  topPreds.innerHTML = data.top3
    .map(
      (p, i) => `
      <div class="pred-row ${i === 0 ? "top" : ""}">
        <span class="digit-label">${p.digit}</span>
        <div class="pred-bar-track"><div class="pred-bar-fill" style="width:${(p.prob * 100).toFixed(1)}%"></div></div>
        <span>${(p.prob * 100).toFixed(1)}%</span>
      </div>`
    )
    .join("");
}

// ---- Tab switching ----
function activateTab(which) {
  const isDraw = which === "draw";
  tabDraw.classList.toggle("active", isDraw);
  tabUpload.classList.toggle("active", !isDraw);
  drawPane.classList.toggle("hidden", !isDraw);
  uploadPane.classList.toggle("hidden", isDraw);
  resetResult(isDraw ? "Draw a digit and click Predict" : "Upload an image and click Predict");
}
tabDraw.addEventListener("click", () => activateTab("draw"));
tabUpload.addEventListener("click", () => activateTab("upload"));

// ---- Upload handling ----
function setSelectedFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    resetResult("Please choose an image file.");
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    uploadPreview.src = e.target.result;
    uploadPreviewWrap.classList.remove("hidden");
    dropZoneText.style.display = "none";
  };
  reader.readAsDataURL(file);
  uploadPredictBtn.disabled = false;
}

dropZone.addEventListener("click", (e) => {
  // label already triggers the input, avoid double-firing on the input itself
});
fileInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files[0]) setSelectedFile(e.target.files[0]);
});

["dragenter", "dragover"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
  })
);
dropZone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) setSelectedFile(file);
});

uploadClearBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  uploadPreviewWrap.classList.add("hidden");
  dropZoneText.style.display = "flex";
  uploadPredictBtn.disabled = true;
  resetResult("Upload an image and click Predict");
});

uploadPredictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  uploadPredictBtn.disabled = true;
  uploadPredictBtn.textContent = "Thinking…";

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);
    const res = await fetch("/predict_upload", { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) {
      resetResult(`Error: ${data.error}`);
    } else {
      showPrediction(data);
    }
  } catch (err) {
    resetResult("Something went wrong. Try again.");
  } finally {
    uploadPredictBtn.disabled = false;
    uploadPredictBtn.textContent = "Predict";
  }
});

let drawing = false;
let hasDrawn = false;

function initCanvas() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 18;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}
initCanvas();

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  if (e.touches && e.touches.length > 0) {
    return {
      x: (e.touches[0].clientX - rect.left) * scaleX,
      y: (e.touches[0].clientY - rect.top) * scaleY,
    };
  }
  return { x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY };
}

function startDraw(e) {
  drawing = true;
  hasDrawn = true;
  const pos = getPos(e);
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y);
  e.preventDefault();
}

function draw(e) {
  if (!drawing) return;
  const pos = getPos(e);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
  e.preventDefault();
}

function endDraw(e) {
  drawing = false;
  if (e) e.preventDefault();
}

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", draw);
window.addEventListener("mouseup", endDraw);

canvas.addEventListener("touchstart", startDraw, { passive: false });
canvas.addEventListener("touchmove", draw, { passive: false });
canvas.addEventListener("touchend", endDraw, { passive: false });

clearBtn.addEventListener("click", () => {
  initCanvas();
  hasDrawn = false;
  resetResult("Draw a digit and click Predict");
});

predictBtn.addEventListener("click", async () => {
  if (!hasDrawn) {
    resetResult("Draw something first!");
    return;
  }

  predictBtn.disabled = true;
  predictBtn.textContent = "Thinking…";

  try {
    const imageData = canvas.toDataURL("image/png");
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageData }),
    });
    const data = await res.json();

    if (data.error) {
      resetResult(`Error: ${data.error}`);
    } else {
      showPrediction(data);
    }
  } catch (err) {
    resetResult("Something went wrong. Try again.");
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Predict";
  }
});
