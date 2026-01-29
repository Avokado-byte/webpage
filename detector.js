const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const btnStart = document.getElementById("btnStart");
const btnStop = document.getElementById("btnStop");

const MODEL_URL = "./assets/best.onnx";
const INPUT_SIZE = 512;          // usa el mismo imgsz con que exportaste
const CONF_THRES = 0.25;         // sube a 0.4 si salen falsos positivos
const CLASS_NAME = ["Puente"];   // tu única clase

let session = null;
let stream = null;
let running = false;

// Para mejorar compatibilidad de WASM en CDN
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

btnStart.onclick = async () => {
  try {
    btnStart.disabled = true;
    status("Cargando modelo...");
    if (!session) {
      // WebGL suele ser más rápido; si falla, cae a wasm
      session = await ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ["webgl", "wasm"],
        graphOptimizationLevel: "all",
      });
    }

    status("Pidiendo cámara...");
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false
    });

    video.srcObject = stream;
    await video.play();

    running = true;
    btnStop.disabled = false;
    status("Corriendo detección...");
    requestAnimationFrame(loop);
  } catch (e) {
    console.error(e);
    status("Error: " + (e?.message || e));
    btnStart.disabled = false;
  }
};

btnStop.onclick = () => {
  running = false;
  btnStop.disabled = true;
  btnStart.disabled = false;
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  status("Detenido.");
};

function status(msg) { statusEl.textContent = msg; }

function resizeCanvasToVideo() {
  const vw = video.videoWidth || 0;
  const vh = video.videoHeight || 0;
  if (vw && vh && (canvas.width !== vw || canvas.height !== vh)) {
    canvas.width = vw;
    canvas.height = vh;
  }
}

// Letterbox: mantiene aspecto, rellena con negro hasta cuadrado INPUT_SIZE
function letterboxToSquare(srcW, srcH, dstSize) {
  const r = Math.min(dstSize / srcW, dstSize / srcH);
  const newW = Math.round(srcW * r);
  const newH = Math.round(srcH * r);
  const padX = Math.floor((dstSize - newW) / 2);
  const padY = Math.floor((dstSize - newH) / 2);
  return { r, newW, newH, padX, padY };
}

// Convierte frame actual a tensor [1,3,INPUT,INPUT] float32
function frameToTensor() {
  const vw = video.videoWidth;
  const vh = video.videoHeight;

  const off = document.createElement("canvas");
  off.width = INPUT_SIZE;
  off.height = INPUT_SIZE;
  const octx = off.getContext("2d");

  // fill negro
  octx.fillStyle = "black";
  octx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);

  const lb = letterboxToSquare(vw, vh, INPUT_SIZE);
  octx.drawImage(video, 0, 0, vw, vh, lb.padX, lb.padY, lb.newW, lb.newH);

  const img = octx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;

  // CHW
  const data = new Float32Array(1 * 3 * INPUT_SIZE * INPUT_SIZE);
  let p = 0;
  const area = INPUT_SIZE * INPUT_SIZE;

  for (let y = 0; y < INPUT_SIZE; y++) {
    for (let x = 0; x < INPUT_SIZE; x++) {
      const i = (y * INPUT_SIZE + x) * 4;
      const r = img[i] / 255.0;
      const g = img[i + 1] / 255.0;
      const b = img[i + 2] / 255.0;
      data[p] = r;
      data[p + area] = g;
      data[p + 2 * area] = b;
      p++;
    }
  }

  return { tensor: new ort.Tensor("float32", data, [1, 3, INPUT_SIZE, INPUT_SIZE]), lb };
}

function drawDetections(dets) {
  // dibuja video
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // boxes
  ctx.lineWidth = Math.max(2, Math.round(canvas.width / 300));
  ctx.font = `${Math.max(14, Math.round(canvas.width / 40))}px system-ui`;

  for (const d of dets) {
    const { x1, y1, x2, y2, score, cls } = d;

    ctx.strokeStyle = "#00ff88";
    ctx.fillStyle = "rgba(0,255,136,0.15)";
    ctx.beginPath();
    ctx.rect(x1, y1, x2 - x1, y2 - y1);
    ctx.stroke();
    ctx.fill();

    const label = `${CLASS_NAME[cls] ?? cls} ${(score * 100).toFixed(0)}%`;
    const tw = ctx.measureText(label).width;
    const th = Math.max(18, Math.round(canvas.width / 45));
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillRect(x1, Math.max(0, y1 - th), tw + 10, th);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x1 + 5, Math.max(14, y1 - 6));
  }
}

// Convierte coords del output a coords en el canvas (deshace letterbox)
function unletterboxBox(x1, y1, x2, y2, lb, vw, vh) {
  // output puede venir normalizado (0..1) o en pixels (0..INPUT_SIZE)
  // heurística:
  const normalized = (x2 <= 1.5 && y2 <= 1.5);

  if (normalized) {
    x1 *= INPUT_SIZE; y1 *= INPUT_SIZE; x2 *= INPUT_SIZE; y2 *= INPUT_SIZE;
  }

  // deshace padding
  x1 = (x1 - lb.padX) / lb.r;
  x2 = (x2 - lb.padX) / lb.r;
  y1 = (y1 - lb.padY) / lb.r;
  y2 = (y2 - lb.padY) / lb.r;

  // clamp a imagen original
  x1 = Math.max(0, Math.min(vw, x1));
  x2 = Math.max(0, Math.min(vw, x2));
  y1 = Math.max(0, Math.min(vh, y1));
  y2 = Math.max(0, Math.min(vh, y2));

  // escala a canvas
  const sx = canvas.width / vw;
  const sy = canvas.height / vh;

  return { x1: x1 * sx, y1: y1 * sy, x2: x2 * sx, y2: y2 * sy };
}

async function loop() {
  if (!running) return;

  try {
    resizeCanvasToVideo();
    const vw = video.videoWidth;
    const vh = video.videoHeight;

    if (!vw || !vh) {
      requestAnimationFrame(loop);
      return;
    }

    const { tensor, lb } = frameToTensor();

    // nombre del input: a veces "images", a veces "input", etc.
    // así lo tomamos genérico:
    const feeds = {};
    const inputName = session.inputNames[0];
    feeds[inputName] = tensor;

    const out = await session.run(feeds);
    const outName = session.outputNames[0];
    const raw = out[outName].data; // Float32Array tamaño 1*300*6

    const dets = [];
    for (let i = 0; i < 300; i++) {
      const base = i * 6;
      const x1 = raw[base];
      const y1 = raw[base + 1];
      const x2 = raw[base + 2];
      const y2 = raw[base + 3];
      const score = raw[base + 4];
      const cls = Math.round(raw[base + 5]);

      if (score >
