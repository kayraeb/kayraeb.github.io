// ---------------------------------------------------------
// 1. CONFIGURATION
// ---------------------------------------------------------
// Note: VISUAL_SIZE and HIDDEN_SIZE are now dynamic based on window size
// but we keep a reference to the hidden buffer for the AI.

// --- VISUAL AESTHETICS (MS Paint Style) ---
const VISUAL_COLOR = '#000000'; // Black Ink
const VISUAL_STROKE_WIDTH = 25; // Pencil thickness (Scales with canvas?) No, kept constant like Paint.

// --- DATA PHYSICS (The Brain) ---
const HIDDEN_STROKE_WIDTH = 45;
const BG_COLOR = '#ffffff';     // Visual canvas background (White paper)
const DATA_BG_COLOR = 'black';  // Hidden tensor background

// ---------------------------------------------------------
// 2. DOM ELEMENTS
// ---------------------------------------------------------
const canvasContainer = document.getElementById('canvasContainer');
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });

// The hidden canvas mimics the visual canvas size to ensure coordinate mapping is 1:1
const hiddenCanvas = document.getElementById('hiddenCanvas');
const hiddenCtx = hiddenCanvas.getContext('2d', { willReadFrequently: true });

const sensorCanvas = document.getElementById('sensorCanvas');
const sensorCtx = sensorCanvas.getContext('2d');

const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const predictionResult = document.getElementById('predictionResult');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceValue = document.getElementById('confidenceValue');
const statusText = document.getElementById('statusText');
const statusIndicator = document.getElementById('statusIndicator');
const contendersList = document.getElementById('contendersList');
const latencyMetric = document.getElementById('latencyMetric');
const resolutionDisplay = document.getElementById('resolutionDisplay');

let isDrawing = false;
let lastX = 0;
let lastY = 0;
let model = null;

// ---------------------------------------------------------
// 3. STARTUP
// ---------------------------------------------------------
async function initApp() {
    if (typeof tf === 'undefined') {
        updateStatus("TF_LIB_ERROR", "error");
        return;
    }

    // Initial sizing
    resizeCanvas();
    
    // Listen for resize
    window.addEventListener('resize', debounce(resizeCanvas, 100));

    try {
        await tf.setBackend('cpu');
        model = await tf.loadLayersModel('./web_model/model.json');

        // Warmup
        const dummy = tf.zeros([1, 784]);
        model.predict(dummy).dispose();
        dummy.dispose();

        updateStatus("System Ready", "active");
        predictBtn.disabled = false;

    } catch (e) {
        console.error(e);
        updateStatus("Model Error", "error");
        alert("Error loading model. Ensure files are served via HTTP.");
    }
}

function updateStatus(msg, type) {
    if(statusText) statusText.innerText = msg;
    if(statusIndicator) statusIndicator.className = `indicator ${type}`;
}

// ---------------------------------------------------------
// 4. DRAWING ENGINE
// ---------------------------------------------------------
function resizeCanvas() {
    // Get the container dimensions
    const w = canvasContainer.clientWidth;
    const h = canvasContainer.clientHeight;

    // Only resize if dimensions changed to avoid clearing content unnecessarily
    // (Though in MS Paint resizing usually clears or crops, we will simply clear for simplicity here)
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
        
        hiddenCanvas.width = w;
        hiddenCanvas.height = h;

        if(resolutionDisplay) resolutionDisplay.innerText = `${w} x ${h}px`;
        
        // Re-apply context styles after resize resets them
        setupContext();
        clearBoard();
    }
}

function setupContext() {
    // 1. Visual Context Style
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = VISUAL_COLOR;
    ctx.lineWidth = VISUAL_STROKE_WIDTH;

    // 2. Hidden Context Style
    hiddenCtx.lineCap = 'round';
    hiddenCtx.lineJoin = 'round';
    hiddenCtx.strokeStyle = '#ffffff';
    hiddenCtx.lineWidth = HIDDEN_STROKE_WIDTH;
}

function clearBoard() {
    ctx.fillStyle = BG_COLOR;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    hiddenCtx.fillStyle = DATA_BG_COLOR;
    hiddenCtx.fillRect(0, 0, hiddenCanvas.width, hiddenCanvas.height);

    sensorCtx.fillStyle = 'black';
    sensorCtx.fillRect(0, 0, 28, 28);

    if(predictionResult) predictionResult.innerText = "?";
    if(confidenceValue) confidenceValue.innerText = "0.0%";
    if(confidenceBar) confidenceBar.style.width = "0%";
    if(contendersList) contendersList.innerHTML = "";
}

function getCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    const cy = e.touches ? e.touches[0].clientY : e.clientY;

    // Since canvas fills the container 1:1, no scaling calc is needed usually,
    // but we keep it for robustness.
    return {
        x: cx - rect.left,
        y: cy - rect.top
    };
}

function start(e) {
    isDrawing = true;
    const {x, y} = getCoords(e);
    lastX = x;
    lastY = y;

    // Initial Dot
    ctx.beginPath();
    ctx.arc(x, y, VISUAL_STROKE_WIDTH/2, 0, Math.PI*2);
    ctx.fillStyle = VISUAL_COLOR;
    ctx.fill();

    hiddenCtx.beginPath();
    hiddenCtx.arc(x, y, HIDDEN_STROKE_WIDTH/2, 0, Math.PI*2);
    hiddenCtx.fillStyle = '#ffffff';
    hiddenCtx.fill();
}

function move(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const {x, y} = getCoords(e);

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();

    hiddenCtx.beginPath();
    hiddenCtx.moveTo(lastX, lastY);
    hiddenCtx.lineTo(x, y);
    hiddenCtx.stroke();

    lastX = x;
    lastY = y;
}

function end() {
    isDrawing = false;
    ctx.beginPath();
    hiddenCtx.beginPath();
}

function debounce(func, wait) {
    let timeout;
    return function() {
        const context = this, args = arguments;
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(context, args), wait);
    };
}

canvas.addEventListener('mousedown', start);
canvas.addEventListener('touchstart', start, {passive: false});
canvas.addEventListener('mousemove', move);
canvas.addEventListener('touchmove', move, {passive: false});
window.addEventListener('mouseup', end);
window.addEventListener('touchend', end);
clearBtn.addEventListener('click', clearBoard);

// ---------------------------------------------------------
// 5. PREPROCESSING (Center of Mass)
// ---------------------------------------------------------
function getBoundingBox(data, w, h) {
    let minX=w, minY=h, maxX=0, maxY=0, found=false;
    for(let y=0; y<h; y++) {
        for(let x=0; x<w; x++) {
            if(data[(y*w + x)*4] > 50) {
                if(x < minX) minX=x;
                if(x > maxX) maxX=x;
                if(y < minY) minY=y;
                if(y > maxY) maxY=y;
                found=true;
            }
        }
    }
    return found ? {minX, minY, w: maxX-minX+1, h: maxY-minY+1} : null;
}

function getCenterOfMass(data, w, h) {
    let sumX=0, sumY=0, mass=0;
    for(let y=0; y<h; y++) {
        for(let x=0; x<w; x++) {
            const val = data[(y*w + x)*4];
            if(val > 50) {
                sumX += x*val;
                sumY += y*val;
                mass += val;
            }
        }
    }
    if(mass === 0) return null;
    return { x: sumX/mass, y: sumY/mass };
}

function extractInputTensor(shiftX = 0, shiftY = 0) {
    const w = hiddenCanvas.width;
    const h = hiddenCanvas.height;
    const rawData = hiddenCtx.getImageData(0, 0, w, h);
    const bbox = getBoundingBox(rawData.data, w, h);
    if(!bbox) return null;

    // 28x28 is the MNIST standard
    const temp = document.createElement('canvas');
    temp.width = 28; temp.height = 28;
    const tCtx = temp.getContext('2d');
    tCtx.fillStyle = 'black';
    tCtx.fillRect(0, 0, 28, 28);

    // Scale to 20x20 to fit in 28x28 box
    const scale = 20 / Math.max(bbox.w, bbox.h);
    const sw = bbox.w * scale;
    const sh = bbox.h * scale;

    const scaler = document.createElement('canvas');
    scaler.width = 28; scaler.height = 28;
    const sCtx = scaler.getContext('2d');
    sCtx.imageSmoothingEnabled = true;
    sCtx.imageSmoothingQuality = 'high';
    sCtx.drawImage(hiddenCanvas, bbox.minX, bbox.minY, bbox.w, bbox.h, 0, 0, sw, sh);

    const helper = document.createElement('canvas');
    helper.width = 28; helper.height = 28;
    helper.getContext('2d').drawImage(scaler, 0, 0, sw, sh, (28-sw)/2, (28-sh)/2, sw, sh);
    const com = getCenterOfMass(helper.getContext('2d').getImageData(0,0,28,28).data, 28, 28);

    let tx = (28-sw)/2;
    let ty = (28-sh)/2;

    if(com) {
        tx += (14 - com.x);
        ty += (14 - com.y);
    }

    tx += shiftX;
    ty += shiftY;

    tCtx.drawImage(scaler, 0, 0, sw, sh, tx, ty, sw, sh);

    if(shiftX === 0 && shiftY === 0) {
        sensorCtx.drawImage(temp, 0, 0);
    }

    const finalData = tCtx.getImageData(0, 0, 28, 28).data;
    const input = [];
    for(let i=0; i<finalData.length; i+=4) {
        let val = finalData[i] / 255.0;
        val = Math.min(1.0, val * 1.3);
        input.push(val);
    }

    return input;
}

// ---------------------------------------------------------
// 6. INFERENCE (TTA)
// ---------------------------------------------------------
predictBtn.addEventListener('click', async () => {
    if(!model) return;

    const t0 = performance.now();
    const shifts = [[0,0], [0,1], [0,-1], [1,0], [-1,0]];
    const batchInputs = [];

    for(let s of shifts) {
        const inp = extractInputTensor(s[0], s[1]);
        if(inp) batchInputs.push(inp);
    }

    if(batchInputs.length === 0) {
        predictionResult.innerText = "?";
        return;
    }

    const batch = tf.tensor2d(batchInputs);
    const preds = model.predict(batch);
    const probs = await preds.data();

    const avg = new Array(10).fill(0);
    for(let i=0; i<batchInputs.length; i++) {
        for(let j=0; j<10; j++) avg[j] += probs[i*10 + j];
    }
    for(let j=0; j<10; j++) avg[j] /= batchInputs.length;

    const max = Math.max(...avg);
    const idx = avg.indexOf(max);
    const t1 = performance.now();

    predictionResult.innerText = idx;
    confidenceValue.innerText = (max * 100).toFixed(1) + "%";
    confidenceBar.style.width = (max * 100) + "%";
    
    if(latencyMetric) latencyMetric.innerText = `Latency: ${Math.round(t1-t0)} ms`;

    // XP colors for progress
    if(max > 0.9) confidenceBar.style.backgroundColor = '#00FF00';
    else if(max > 0.6) confidenceBar.style.backgroundColor = '#FFFF00';
    else confidenceBar.style.backgroundColor = '#FF0000';

    if(contendersList) {
        const sorted = avg.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p).slice(0, 3);
        contendersList.innerHTML = "";
        sorted.forEach(item => {
            const pct = (item.p * 100).toFixed(1);
            const row = document.createElement('div');
            row.className = 'contender-row';
            row.innerHTML = `
                <span style="width:15px;">${item.i}</span>
                <div class="c-bar-bg">
                    <div class="c-bar-fill" style="width:${pct}%; background-color:#000080;"></div>
                </div>
                <span style="width:35px; text-align:right;">${pct}%</span>
            `;
            contendersList.appendChild(row);
        });
    }

    batch.dispose();
    preds.dispose();
});

window.onload = initApp;
