const TAU = Math.PI * 2;

const PHASES = [
  {
    angle: 0,
    name: 'New Moon',
    short: 'Almost all dark',
    story: 'The Moon is between Earth and the Sun, so the bright half is facing away from us.',
  },
  {
    angle: 45,
    name: 'Waxing Crescent',
    short: 'A thin bright smile is growing',
    story: 'We can see a small bright crescent. Waxing means the bright part is growing.',
  },
  {
    angle: 90,
    name: 'First Quarter',
    short: 'Half looks bright',
    story: 'Half the Moon looks bright. Quarter means the Moon is one quarter of the way around Earth.',
  },
  {
    angle: 135,
    name: 'Waxing Gibbous',
    short: 'Almost full and still growing',
    story: 'More than half is bright, and each night the bright part is still getting bigger.',
  },
  {
    angle: 180,
    name: 'Full Moon',
    short: 'The whole face looks bright',
    story: 'The lit half is facing us, so the Moon looks almost fully bright.',
  },
  {
    angle: 225,
    name: 'Waning Gibbous',
    short: 'Still big, but shrinking',
    story: 'More than half is bright, but now the bright part is getting smaller.',
  },
  {
    angle: 270,
    name: 'Last Quarter',
    short: 'Half looks bright again',
    story: 'Half the Moon looks bright again, but this time the other side is lit.',
  },
  {
    angle: 315,
    name: 'Waning Crescent',
    short: 'A tiny bright piece is left',
    story: 'Only a thin crescent is bright before the Moon returns to new Moon.',
  },
];

const state = {
  angleDeg: 180,
  playing: false,
  shadowMode: 'phase',
  orbitDragging: false,
  lastFrame: 0,
  quizScore: 0,
  quizLocked: false,
  quizQuestion: null,
};

const orbitInteraction = {
  centerX: 0,
  centerY: 0,
  radius: 0,
};

const elements = {
  heroMoonCanvas: document.getElementById('heroMoonCanvas'),
  orbitCanvas: document.getElementById('orbitCanvas'),
  moonCanvas: document.getElementById('moonCanvas'),
  shadowCanvas: document.getElementById('shadowCanvas'),
  quizCanvas: document.getElementById('quizCanvas'),
  phaseSlider: document.getElementById('phaseSlider'),
  playButton: document.getElementById('playButton'),
  dayReadout: document.getElementById('dayReadout'),
  phaseName: document.getElementById('phaseName'),
  phaseTrend: document.getElementById('phaseTrend'),
  phaseStory: document.getElementById('phaseStory'),
  phaseDetail: document.getElementById('phaseDetail'),
  phaseCards: Array.from(document.querySelectorAll('.phase-card')),
  phaseCanvases: Array.from(document.querySelectorAll('[data-phase-canvas]')),
  shadowButtons: Array.from(document.querySelectorAll('[data-shadow-mode]')),
  shadowCopy: document.getElementById('shadowCopy'),
  quizOptions: document.getElementById('quizOptions'),
  quizFeedback: document.getElementById('quizFeedback'),
  quizScore: document.getElementById('quizScore'),
  nextQuestion: document.getElementById('nextQuestion'),
};

function resizeCanvas(canvas) {
  const ratio = Math.max(window.devicePixelRatio || 1, 1);
  const cssWidth = canvas.clientWidth || canvas.width;
  const cssHeight = canvas.clientHeight || canvas.height;
  canvas.width = Math.round(cssWidth * ratio);
  canvas.height = Math.round(cssHeight * ratio);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: cssWidth, height: cssHeight };
}

function normalizeAngle(angleDeg) {
  const normalized = angleDeg % 360;
  return normalized < 0 ? normalized + 360 : normalized;
}

function angleToRadians(angleDeg) {
  return normalizeAngle(angleDeg) * Math.PI / 180;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function getPhaseIndex(angleDeg) {
  return Math.floor((normalizeAngle(angleDeg) + 22.5) / 45) % 8;
}

function getPhase(angleDeg) {
  return PHASES[getPhaseIndex(angleDeg)];
}

function getIllumination(angleDeg) {
  return (1 - Math.cos(angleToRadians(angleDeg))) / 2;
}

function getTrend(angleDeg) {
  const angle = normalizeAngle(angleDeg);
  if (angle === 0) return 'New Moon: waxing begins next';
  if (angle === 180) return 'Full Moon: waning begins next';
  return angle < 180 ? 'Waxing: the bright part grows' : 'Waning: the bright part shrinks';
}

function getDayNumber(angleDeg) {
  return 1 + (29.5 * normalizeAngle(angleDeg)) / 360;
}

function lerpColor(a, b, t) {
  return [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ];
}

function drawMoonFace(ctx, cx, cy, radius, angleDeg, options = {}) {
  const darkColor = options.darkColor || [24, 34, 58];
  const lightColor = options.lightColor || [246, 238, 212];
  const craterColor = options.craterColor || 'rgba(94, 104, 136, 0.18)';
  const size = Math.max(8, Math.floor(radius * 2));
  const image = ctx.createImageData(size, size);
  const sunAngle = angleToRadians(angleDeg);
  const sx = Math.sin(sunAngle);
  const sy = 0;
  const sz = -Math.cos(sunAngle);

  for (let py = 0; py < size; py += 1) {
    for (let px = 0; px < size; px += 1) {
      const nx = (px + 0.5 - radius) / radius;
      const ny = (py + 0.5 - radius) / radius;
      const rr = nx * nx + ny * ny;
      const index = (py * size + px) * 4;
      if (rr > 1) {
        image.data[index + 3] = 0;
        continue;
      }

      const nz = Math.sqrt(1 - rr);
      const lit = Math.max(0, nx * sx + ny * sy + nz * sz);
      const base = lit > 0 ? 0.35 + 0.65 * lit : 0.08;
      const texture = 0.04 * Math.sin(nx * 13 + ny * 7) + 0.03 * Math.cos(nx * 19 - ny * 11);
      const brightness = clamp(base + texture, 0.04, 1);
      const rgb = lerpColor(darkColor, lightColor, brightness);

      image.data[index] = rgb[0];
      image.data[index + 1] = rgb[1];
      image.data[index + 2] = rgb[2];
      image.data[index + 3] = 255;
    }
  }

  ctx.save();
  if (options.glow) {
    ctx.shadowBlur = radius * 0.4;
    ctx.shadowColor = 'rgba(255, 223, 147, 0.35)';
  }
  ctx.putImageData(image, Math.round(cx - radius), Math.round(cy - radius));
  ctx.restore();

  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, TAU);
  ctx.clip();
  ctx.fillStyle = craterColor;
  const craters = [
    [-0.35, -0.18, 0.14],
    [0.22, -0.3, 0.1],
    [0.1, 0.22, 0.16],
    [-0.08, -0.02, 0.07],
    [0.38, 0.06, 0.09],
  ];
  craters.forEach(([dx, dy, rr]) => {
    ctx.beginPath();
    ctx.arc(cx + dx * radius, cy + dy * radius, rr * radius, 0, TAU);
    ctx.fill();
  });
  ctx.restore();

  ctx.strokeStyle = options.strokeStyle || 'rgba(255, 255, 255, 0.14)';
  ctx.lineWidth = options.strokeWidth || 1.6;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, TAU);
  ctx.stroke();
}

function drawSmallOrbitMoon(ctx, x, y, radius) {
  ctx.fillStyle = '#18223d';
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, TAU);
  ctx.fill();

  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, TAU);
  ctx.clip();
  ctx.fillStyle = '#f6eed4';
  ctx.beginPath();
  ctx.arc(x, y, radius, Math.PI / 2, -Math.PI / 2, true);
  ctx.closePath();
  ctx.fill();
  ctx.restore();

  ctx.strokeStyle = 'rgba(255,255,255,0.22)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, TAU);
  ctx.stroke();
}

function drawHeroMoon() {
  const { ctx, width, height } = resizeCanvas(elements.heroMoonCanvas);
  ctx.clearRect(0, 0, width, height);
  drawMoonFace(ctx, width / 2, height / 2, Math.min(width, height) * 0.37, 180, { glow: true });
}

function drawOrbitScene() {
  const { ctx, width, height } = resizeCanvas(elements.orbitCanvas);
  ctx.clearRect(0, 0, width, height);

  const sunX = width * 0.12;
  const sunY = height * 0.52;
  const earthX = width * 0.52;
  const earthY = height * 0.52;
  const orbitRadius = Math.min(width, height) * 0.29;
  const moonRadius = Math.max(12, Math.min(width, height) * 0.038);
  const angle = angleToRadians(state.angleDeg);
  const moonX = earthX - Math.cos(angle) * orbitRadius;
  const moonY = earthY + Math.sin(angle) * orbitRadius;

  orbitInteraction.centerX = earthX;
  orbitInteraction.centerY = earthY;
  orbitInteraction.radius = orbitRadius;

  ctx.fillStyle = 'rgba(255,255,255,0.03)';
  ctx.fillRect(0, 0, width, height);

  ctx.save();
  ctx.strokeStyle = 'rgba(255, 212, 91, 0.18)';
  ctx.lineWidth = 3;
  for (let beam = -2; beam <= 2; beam += 1) {
    const offset = beam * 22;
    ctx.beginPath();
    ctx.moveTo(sunX + 36, sunY + offset);
    ctx.lineTo(width - 20, sunY + offset);
    ctx.stroke();
  }
  ctx.restore();

  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.22)';
  ctx.setLineDash([8, 8]);
  ctx.beginPath();
  ctx.arc(earthX, earthY, orbitRadius, 0, TAU);
  ctx.stroke();
  ctx.restore();

  const markers = PHASES.map((phase) => {
    const a = angleToRadians(phase.angle);
    return {
      x: earthX - Math.cos(a) * orbitRadius,
      y: earthY + Math.sin(a) * orbitRadius,
      phase,
    };
  });

  ctx.font = '13px "Avenir Next", sans-serif';
  ctx.fillStyle = 'rgba(195, 209, 234, 0.92)';
  markers.forEach((marker) => {
    ctx.beginPath();
    ctx.arc(marker.x, marker.y, 3.5, 0, TAU);
    ctx.fill();
  });

  ctx.save();
  ctx.shadowBlur = 26;
  ctx.shadowColor = 'rgba(255, 212, 91, 0.72)';
  ctx.fillStyle = '#ffd45b';
  ctx.beginPath();
  ctx.arc(sunX, sunY, Math.min(width, height) * 0.09, 0, TAU);
  ctx.fill();
  ctx.restore();
  ctx.fillText('Sun', sunX - 12, sunY + Math.min(width, height) * 0.13);

  ctx.save();
  ctx.fillStyle = '#4fa7ff';
  ctx.beginPath();
  ctx.arc(earthX, earthY, Math.min(width, height) * 0.065, 0, TAU);
  ctx.fill();
  ctx.fillStyle = '#6ed49c';
  ctx.beginPath();
  ctx.arc(earthX + 6, earthY - 2, Math.min(width, height) * 0.02, 0, TAU);
  ctx.fill();
  ctx.restore();
  ctx.fillStyle = '#f8f6ef';
  ctx.fillText('Earth', earthX - 16, earthY + Math.min(width, height) * 0.12);

  ctx.save();
  ctx.strokeStyle = 'rgba(127, 214, 255, 0.28)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(earthX, earthY);
  ctx.lineTo(moonX, moonY);
  ctx.stroke();
  ctx.restore();

  drawSmallOrbitMoon(ctx, moonX, moonY, moonRadius);

  ctx.fillStyle = '#f8f6ef';
  ctx.beginPath();
  ctx.arc(moonX, moonY, moonRadius + 5, 0, TAU);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.28)';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.fillStyle = 'rgba(195, 209, 234, 0.9)';
  ctx.fillText('Moon', moonX - 14, moonY - moonRadius - 12);
  ctx.fillText('Sunlight', width - 108, sunY - 38);
}

function drawEarthView() {
  const { ctx, width, height } = resizeCanvas(elements.moonCanvas);
  ctx.clearRect(0, 0, width, height);
  drawMoonFace(ctx, width / 2, height / 2, Math.min(width, height) * 0.38, state.angleDeg, { glow: true });
}

function drawPhaseCards() {
  elements.phaseCanvases.forEach((canvas) => {
    const phase = PHASES[Number(canvas.dataset.phaseCanvas)];
    const { ctx, width, height } = resizeCanvas(canvas);
    ctx.clearRect(0, 0, width, height);
    drawMoonFace(ctx, width / 2, height / 2, Math.min(width, height) * 0.38, phase.angle);
  });
}

function drawShadowScene() {
  const { ctx, width, height } = resizeCanvas(elements.shadowCanvas);
  ctx.clearRect(0, 0, width, height);

  const sunX = width * 0.12;
  const sunY = height * 0.5;
  const earthX = width * 0.44;
  const earthY = height * 0.5;
  const moonX = width * 0.8;
  const moonY = state.shadowMode === 'phase' ? height * 0.34 : height * 0.5;

  ctx.fillStyle = 'rgba(255,255,255,0.03)';
  ctx.fillRect(0, 0, width, height);

  ctx.save();
  ctx.shadowBlur = 22;
  ctx.shadowColor = 'rgba(255, 212, 91, 0.66)';
  ctx.fillStyle = '#ffd45b';
  ctx.beginPath();
  ctx.arc(sunX, sunY, height * 0.12, 0, TAU);
  ctx.fill();
  ctx.restore();

  ctx.fillStyle = 'rgba(38, 57, 98, 0.72)';
  ctx.beginPath();
  ctx.moveTo(earthX + 18, earthY - 44);
  ctx.lineTo(width - 26, earthY - 78);
  ctx.lineTo(width - 26, earthY + 78);
  ctx.lineTo(earthX + 18, earthY + 44);
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = '#4fa7ff';
  ctx.beginPath();
  ctx.arc(earthX, earthY, height * 0.1, 0, TAU);
  ctx.fill();

  if (state.shadowMode === 'phase') {
    drawMoonFace(ctx, moonX, moonY, height * 0.08, 180, { glow: true, strokeWidth: 1.2 });
  } else {
    ctx.fillStyle = '#a86a6a';
    ctx.beginPath();
    ctx.arc(moonX, moonY, height * 0.08, 0, TAU);
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.24)';
    ctx.beginPath();
    ctx.arc(moonX, moonY, height * 0.08, 0, TAU);
    ctx.stroke();
  }

  ctx.fillStyle = '#f8f6ef';
  ctx.font = '14px "Avenir Next", sans-serif';
  ctx.fillText('Sun', sunX - 12, sunY + height * 0.18);
  ctx.fillText('Earth', earthX - 16, earthY + height * 0.16);
  ctx.fillText('Moon', moonX - 16, moonY + height * 0.16);
  ctx.fillText('Earth shadow', width * 0.58, earthY - 90);
}

function shuffle(items) {
  const copy = [...items];
  for (let index = copy.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [copy[index], copy[swapIndex]] = [copy[swapIndex], copy[index]];
  }
  return copy;
}

function makeQuizQuestion() {
  const answer = PHASES[Math.floor(Math.random() * PHASES.length)];
  const otherNames = shuffle(PHASES.filter((phase) => phase.name !== answer.name)).slice(0, 3);
  state.quizQuestion = {
    answer,
    options: shuffle([answer, ...otherNames]),
  };
  state.quizLocked = false;
  elements.quizFeedback.textContent = 'Pick an answer to check.';
  renderQuiz();
}

function renderQuiz() {
  const question = state.quizQuestion;
  if (!question) return;

  const { ctx, width, height } = resizeCanvas(elements.quizCanvas);
  ctx.clearRect(0, 0, width, height);
  drawMoonFace(ctx, width / 2, height / 2, Math.min(width, height) * 0.38, question.answer.angle, { glow: true });

  elements.quizOptions.innerHTML = '';
  question.options.forEach((option) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'quiz-option';
    button.textContent = option.name;
    button.addEventListener('click', () => {
      if (state.quizLocked) return;
      state.quizLocked = true;
      const correct = option.name === question.answer.name;
      if (correct) {
        state.quizScore += 1;
        elements.quizFeedback.textContent = `Yes! This is the ${question.answer.name}. ${question.answer.short}.`;
      } else {
        elements.quizFeedback.textContent = `Not this time. This is the ${question.answer.name}. ${question.answer.short}.`;
      }
      elements.quizScore.textContent = `Score: ${state.quizScore}`;
      Array.from(elements.quizOptions.children).forEach((choice) => {
        const isCorrect = choice.textContent === question.answer.name;
        choice.classList.add(isCorrect ? 'is-right' : 'is-wrong');
      });
    });
    elements.quizOptions.appendChild(button);
  });
}

function updateText() {
  const phase = getPhase(state.angleDeg);
  const illumination = getIllumination(state.angleDeg);
  elements.dayReadout.textContent = `About day ${getDayNumber(state.angleDeg).toFixed(1)} of 29.5`;
  elements.phaseName.textContent = phase.name;
  elements.phaseTrend.textContent = getTrend(state.angleDeg);
  elements.phaseStory.textContent = phase.story;
  elements.phaseDetail.textContent = `Bright part we can see: ${formatPercent(illumination)}`;
  elements.playButton.textContent = state.playing ? 'Pause' : 'Play';
  elements.phaseSlider.value = String(Math.round(normalizeAngle(state.angleDeg)));
  elements.phaseCards.forEach((card) => {
    const isActive = Number(card.dataset.phaseIndex) === getPhaseIndex(state.angleDeg);
    card.classList.toggle('is-active', isActive);
  });
  elements.shadowButtons.forEach((button) => {
    button.classList.toggle('is-active', button.dataset.shadowMode === state.shadowMode);
  });
  elements.shadowCopy.textContent = state.shadowMode === 'phase'
    ? 'On most nights, the Moon is a little above or below Earth\'s shadow, so the phase comes only from sunlight and our viewing angle.'
    : 'A lunar eclipse is special. Earth lines up with the Sun and Moon, so Earth\'s shadow really does fall on the Moon.';
}

function renderAll() {
  updateText();
  drawOrbitScene();
  drawEarthView();
  drawShadowScene();
}

function setAngle(angleDeg) {
  state.angleDeg = normalizeAngle(angleDeg);
  renderAll();
}

function angleFromPointer(clientX, clientY) {
  const rect = elements.orbitCanvas.getBoundingClientRect();
  const px = clientX - rect.left;
  const py = clientY - rect.top;
  const dx = px - orbitInteraction.centerX;
  const dy = py - orbitInteraction.centerY;
  return normalizeAngle(Math.atan2(dy, -dx) * 180 / Math.PI);
}

function tick(timestamp) {
  if (!state.playing) return;
  if (state.lastFrame) {
    const elapsed = (timestamp - state.lastFrame) / 1000;
    state.angleDeg = normalizeAngle(state.angleDeg + elapsed * 28);
    renderAll();
  }
  state.lastFrame = timestamp;
  window.requestAnimationFrame(tick);
}

function wireOrbitDrag() {
  const canvas = elements.orbitCanvas;

  canvas.addEventListener('pointerdown', (event) => {
    state.orbitDragging = true;
    canvas.classList.add('is-dragging');
    canvas.setPointerCapture(event.pointerId);
    setAngle(angleFromPointer(event.clientX, event.clientY));
  });

  canvas.addEventListener('pointermove', (event) => {
    if (!state.orbitDragging) return;
    setAngle(angleFromPointer(event.clientX, event.clientY));
  });

  ['pointerup', 'pointercancel'].forEach((type) => {
    canvas.addEventListener(type, () => {
      state.orbitDragging = false;
      canvas.classList.remove('is-dragging');
    });
  });
}

function wireInputs() {
  elements.phaseSlider.addEventListener('input', () => {
    setAngle(Number(elements.phaseSlider.value));
  });

  elements.playButton.addEventListener('click', () => {
    state.playing = !state.playing;
    state.lastFrame = 0;
    updateText();
    if (state.playing) {
      window.requestAnimationFrame(tick);
    }
  });

  elements.phaseCards.forEach((card) => {
    card.addEventListener('click', () => {
      state.playing = false;
      setAngle(PHASES[Number(card.dataset.phaseIndex)].angle);
    });
  });

  elements.shadowButtons.forEach((button) => {
    button.addEventListener('click', () => {
      state.shadowMode = button.dataset.shadowMode;
      renderAll();
    });
  });

  elements.nextQuestion.addEventListener('click', makeQuizQuestion);
  window.addEventListener('resize', renderAll);
}

drawHeroMoon();
drawPhaseCards();
makeQuizQuestion();
wireOrbitDrag();
wireInputs();
renderAll();
