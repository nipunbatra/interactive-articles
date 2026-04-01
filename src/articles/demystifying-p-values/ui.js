export function renderBubbles(container, values, type = 'neutral') {
  container.innerHTML = '';
  values.forEach((v, i) => {
    const el = document.createElement('div');
    el.className = `bubble ${type}`;
    el.textContent = v;
    el.style.animationDelay = `${i * 0.04}s`;
    container.appendChild(el);
  });
}

export function setupCanvas(canvas, logicalWidth, logicalHeight) {
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== logicalWidth * dpr) {
    canvas.width = logicalWidth * dpr;
    canvas.height = logicalHeight * dpr;
  }
  const ctx = canvas.getContext('2d');
  ctx.resetTransform();
  ctx.scale(dpr, dpr);
  return { ctx, w: logicalWidth, h: logicalHeight };
}