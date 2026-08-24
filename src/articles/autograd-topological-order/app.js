'use strict';

(() => {
  const EXPECTED_ORDER = ["w", "x", "m", "b", "a", "y", "e", "L"];
  const EXPECTED_BACKWARD = [...EXPECTED_ORDER].reverse();
  const PARENT_LINKS = {
    w: [],
    x: [],
    m: [{ value: "w", local_grad: 3 }, { value: "x", local_grad: 2 }],
    b: [],
    a: [{ value: "m", local_grad: 1 }, { value: "b", local_grad: 1 }],
    y: [],
    e: [{ value: "a", local_grad: 1 }, { value: "y", local_grad: -1 }],
    L: [{ value: "e", local_grad: -6 }],
  };

  function escapeHTML(value) {
    const span = document.createElement("span");
    span.textContent = String(value);
    return span.innerHTML;
  }

  function createDefaultPayload() {
    const safeOrder = [];
    const seen = new Set();
    const seenValues = [];
    const callStack = [];
    const events = [];

    function snapshot(kind, details) {
      const event = {
        kind,
        stack: [...callStack],
        seen: [...seenValues],
        order: [...safeOrder],
        line: details.line,
        message: details.message,
      };
      ["node", "parent", "link_index", "local_grad"].forEach((key) => {
        if (details[key] !== undefined) event[key] = details[key];
      });
      events.push(event);
    }

    snapshot("ready", {
      line: 8,
      message:
        "Press <strong>Next</strong> or <strong>Play</strong> to call " +
        "<code>append_after_parents(L)</code>.",
    });

    function appendAfterParents(node) {
      callStack.push(node);
      const label = escapeHTML(node);

      if (seen.has(node)) {
        snapshot("skip", {
          node,
          line: 2,
          message:
            "<strong>" + label + "</strong> is already in <code>seen</code>, so this repeated " +
            "call returns without appending it again.",
        });
        callStack.pop();
        return;
      }

      seen.add(node);
      seenValues.push(node);
      const links = PARENT_LINKS[node];
      const parentCount = links.length;
      const enterMessage = parentCount
        ? "Enter <strong>" + label + "</strong>. It is new, so mark it seen; this call must now visit " +
          (parentCount === 1 ? "its parent." : "all " + parentCount + " parents.")
        : "Enter leaf <strong>" + label + "</strong>. It is new, so mark it seen. " +
          "It has no ParentLinks, so it can be appended next.";

      snapshot("enter", { node, line: 3, message: enterMessage });

      links.forEach((link, linkIndex) => {
        const parentLabel = escapeHTML(link.value);
        snapshot("follow", {
          node,
          parent: link.value,
          link_index: linkIndex,
          local_grad: link.local_grad,
          line: 5,
          message:
            "Follow <code>" + label + ".parents[" + linkIndex + "]</code> to " +
            "<strong>" + parentLabel + "</strong>, then call " +
            "<code>append_after_parents(" + parentLabel + ")</code>.",
        });

        appendAfterParents(link.value);

        snapshot("unwind", {
          node,
          parent: link.value,
          line: 5,
          message:
            "The call for <strong>" + parentLabel + "</strong> has returned to " +
            "<strong>" + label + "</strong>. " +
            (linkIndex + 1 < parentCount
              ? "Continue to the next ParentLink."
              : "All of this node's parent calls are now finished."),
        });
      });

      safeOrder.push(node);
      snapshot("append", {
        node,
        line: 6,
        message:
          "Append <strong>" + label + "</strong> to <code>safe_order</code>. Every direct " +
          "parent of " + label + " is already earlier in the list.",
      });
      callStack.pop();
    }

    appendAfterParents("L");
    snapshot("return", {
      line: 9,
      message:
        "The recursive helper is finished. Return the dependency-first " +
        "<code>safe_order</code> unchanged.",
    });
    snapshot("reverse", {
      line: 11,
      message:
        "Now move into <code>backward</code>. Its loop reads " +
        "<code>reversed(safe_order)</code> before performing any local-gradient arithmetic.",
    });

    if (events.length !== 33) {
      throw new Error("The default topological-order trace must contain exactly 33 frames.");
    }

    return {
      events,
      parents: Object.fromEntries(
        EXPECTED_ORDER.map((node) => [node, PARENT_LINKS[node].map((link) => link.value)])
      ),
      order: [...safeOrder],
      backward: [...safeOrder].reverse(),
    };
  }

  function sanitizeMessage(message) {
    const template = document.createElement("template");
    template.innerHTML = message;
    template.content.querySelectorAll("*").forEach((element) => {
      if (!["STRONG", "CODE", "BR"].includes(element.tagName)) {
        element.replaceWith(document.createTextNode(element.textContent || ""));
        return;
      }
      [...element.attributes].forEach((attribute) => element.removeAttribute(attribute.name));
    });
    return template.innerHTML;
  }

  function arraysEqual(left, right) {
    return (
      Array.isArray(left) &&
      Array.isArray(right) &&
      left.length === right.length &&
      left.every((value, index) => value === right[index])
    );
  }

  function structuralEvent(event) {
    return {
      kind: event.kind,
      stack: event.stack,
      seen: event.seen,
      order: event.order,
      line: event.line,
      node: event.node,
      parent: event.parent,
      link_index: event.link_index,
      local_grad: event.local_grad,
    };
  }

  const defaultPayload = createDefaultPayload();

  function validateNotebookMessage(message) {
    if (
      !message ||
      typeof message !== "object" ||
      message.type !== "scalar-topology-animation-data" ||
      typeof message.channel !== "string" ||
      message.channel.length < 1 ||
      message.channel.length > 256 ||
      /[\u0000-\u001f\u007f]/.test(message.channel)
    ) {
      return null;
    }

    const candidate = message.payload && typeof message.payload === "object"
      ? message.payload
      : message;

    if (
      !arraysEqual(candidate.order, EXPECTED_ORDER) ||
      !arraysEqual(candidate.backward, EXPECTED_BACKWARD) ||
      !candidate.parents ||
      typeof candidate.parents !== "object" ||
      !Array.isArray(candidate.events) ||
      candidate.events.length !== defaultPayload.events.length
    ) {
      return null;
    }

    for (const node of EXPECTED_ORDER) {
      if (!arraysEqual(candidate.parents[node], defaultPayload.parents[node])) return null;
    }

    const events = [];
    for (let index = 0; index < defaultPayload.events.length; index += 1) {
      const event = candidate.events[index];
      const expected = defaultPayload.events[index];
      if (
        !event ||
        typeof event !== "object" ||
        typeof event.message !== "string" ||
        event.message.length < 1 ||
        event.message.length > 800 ||
        JSON.stringify(structuralEvent(event)) !== JSON.stringify(structuralEvent(expected))
      ) {
        return null;
      }
      events.push({ ...structuralEvent(event), message: sanitizeMessage(event.message) });
    }

    return {
      channel: message.channel,
      payload: {
        events,
        parents: Object.fromEntries(
          EXPECTED_ORDER.map((node) => [node, [...candidate.parents[node]]])
        ),
        order: [...candidate.order],
        backward: [...candidate.backward],
      },
    };
  }

  const root = document.querySelector("[data-toposort-animation]");
  if (!root || root.dataset.enhanced === "true") return;
  root.dataset.enhanced = "true";
  const instanceId = "toposort-animation";
  root.id = root.id || instanceId;
  const svg = root.querySelector("svg.tsa-graph");
  const svgTitle = svg.querySelector("title");
  const svgDesc = svg.querySelector("desc");
  svgTitle.id = instanceId + "-graph-title";
  svgDesc.id = instanceId + "-graph-desc";
  svg.setAttribute("aria-labelledby", svgTitle.id + " " + svgDesc.id);

  let payload = defaultPayload;
  let events = payload.events;
  const labels = {
    ready:"Ready", enter:"Enter", follow:"Follow link", skip:"Skip seen",
    append:"Append", unwind:"Unwind", return:"Return", reverse:"Reverse"
  };
  const buttons = {
    reset:root.querySelector('[data-action="reset"]'),
    previous:root.querySelector('[data-action="previous"]'),
    play:root.querySelector('[data-action="play"]'),
    next:root.querySelector('[data-action="next"]')
  };
  const speed = root.querySelector('[data-action="speed"]');
  const actionBadge = root.querySelector("[data-action-badge]");
  const status = root.querySelector("[data-status]");
  const stepCount = root.querySelector("[data-step-count]");
  const backwardPanel = root.querySelector("[data-backward]");
  const linkState = root.querySelector("[data-link-state]");
  const nodeElements = [...root.querySelectorAll("[data-node]")];
  const edgeElements = [...root.querySelectorAll(".tsa-edge")];
  const codeLines = [...root.querySelectorAll("[data-line]")];
  let index = 0;
  let timer = null;
  let playing = false;
  let lastReportedHeight = 0;
  let messageChannel = null;

  function reportHeight() {
    const height = Math.ceil(root.getBoundingClientRect().height + 16);
    if (height === lastReportedHeight) return;
    lastReportedHeight = height;
    window.parent.postMessage({
      type: "scalar-topology-animation-height",
      channel: messageChannel,
      height,
    }, "*");
  }

  function chips(target, values, kind, emptyText) {
    target.replaceChildren();
    if (!values.length) {
      const empty = document.createElement("span");
      empty.className = "tsa-empty";
      empty.textContent = emptyText;
      target.append(empty);
      return;
    }
    values.forEach((value, position) => {
      if (position) {
        const arrow = document.createElement("span");
        arrow.className = "tsa-chip-arrow";
        arrow.textContent = "→";
        arrow.setAttribute("aria-hidden", "true");
        target.append(arrow);
      }
      const chip = document.createElement("span");
      chip.className = `tsa-chip ${kind}`;
      chip.textContent = value;
      target.append(chip);
    });
  }

  function stop() {
    if (timer !== null) window.clearTimeout(timer);
    timer = null;
    playing = false;
    buttons.play.textContent = "Play";
    buttons.play.setAttribute("aria-pressed", "false");
  }

  function render() {
    const event = events[index];
    const backwardOrder = [...event.order].reverse();
    actionBadge.dataset.kind = event.kind;
    actionBadge.textContent = labels[event.kind];
    status.innerHTML = event.message;
    stepCount.textContent = `Step ${index} of ${events.length - 1}`;
    buttons.previous.disabled = index === 0;
    buttons.next.disabled = index === events.length - 1;

    codeLines.forEach(line => line.classList.toggle("is-active", Number(line.dataset.line) === event.line));
    nodeElements.forEach(nodeElement => {
      const name = nodeElement.dataset.node;
      nodeElement.classList.toggle("is-seen", event.seen.includes(name));
      nodeElement.classList.toggle("is-stack", event.stack.includes(name));
      nodeElement.classList.toggle("is-appended", event.order.includes(name));
      nodeElement.classList.toggle("is-current", event.node === name && event.kind !== "follow");
      nodeElement.classList.toggle("is-link-target", event.kind === "follow" && event.parent === name);
    });
    edgeElements.forEach(edge => edge.classList.toggle(
      "is-active",
      event.kind === "follow" && edge.dataset.child === event.node && edge.dataset.parent === event.parent
    ));

    linkState.replaceChildren();
    if (event.kind === "follow") {
      const linkName = document.createElement("code");
      linkName.textContent = `${event.node}.parents[${event.link_index}]`;
      const valueLine = document.createElement("strong");
      valueLine.style.display = "block";
      valueLine.textContent = `.value = ${event.parent}  ← followed now`;
      const localLine = document.createElement("span");
      localLine.className = "tsa-link-local";
      localLine.textContent = `.local_grad = ${event.local_grad}  · stored, not read by sorting`;
      linkState.append(linkName, valueLine, localLine);
    } else {
      const empty = document.createElement("span");
      empty.className = "tsa-empty";
      empty.textContent = "No link is being followed in this step.";
      linkState.append(empty);
    }

    chips(root.querySelector("[data-stack]"), event.stack, "stack", "empty");
    chips(root.querySelector("[data-seen]"), event.seen, "seen", "none yet");
    chips(root.querySelector("[data-order]"), event.order, "output", "empty");
    chips(root.querySelector("[data-backward-order]"), backwardOrder, "backward", "empty");
    backwardPanel.hidden = event.kind !== "reverse";
    window.requestAnimationFrame(reportHeight);
  }

  function scheduleNext() {
    if (!playing) return;
    timer = window.setTimeout(() => {
      if (index < events.length - 1) {
        index += 1;
        render();
        scheduleNext();
      } else {
        stop();
      }
    }, Number(speed.value));
  }

  function togglePlay() {
    if (playing) {
      stop();
      return;
    }
    if (index === events.length - 1) index = 0;
    playing = true;
    buttons.play.textContent = "Pause";
    buttons.play.setAttribute("aria-pressed", "true");
    render();
    scheduleNext();
  }

  buttons.reset.addEventListener("click", () => { stop(); index = 0; render(); });
  buttons.previous.addEventListener("click", () => { stop(); index = Math.max(0, index - 1); render(); });
  buttons.next.addEventListener("click", () => { stop(); index = Math.min(events.length - 1, index + 1); render(); });
  buttons.play.addEventListener("click", togglePlay);
  speed.addEventListener("change", () => {
    if (playing) { window.clearTimeout(timer); scheduleNext(); }
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stop();
  });
  window.addEventListener("resize", () => window.requestAnimationFrame(reportHeight));
  root.addEventListener("keydown", event => {
    if (event.target !== root) return;
    if (event.key === "ArrowRight") { event.preventDefault(); buttons.next.click(); }
    else if (event.key === "ArrowLeft") { event.preventDefault(); buttons.previous.click(); }
    else if (event.key === "Home") { event.preventDefault(); buttons.reset.click(); }
    else if (event.key === " ") { event.preventDefault(); togglePlay(); }
  });
  window.addEventListener("message", (event) => {
    if (window.parent === window || event.source !== window.parent) return;
    const validatedMessage = validateNotebookMessage(event.data);
    if (!validatedMessage) return;

    stop();
    messageChannel = validatedMessage.channel;
    payload = validatedMessage.payload;
    events = payload.events;
    index = 0;
    lastReportedHeight = -1;
    render();
    window.requestAnimationFrame(reportHeight);

    const payloadStatus = document.querySelector("[data-payload-status]");
    if (payloadStatus) payloadStatus.textContent = "Using the trace generated by the notebook.";
  });

  render();
})();
