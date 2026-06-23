(function () {
  const API_CANDIDATES = [
    "http://127.0.0.1:8008",
    "http://127.0.0.1:8009",
    "http://127.0.0.1:8010",
  ];
  const SIDEBAR_ID = "funasr-sidebar-host";
  const GROUP_SECONDS = 10;
  let apiBase = API_CANDIDATES[0];

  if (document.getElementById(SIDEBAR_ID)) {
    return;
  }

  function formatTime(seconds) {
    const total = Math.max(0, Math.floor(Number(seconds) || 0));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    if (h > 0) {
      return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    }
    return `${m}:${String(s).padStart(2, "0")}`;
  }

  function findVideo() {
    return document.querySelector("video");
  }

  function jumpTo(seconds) {
    const video = document.querySelector("video");
    if (!video) {
      setMessage("没有找到当前页面的视频播放器。");
      return;
    }
    video.currentTime = Number(seconds) || 0;
    video.play().catch(() => {});
  }

  function cleanBilibiliUrl() {
    const url = new URL(window.location.href);
    return `${url.origin}${url.pathname}${url.search}`;
  }

  const host = document.createElement("div");
  host.id = SIDEBAR_ID;
  host.innerHTML = `
    <aside class="funasr-sidebar">
      <div class="funasr-header">
        <div class="funasr-title">FunASR AI 学习模式</div>
        <div class="funasr-status" data-role="status">idle</div>
      </div>
      <div class="funasr-body">
        <div class="funasr-actions">
          <button data-role="analyze">分析当前视频</button>
          <button class="secondary" data-role="load-cache" hidden>加载已有结果</button>
          <button class="secondary" data-role="summarize" hidden>生成总结</button>
          <button class="secondary" data-role="mindmap" hidden>生成导图</button>
          <button class="secondary" data-role="frames" hidden>生成关键图</button>
          <button class="secondary" data-role="collapse">隐藏</button>
        </div>
        <div class="funasr-message" data-role="message">连接本地服务后，会显示可点击的字幕节点。第一版先打通 B站侧栏和跳转。</div>
        <div class="funasr-study-state" data-role="study-state" hidden></div>
        <div class="funasr-project-files" data-role="project-files" hidden>
          <div class="funasr-output-dir" data-role="output-dir"></div>
          <div class="funasr-file-actions">
            <a data-role="file-summary-md" class="disabled" href="#" target="_blank" rel="noreferrer">Summary.md</a>
            <a data-role="file-mindmap" class="disabled" href="#" target="_blank" rel="noreferrer">Mindmap.json</a>
            <a data-role="file-chat" class="disabled" href="#" target="_blank" rel="noreferrer">Chat.json</a>
            <button class="secondary" data-role="clear-chat" type="button">清空对话</button>
          </div>
        </div>
        <div class="funasr-tabs" role="tablist">
          <button class="funasr-tab active" data-tab="summary" type="button">总结</button>
          <button class="funasr-tab" data-tab="mindmap" type="button">导图</button>
          <button class="funasr-tab" data-tab="frames" type="button">关键图</button>
          <button class="funasr-tab" data-tab="chat" type="button">对话</button>
          <button class="funasr-tab" data-tab="transcript" type="button">字幕</button>
          <button class="funasr-tab" data-tab="log" type="button">日志</button>
        </div>
        <section class="funasr-panel active" data-panel="summary">
          <input class="funasr-search" data-role="api-key" type="password" placeholder="DeepSeek API Key（可选，默认读本地环境变量）" />
          <div class="funasr-summary-prompt-panel" data-role="summary-prompt-panel" hidden>
            <textarea class="funasr-summary-prompt" data-role="summary-prompt" rows="3" placeholder="可选：写下这次总结的额外要求，例如：整理成考试复习版，重点解释公式和易错点。"></textarea>
            <div class="funasr-summary-prompt-actions">
              <button data-role="summary-confirm" type="button">确认生成</button>
              <button class="secondary" data-role="summary-cancel" type="button">取消</button>
            </div>
          </div>
          <div class="funasr-summary" data-role="summary"></div>
        </section>
        <section class="funasr-panel" data-panel="mindmap" hidden>
          <div class="funasr-mindmap" data-role="mindmap-view"></div>
        </section>
        <section class="funasr-panel" data-panel="frames" hidden>
          <div class="funasr-frames" data-role="frames-view"></div>
        </section>
        <section class="funasr-panel" data-panel="chat" hidden>
          <div class="funasr-chat-title">视频问答</div>
          <div class="funasr-chat-list" data-role="chat-list"></div>
          <textarea class="funasr-chat-input" data-role="chat-input" rows="3" placeholder="问这个视频里的概念、片段或复习题..."></textarea>
          <button data-role="chat-send" type="button">发送</button>
        </section>
        <section class="funasr-panel" data-panel="transcript" hidden>
          <input class="funasr-search" data-role="search" type="search" placeholder="搜索字幕" />
          <div class="funasr-list" data-role="list"></div>
        </section>
        <section class="funasr-panel" data-panel="log" hidden>
          <pre class="funasr-log" data-role="log"></pre>
        </section>
      </div>
    </aside>
  `;
  document.body.appendChild(host);

  const statusEl = host.querySelector('[data-role="status"]');
  const messageEl = host.querySelector('[data-role="message"]');
  const studyStateEl = host.querySelector('[data-role="study-state"]');
  const projectFilesEl = host.querySelector('[data-role="project-files"]');
  const outputDirEl = host.querySelector('[data-role="output-dir"]');
  const fileSummaryMdEl = host.querySelector('[data-role="file-summary-md"]');
  const fileMindmapEl = host.querySelector('[data-role="file-mindmap"]');
  const fileChatEl = host.querySelector('[data-role="file-chat"]');
  const apiKeyEl = host.querySelector('[data-role="api-key"]');
  const summaryPromptPanelEl = host.querySelector('[data-role="summary-prompt-panel"]');
  const summaryPromptEl = host.querySelector('[data-role="summary-prompt"]');
  const searchEl = host.querySelector('[data-role="search"]');
  const summaryEl = host.querySelector('[data-role="summary"]');
  const mindmapEl = host.querySelector('[data-role="mindmap-view"]');
  const framesEl = host.querySelector('[data-role="frames-view"]');
  const chatListEl = host.querySelector('[data-role="chat-list"]');
  const chatInputEl = host.querySelector('[data-role="chat-input"]');
  const listEl = host.querySelector('[data-role="list"]');
  const logEl = host.querySelector('[data-role="log"]');
  const analyzeButton = host.querySelector('[data-role="analyze"]');
  const loadCacheButton = host.querySelector('[data-role="load-cache"]');
  const summarizeButton = host.querySelector('[data-role="summarize"]');
  const summaryConfirmButton = host.querySelector('[data-role="summary-confirm"]');
  const summaryCancelButton = host.querySelector('[data-role="summary-cancel"]');
  const mindmapButton = host.querySelector('[data-role="mindmap"]');
  const framesButton = host.querySelector('[data-role="frames"]');
  const chatSendButton = host.querySelector('[data-role="chat-send"]');
  const clearChatButton = host.querySelector('[data-role="clear-chat"]');
  const collapseButton = host.querySelector('[data-role="collapse"]');
  let cachedItem = null;
  let groupedSegments = [];
  let liveTranscriptActivated = false;
  let learningState = {
    transcript: false,
    summary: false,
    mindmap: false,
    chatTurns: 0,
    outputDir: "",
  };
  let mindmapViewState = {
    scale: 1,
    x: 0,
    y: 0,
    collapsed: new Set(),
    selectedMindmapNode: null,
    dragging: false,
    dragStartX: 0,
    dragStartY: 0,
    dragOriginX: 0,
    dragOriginY: 0,
  };

  function setActiveTab(name) {
    for (const tab of host.querySelectorAll("[data-tab]")) {
      tab.classList.toggle("active", tab.dataset.tab === name);
    }
    for (const panel of host.querySelectorAll("[data-panel]")) {
      const active = panel.dataset.panel === name;
      panel.hidden = !active;
      panel.classList.toggle("active", active);
    }
  }

  function setStatus(value) {
    statusEl.textContent = value;
  }

  function setMessage(value) {
    messageEl.textContent = value;
  }

  function setLog(value) {
    logEl.textContent = value || "";
    logEl.scrollTop = logEl.scrollHeight;
  }

  function messageForJobState(state) {
    return "";
  }

  function showSummaryPromptPanel() {
    summaryPromptPanelEl.hidden = false;
    setActiveTab("summary");
    summaryPromptEl.focus();
  }

  function hideSummaryPromptPanel() {
    summaryPromptPanelEl.hidden = true;
  }

  function setFileLink(anchor, href) {
    if (href) {
      anchor.href = `${apiBase}${href}`;
      anchor.classList.remove("disabled");
    } else {
      anchor.href = "#";
      anchor.classList.add("disabled");
    }
  }

  function renderProjectFiles(item) {
    if (!item || !item.found) {
      projectFilesEl.hidden = true;
      return;
    }
    projectFilesEl.hidden = false;
    outputDirEl.textContent = `输出目录：${item.output_dir || "-"}`;
    const files = item.files || {};
    setFileLink(fileSummaryMdEl, files.summary_md);
    setFileLink(fileMindmapEl, files.mindmap);
    setFileLink(fileChatEl, files.chat);
  }

  function renderStudyState(nextState) {
    learningState = { ...learningState, ...(nextState || {}) };
    const hasAny =
      learningState.transcript ||
      learningState.summary ||
      learningState.mindmap ||
      learningState.chatTurns > 0;
    studyStateEl.hidden = !hasAny;
    studyStateEl.textContent = "";
    if (!hasAny) {
      return;
    }

    const states = [
      ["字幕", learningState.transcript],
      ["总结", learningState.summary],
      ["导图", learningState.mindmap],
      [`对话 ${learningState.chatTurns} 轮`, learningState.chatTurns > 0],
    ];
    for (const [label, ready] of states) {
      const badge = document.createElement("span");
      badge.className = `funasr-state-badge ${ready ? "ready" : "missing"}`;
      badge.textContent = label;
      studyStateEl.appendChild(badge);
    }
  }

  function groupSegments(segments) {
    const grouped = [];
    let current = null;
    for (const raw of segments) {
      const start = Number(raw.start) || 0;
      const end = Number(raw.end) || start;
      const text = String(raw.text || "").trim();
      if (!text) {
        continue;
      }
      if (!current || start - current.start >= GROUP_SECONDS) {
        current = { start, end, text };
        grouped.push(current);
      } else {
        current.end = Math.max(current.end, end);
        current.text = `${current.text} ${text}`.trim();
      }
    }
    return grouped;
  }

  function filterSegments(query) {
    const needle = query.trim().toLowerCase();
    if (!needle) {
      return groupedSegments;
    }
    return groupedSegments.filter((item) => item.text.toLowerCase().includes(needle));
  }

  function highlightCurrentSegment() {
    const video = findVideo();
    if (!video) {
      return;
    }
    const now = Number(video.currentTime) || 0;
    for (const row of listEl.querySelectorAll(".funasr-item")) {
      const start = Number(row.dataset.start) || 0;
      const end = Number(row.dataset.end) || start + GROUP_SECONDS;
      const isActive = now >= start && now < end;
      row.classList.toggle("funasr-active", isActive);
    }
  }

  function renderSegmentRows(segments) {
    listEl.textContent = "";
    if (!segments.length) {
      setMessage("没有匹配的字幕节点。");
      return;
    }
    setMessage(`已显示 ${segments.length} 个约 ${GROUP_SECONDS}s 字幕节点，点击时间戳可跳转当前视频。`);
    for (const item of segments) {
      const row = document.createElement("div");
      row.className = "funasr-item";
      row.dataset.start = String(item.start);
      row.dataset.end = String(item.end || item.start + GROUP_SECONDS);

      const timeButton = document.createElement("button");
      timeButton.className = "funasr-time";
      timeButton.textContent = `[${formatTime(item.start)}-${formatTime(item.end || item.start)}]`;
      timeButton.addEventListener("click", () => jumpTo(item.start));

      const text = document.createElement("div");
      text.className = "funasr-text";
      text.textContent = item.text || "";

      row.appendChild(timeButton);
      row.appendChild(text);
      listEl.appendChild(row);
    }
    highlightCurrentSegment();
  }

  function renderSegments(segments) {
    groupedSegments = groupSegments(segments);
    renderSegmentRows(filterSegments(searchEl.value || ""));
    renderStudyState({ transcript: groupedSegments.length > 0 });
  }

  function parseTimestampToSeconds(value) {
    const parts = String(value || "").split(":").map((part) => Number(part));
    if (parts.some((part) => Number.isNaN(part))) {
      return 0;
    }
    if (parts.length === 3) {
      return parts[0] * 3600 + parts[1] * 60 + parts[2];
    }
    if (parts.length === 2) {
      return parts[0] * 60 + parts[1];
    }
    return parts[0] || 0;
  }

  function appendRichTextWithTimestamps(container, text) {
    const pattern = /(\[?(?:(?:\d{1,2}:)?\d{1,2}:\d{2})(?:\s*[-~—]\s*(?:(?:\d{1,2}:)?\d{1,2}:\d{2}))?\]?)/g;
    let lastIndex = 0;
    for (const match of String(text || "").matchAll(pattern)) {
      if (match.index > lastIndex) {
        container.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
      }
      const raw = match[0];
      const clean = raw.replace(/[\[\]]/g, "");
      const startText = clean.split(/[-~—]/)[0].trim();
      const button = document.createElement("button");
      button.className = "funasr-timestamp-link";
      button.type = "button";
      button.textContent = raw;
      button.addEventListener("click", () => jumpTo(parseTimestampToSeconds(startText)));
      container.appendChild(button);
      lastIndex = match.index + raw.length;
    }
    if (lastIndex < String(text || "").length) {
      container.appendChild(document.createTextNode(String(text || "").slice(lastIndex)));
    }
  }

  function appendInlineMarkdown(container, text) {
    const parts = String(text || "").split(/(\*\*[^*]+\*\*)/g);
    for (const part of parts) {
      if (part.startsWith("**") && part.endsWith("**")) {
        const strong = document.createElement("strong");
        appendRichTextWithTimestamps(strong, part.slice(2, -2));
        container.appendChild(strong);
      } else {
        appendRichTextWithTimestamps(container, part);
      }
    }
  }

  function renderMarkdown(markdown) {
    const root = document.createElement("div");
    root.className = "funasr-markdown-rendered";
    const lines = String(markdown || "").split(/\r?\n/);
    let list = null;

    function closeList() {
      if (list) {
        root.appendChild(list);
        list = null;
      }
    }

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) {
        closeList();
        continue;
      }
      const heading = trimmed.match(/^(#{1,4})\s+(.+)$/);
      if (heading) {
        closeList();
        const level = Math.min(4, heading[1].length + 2);
        const node = document.createElement(`h${level}`);
        appendInlineMarkdown(node, heading[2]);
        root.appendChild(node);
        continue;
      }
      const bullet = trimmed.match(/^[-*]\s+(.+)$/);
      if (bullet) {
        if (!list) {
          list = document.createElement("ul");
        }
        const item = document.createElement("li");
        appendInlineMarkdown(item, bullet[1]);
        list.appendChild(item);
        continue;
      }
      closeList();
      const paragraph = document.createElement("p");
      appendInlineMarkdown(paragraph, trimmed);
      root.appendChild(paragraph);
    }
    closeList();
    return root;
  }

  function renderSummary(summary, options = {}) {
    summaryEl.textContent = "";
    const title = document.createElement("div");
    title.className = "funasr-summary-title";
    title.textContent = summary.title || "视频学习笔记";
    summaryEl.appendChild(title);

    if (Array.isArray(summary.takeaways) && summary.takeaways.length) {
      const list = document.createElement("ul");
      list.className = "funasr-takeaways";
      for (const takeaway of summary.takeaways) {
        const item = document.createElement("li");
        item.textContent = takeaway;
        list.appendChild(item);
      }
      summaryEl.appendChild(list);
    }

    if (Array.isArray(summary.highlights) && summary.highlights.length) {
      const highlights = document.createElement("div");
      highlights.className = "funasr-highlights";
      for (const highlight of summary.highlights) {
        const row = document.createElement("button");
        row.className = "funasr-highlight";
        row.type = "button";
        row.addEventListener("click", () => jumpTo(highlight.start));
        row.textContent = `[${formatTime(highlight.start)}] ${highlight.title || "重点片段"}：${highlight.summary || ""}`;
        highlights.appendChild(row);
      }
      summaryEl.appendChild(highlights);
    }

    if (summary.markdown) {
      summaryEl.appendChild(renderMarkdown(summary.markdown));
    }
    renderStudyState({ summary: true });
    if (options.activate !== false) {
      setActiveTab("summary");
    }
  }

  function nodeIdForPath(path) {
    return path.join("-");
  }

  function visibleMindmapNodes(nodes, depth = 0, path = []) {
    const out = [];
    nodes.forEach((node, index) => {
      const nextPath = [...path, index];
      const id = nodeIdForPath(nextPath);
      const children = Array.isArray(node.children) ? node.children : [];
      out.push({ id, node, depth, hasChildren: children.length > 0 });
      if (!mindmapViewState.collapsed.has(id)) {
        out.push(...visibleMindmapNodes(children, depth + 1, nextPath));
      }
    });
    return out;
  }

  function layoutMindmapBranches(mindmap) {
    const nodeHeight = 42;
    const rootHeight = 54;
    const rootWidth = 210;
    const branchGap = 80;
    const levelGap = 240;
    const siblingGap = 24;
    const root = {
      id: "root",
      node: {
        title: mindmap.title || "视频思维导图",
        summary: mindmap.summary || "",
        kind: "root",
        children: mindmap.nodes || [],
      },
      depth: 0,
      side: 0,
      x: 720,
      y: 340,
      width: rootWidth,
      height: rootHeight,
      hasChildren: true,
    };
    const laidOut = [root];
    const nodes = Array.isArray(mindmap.nodes) ? mindmap.nodes : [];

    function visibleChildren(node, id) {
      const children = Array.isArray(node.children) ? node.children : [];
      return mindmapViewState.collapsed.has(id) ? [] : children;
    }

    function subtreeHeight(node, id) {
      const children = visibleChildren(node, id);
      if (!children.length) return nodeHeight;
      const childHeights = children.map((child, index) => subtreeHeight(child, `${id}-${index}`));
      const childrenHeight =
        childHeights.reduce((total, value) => total + value, 0) + siblingGap * Math.max(0, childHeights.length - 1);
      return Math.max(nodeHeight, childrenHeight);
    }

    function placeNode(node, id, depth, side, parentId, top, height) {
      const children = visibleChildren(node, id);
      const y = top + height / 2 - nodeHeight / 2;
      const item = {
        id,
        parentId,
        node,
        depth,
        side,
        x:
          side > 0
            ? root.x + root.width + branchGap + levelGap * (depth - 1)
            : root.x - branchGap - (depth === 1 ? 190 : 174) - levelGap * (depth - 1),
        y,
        width: depth === 1 ? 190 : 174,
        height: nodeHeight,
        hasChildren: Array.isArray(node.children) && node.children.length > 0,
      };
      laidOut.push(item);

      const childHeights = children.map((child, index) => subtreeHeight(child, `${id}-${index}`));
      const childrenHeight =
        childHeights.reduce((total, value) => total + value, 0) + siblingGap * Math.max(0, childHeights.length - 1);
      let childTop = top + Math.max(0, height - childrenHeight) / 2;
      children.forEach((child, index) => {
        const childId = `${id}-${index}`;
        placeNode(child, childId, depth + 1, side, id, childTop, childHeights[index]);
        childTop += childHeights[index] + siblingGap;
      });
    }

    function placeSide(sideNodes, side) {
      const heights = sideNodes.map(({ node, id }) => subtreeHeight(node, id));
      const totalHeight =
        heights.reduce((total, value) => total + value, 0) + siblingGap * Math.max(0, heights.length - 1);
      let top = root.y + root.height / 2 - totalHeight / 2;
      sideNodes.forEach(({ node, id }, index) => {
        placeNode(node, id, 1, side, "root", top, heights[index]);
        top += heights[index] + siblingGap;
      });
    }

    const rightNodes = [];
    const leftNodes = [];
    nodes.forEach((node, index) => {
      const target = index % 2 === 0 ? rightNodes : leftNodes;
      target.push({ node, id: String(index) });
    });
    placeSide(rightNodes, 1);
    placeSide(leftNodes, -1);

    const minY = Math.min(...laidOut.map((item) => item.y));
    if (minY < 40) {
      const offset = 40 - minY;
      laidOut.forEach((item) => {
        item.y += offset;
      });
    }
    const minX = Math.min(...laidOut.map((item) => item.x));
    if (minX < 40) {
      const offset = 40 - minX;
      laidOut.forEach((item) => {
        item.x += offset;
      });
    }
    return laidOut;
  }

  function setMindmapTransform(transformEl, svgEl) {
    const value = `translate(${mindmapViewState.x}px, ${mindmapViewState.y}px) scale(${mindmapViewState.scale})`;
    transformEl.style.transform = value;
    svgEl.style.transform = value;
  }

  function centerMindmapCanvas(viewportEl, transformEl, svgEl) {
    mindmapViewState.scale = 1;
    mindmapViewState.x = Math.max(16, Math.floor(viewportEl.clientWidth / 2) - 100);
    mindmapViewState.y = 24;
    setMindmapTransform(transformEl, svgEl);
  }

  function mindmapTimestamps(node) {
    const values = Array.isArray(node.timestamps) ? node.timestamps : [];
    const timestamps = values
      .map((value) => Number(value))
      .filter((value) => !Number.isNaN(value));
    if (!timestamps.length && node.time !== undefined) {
      timestamps.push(Number(node.time) || 0);
    }
    return [...new Set(timestamps)].slice(0, 4);
  }

  function appendMindmapList(card, className, items, limit) {
    if (!Array.isArray(items) || !items.length) return;
    for (const value of items.slice(0, limit)) {
      const row = document.createElement("span");
      row.className = className;
      row.textContent = String(value || "");
      card.appendChild(row);
    }
  }

  function appendMindmapEvidence(card, evidence) {
    if (!Array.isArray(evidence) || !evidence.length) return;
    for (const item of evidence.slice(0, 2)) {
      const row = document.createElement("span");
      row.className = "funasr-mindmap-evidence";
      const text = typeof item === "object" && item !== null ? item.text : item;
      const time = typeof item === "object" && item !== null ? item.time : undefined;
      row.textContent = time !== undefined ? `${formatTime(time)} ${text || ""}` : String(text || "");
      card.appendChild(row);
    }
  }

  function appendMindmapTimestampChips(card, node) {
    const timestamps = mindmapTimestamps(node);
    if (!timestamps.length) return;
    const row = document.createElement("span");
    row.className = "funasr-mindmap-timestamps";
    for (const seconds of timestamps) {
      const button = document.createElement("button");
      button.className = "funasr-mindmap-time";
      button.type = "button";
      button.textContent = formatTime(seconds);
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        jumpTo(seconds);
      });
      row.appendChild(button);
    }
    card.appendChild(row);
  }

  function renderMindmapInspector(container, node) {
    container.textContent = "";
    const selected = node || { title: "点击导图节点查看内容", summary: "节点保持紧凑，详细解释、证据和复习问题会显示在这里。" };
    const title = document.createElement("div");
    title.className = "funasr-mindmap-inspector-title";
    title.textContent = selected.title || "未命名节点";
    container.appendChild(title);
    if (selected.summary) {
      const summary = document.createElement("div");
      summary.className = "funasr-mindmap-inspector-summary";
      summary.textContent = selected.summary;
      container.appendChild(summary);
    }
    appendMindmapList(container, "funasr-mindmap-detail", selected.details, 5);
    appendMindmapEvidence(container, selected.evidence);
    appendMindmapList(container, "funasr-mindmap-question", selected.questions, 3);
    appendMindmapTimestampChips(container, selected);
  }

  function renderMindmapCanvas(mindmap, options = {}) {
    mindmapEl.textContent = "";
    const title = document.createElement("div");
    title.className = "funasr-mindmap-title";
    title.textContent = mindmap.title || "视频思维导图";
    mindmapEl.appendChild(title);

    const nodes = Array.isArray(mindmap.nodes) ? mindmap.nodes : [];
    if (!nodes.length) {
      const empty = document.createElement("div");
      empty.className = "funasr-chat-empty";
      empty.textContent = "还没有导图。";
      mindmapEl.appendChild(empty);
      return;
    }

    const toolbar = document.createElement("div");
    toolbar.className = "funasr-mindmap-toolbar";
    const inspector = document.createElement("div");
    inspector.className = "funasr-mindmap-inspector";
    const viewport = document.createElement("div");
    viewport.className = "funasr-mindmap-viewport";
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.classList.add("funasr-mindmap-svg");
    const transform = document.createElement("div");
    transform.className = "funasr-mindmap-transform";

    const laidOut = layoutMindmapBranches(mindmap);
    const byId = new Map(laidOut.map((item) => [item.id, item]));
    const canvasHeight = Math.max(520, Math.max(...laidOut.map((item) => item.y + item.height)) + 80);
    const canvasWidth = Math.max(1280, Math.max(...laidOut.map((item) => item.x + item.width)) + 260);
    svg.setAttribute("width", String(canvasWidth));
    svg.setAttribute("height", String(canvasHeight));
    renderMindmapInspector(inspector, mindmapViewState.selectedMindmapNode);

    for (const item of laidOut) {
      if (!item.parentId) continue;
      const parent = byId.get(item.parentId);
      if (!parent) continue;
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      const parentMidY = parent.y + parent.height / 2;
      const itemMidY = item.y + item.height / 2;
      const parentX = item.side < 0 ? parent.x : parent.x + parent.width;
      const itemX = item.side < 0 ? item.x + item.width : item.x;
      const curve = item.side * 52;
      path.setAttribute(
        "d",
        `M ${parentX} ${parentMidY} C ${parentX + curve} ${parentMidY}, ${itemX - curve} ${itemMidY}, ${itemX} ${itemMidY}`
      );
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", item.side < 0 ? "#c084fc" : "#38bdf8");
      path.setAttribute("stroke-width", item.depth === 1 ? "3" : "2");
      svg.appendChild(path);
    }

    for (const item of laidOut) {
      const node = item.node;
      const card = document.createElement("div");
      card.className = `funasr-mindmap-canvas-node ${item.id === "root" ? "funasr-mindmap-root-node" : ""}`;
      card.tabIndex = 0;
      card.setAttribute("role", "button");
      card.style.left = `${item.x}px`;
      card.style.top = `${item.y}px`;
      card.style.width = `${item.width}px`;
      card.style.height = `${item.height}px`;
      card.setAttribute(
        "aria-label",
        node.time !== undefined
          ? `跳转到 ${formatTime(node.time)}：${node.title || "未命名节点"}`
          : node.title || "未命名节点"
      );

      if (item.hasChildren) {
        const fold = document.createElement("span");
        fold.className = "funasr-mindmap-fold";
        fold.textContent = mindmapViewState.collapsed.has(item.id) ? "+" : "-";
        fold.addEventListener("click", (event) => {
          event.stopPropagation();
          if (mindmapViewState.collapsed.has(item.id)) {
            mindmapViewState.collapsed.delete(item.id);
          } else {
            mindmapViewState.collapsed.add(item.id);
          }
          renderMindmapCanvas(mindmap, { activate: false });
        });
        card.appendChild(fold);
      }

      if (node.kind) {
        const kind = document.createElement("span");
        kind.className = "funasr-mindmap-kind";
        kind.textContent = node.kind;
        card.appendChild(kind);
      }

      const label = document.createElement("span");
      label.className = "funasr-mindmap-node-title";
      label.textContent = node.title || "未命名节点";
      card.appendChild(label);
      const timestamps = mindmapTimestamps(node);
      if (timestamps.length) {
        const time = document.createElement("span");
        time.className = "funasr-mindmap-time";
        time.textContent = formatTime(timestamps[0]);
        card.appendChild(time);
      }
      card.title = node.summary || node.title || "";
      card.addEventListener("click", () => {
        mindmapViewState.selectedMindmapNode = node;
        renderMindmapInspector(inspector, node);
      });
      card.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          mindmapViewState.selectedMindmapNode = node;
          renderMindmapInspector(inspector, node);
        }
      });
      transform.appendChild(card);
    }

    function addTool(label, handler) {
      const button = document.createElement("button");
      button.className = "secondary";
      button.type = "button";
      button.textContent = label;
      button.addEventListener("click", handler);
      toolbar.appendChild(button);
    }

    addTool("居中", () => centerMindmapCanvas(viewport, transform, svg));
    addTool("+", () => {
      mindmapViewState.scale = Math.min(1.8, mindmapViewState.scale + 0.1);
      setMindmapTransform(transform, svg);
    });
    addTool("-", () => {
      mindmapViewState.scale = Math.max(0.4, mindmapViewState.scale - 0.1);
      setMindmapTransform(transform, svg);
    });
    addTool("展开", () => {
      mindmapViewState.collapsed.clear();
      renderMindmapCanvas(mindmap, { activate: false });
    });
    addTool("收起", () => {
      for (const item of laidOut) {
        if (item.hasChildren && item.id !== "root") mindmapViewState.collapsed.add(item.id);
      }
      renderMindmapCanvas(mindmap, { activate: false });
    });

    viewport.addEventListener("pointerdown", (event) => {
      if (event.target.closest(".funasr-mindmap-canvas-node")) return;
      mindmapViewState.dragging = true;
      mindmapViewState.dragStartX = event.clientX;
      mindmapViewState.dragStartY = event.clientY;
      mindmapViewState.dragOriginX = mindmapViewState.x;
      mindmapViewState.dragOriginY = mindmapViewState.y;
      viewport.setPointerCapture(event.pointerId);
    });
    viewport.addEventListener("pointermove", (event) => {
      if (!mindmapViewState.dragging) return;
      mindmapViewState.x = mindmapViewState.dragOriginX + event.clientX - mindmapViewState.dragStartX;
      mindmapViewState.y = mindmapViewState.dragOriginY + event.clientY - mindmapViewState.dragStartY;
      setMindmapTransform(transform, svg);
    });
    viewport.addEventListener("pointerup", () => {
      mindmapViewState.dragging = false;
    });
    viewport.addEventListener("wheel", (event) => {
      event.preventDefault();
      const delta = event.deltaY > 0 ? -0.08 : 0.08;
      mindmapViewState.scale = Math.max(0.4, Math.min(1.8, mindmapViewState.scale + delta));
      setMindmapTransform(transform, svg);
    }, { passive: false });

    viewport.appendChild(svg);
    viewport.appendChild(transform);
    mindmapEl.appendChild(toolbar);
    mindmapEl.appendChild(inspector);
    mindmapEl.appendChild(viewport);
    requestAnimationFrame(() => centerMindmapCanvas(viewport, transform, svg));
    renderStudyState({ mindmap: true });
    if (options.activate !== false) {
      setActiveTab("mindmap");
    }
  }

  function renderMindmap(mindmap, options = {}) {
    renderMindmapCanvas(mindmap, options);
  }

  function renderFrames(frames, options = {}) {
    framesEl.textContent = "";
    const items = Array.isArray(frames && frames.frames) ? frames.frames : [];
    if (!items.length) {
      const empty = document.createElement("div");
      empty.className = "funasr-chat-empty";
      empty.textContent = "还没有关键图。需要先生成总结，并且本地项目里保留了视频文件。";
      framesEl.appendChild(empty);
      return;
    }
    for (const frame of items) {
      const card = document.createElement("button");
      card.className = "funasr-frame-card";
      card.type = "button";
      card.addEventListener("click", () => jumpTo(frame.start));

      if (frame.image_url) {
        const image = document.createElement("img");
        image.src = `${apiBase}${frame.image_url}`;
        image.alt = frame.title || "关键画面";
        image.loading = "lazy";
        card.appendChild(image);
      }

      const meta = document.createElement("div");
      meta.className = "funasr-frame-meta";
      const time = document.createElement("span");
      time.className = "funasr-frame-time";
      time.textContent = `[${formatTime(frame.start)}]`;
      const title = document.createElement("strong");
      title.textContent = frame.title || "关键画面";
      meta.appendChild(time);
      meta.appendChild(title);
      card.appendChild(meta);

      if (frame.summary) {
        const summary = document.createElement("div");
        summary.className = "funasr-frame-summary";
        summary.textContent = frame.summary;
        card.appendChild(summary);
      }
      framesEl.appendChild(card);
    }
    if (options.activate !== false) {
      setActiveTab("frames");
    }
  }

  function renderChatSession(session) {
    chatListEl.textContent = "";
    const messages = Array.isArray(session && session.messages) ? session.messages : [];
    renderStudyState({ chatTurns: messages.length });
    if (!messages.length) {
      const empty = document.createElement("div");
      empty.className = "funasr-chat-empty";
      empty.textContent = "还没有对话，可以问：这个视频主要讲了什么？";
      chatListEl.appendChild(empty);
      return;
    }
    for (const message of messages) {
      const row = document.createElement("div");
      row.className = "funasr-chat-item";
      const question = document.createElement("div");
      question.className = "funasr-chat-question";
      question.textContent = message.question || "";
      const answer = document.createElement("div");
      answer.className = "funasr-chat-answer";
      answer.appendChild(renderMarkdown(message.answer || ""));
      row.appendChild(question);
      row.appendChild(answer);
      chatListEl.appendChild(row);
    }
    chatListEl.scrollTop = chatListEl.scrollHeight;
  }

  async function fetchJson(url) {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`${res.status} ${res.statusText}`);
    }
    return res.json();
  }

  async function detectApiBase() {
    for (const candidate of API_CANDIDATES) {
      try {
        await fetchJson(`${candidate}/health`);
        apiBase = candidate;
        return candidate;
      } catch (_) {
        // Try the next common local service port.
      }
    }
    throw new Error(`未找到本地服务，请确认已启动 ${API_CANDIDATES.join(" 或 ")}`);
  }

  async function loadTranscript(jobId) {
    const res = await fetch(`${apiBase}/api/jobs/${jobId}/files/json`);
    if (!res.ok) {
      return [];
    }
    const data = await res.json();
    return Array.isArray(data.segments) ? data.segments : [];
  }

  async function loadTranscriptFromUrl(url) {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`${res.status} ${res.statusText}`);
    }
    const data = await res.json();
    return Array.isArray(data.segments) ? data.segments : [];
  }

  async function lookupCachedResult() {
    try {
      const detected = await detectApiBase();
      const url = encodeURIComponent(cleanBilibiliUrl());
      const state = await fetchJson(`${detected}/api/library/lookup?url=${url}`);
      if (state.found && state.files && state.files.json) {
        cachedItem = state;
        renderProjectFiles(state);
        loadCacheButton.hidden = false;
        summarizeButton.hidden = false;
        mindmapButton.hidden = false;
        framesButton.hidden = false;
        setStatus("cached");
        renderStudyState({ transcript: true, outputDir: state.output_dir || "" });
        setMessage(`已找到本地结果，正在恢复历史学习记录：${state.output_dir}`);
        await autoRestoreCachedResult();
      }
    } catch (_) {
      // The user can still click analyze, which will show a clearer error.
    }
  }

  async function loadCachedResult(options = {}) {
    if (!cachedItem || !cachedItem.files || !cachedItem.files.json) {
      setMessage("没有可加载的本地结果。");
      return;
    }
    try {
      setStatus("loading-cache");
      const segments = await loadTranscriptFromUrl(`${apiBase}${cachedItem.files.json}`);
      renderSegments(segments);
      setStatus("cached");
      summarizeButton.hidden = false;
      mindmapButton.hidden = false;
      framesButton.hidden = false;
      if (options.activate !== false) {
        setActiveTab("transcript");
      }
    } catch (err) {
      setStatus("failed");
      setMessage(`加载已有结果失败：${err.message}`);
    }
  }

  async function loadSavedSummary() {
    const url = encodeURIComponent(cleanBilibiliUrl());
    const state = await fetchJson(`${apiBase}/api/library/summary?url=${url}`);
    if (state.found && state.summary) {
      renderSummary(state.summary, { activate: false });
      return true;
    }
    renderStudyState({ summary: false });
    return false;
  }

  async function loadSavedMindmap() {
    const url = encodeURIComponent(cleanBilibiliUrl());
    const state = await fetchJson(`${apiBase}/api/library/mindmap?url=${url}`);
    if (state.found && state.mindmap) {
      renderMindmap(state.mindmap, { activate: false });
      return true;
    }
    renderStudyState({ mindmap: false });
    return false;
  }

  async function loadSavedFrames() {
    const url = encodeURIComponent(cleanBilibiliUrl());
    const state = await fetchJson(`${apiBase}/api/library/frames?url=${url}`);
    if (state.found && state.frames) {
      renderFrames(state.frames, { activate: false });
      return true;
    }
    return false;
  }

  async function loadChatSession() {
    const url = encodeURIComponent(cleanBilibiliUrl());
    const state = await fetchJson(`${apiBase}/api/library/chat?url=${url}`);
    if (state.found && state.session) {
      renderChatSession(state.session);
      return Array.isArray(state.session.messages) ? state.session.messages.length : 0;
    }
    renderStudyState({ chatTurns: 0 });
    return 0;
  }

  async function autoRestoreCachedResult() {
    if (!cachedItem || !cachedItem.files || !cachedItem.files.json) {
      return;
    }
    await loadCachedResult({ activate: false });
    const hasSummary = await loadSavedSummary();
    const hasMindmap = await loadSavedMindmap();
    await loadSavedFrames();
    const chatTurns = await loadChatSession();
    renderStudyState({
      transcript: true,
      summary: hasSummary,
      mindmap: hasMindmap,
      chatTurns,
    });
    setStatus("restored");
    setMessage("已恢复历史学习记录，可以继续看字幕、总结、导图或接着对话。");
    setActiveTab(hasSummary ? "summary" : "transcript");
  }

  async function summarizeCurrentVideo() {
    summarizeButton.disabled = true;
    summaryConfirmButton.disabled = true;
    setStatus("summarizing");
    setMessage("正在调用 DeepSeek 生成视频学习笔记...");
    try {
      const form = new FormData();
      form.append("url", cleanBilibiliUrl());
      form.append("api_key", apiKeyEl.value || "");
      form.append("custom_prompt", summaryPromptEl.value || "");
      const state = await fetch(`${apiBase}/api/library/summarize`, {
        method: "POST",
        body: form,
      });
      if (!state.ok) {
        const body = await state.json().catch(() => ({}));
        throw new Error(body.detail || `${state.status} ${state.statusText}`);
      }
      const data = await state.json();
      renderSummary(data.summary);
      if (cachedItem) {
        cachedItem.files = cachedItem.files || {};
        cachedItem.files.summary = cachedItem.files.summary || `/api/library/files/${cachedItem.key}/summary`;
        cachedItem.files.summary_md = cachedItem.files.summary_md || `/api/library/files/${cachedItem.key}/summary_md`;
        renderProjectFiles(cachedItem);
      }
      hideSummaryPromptPanel();
      setStatus("summarized");
      setMessage("总结已生成并保存到本地输出目录。");
    } catch (err) {
      setStatus("failed");
      setMessage(`生成总结失败：${err.message}`);
    } finally {
      summarizeButton.disabled = false;
      summaryConfirmButton.disabled = false;
    }
  }

  async function generateMindmap() {
    mindmapButton.disabled = true;
    setStatus("mindmapping");
    setMessage("正在生成视频思维导图...");
    try {
      const form = new FormData();
      form.append("url", cleanBilibiliUrl());
      form.append("api_key", apiKeyEl.value || "");
      const response = await fetch(`${apiBase}/api/library/mindmap`, {
        method: "POST",
        body: form,
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `${response.status} ${response.statusText}`);
      }
      const data = await response.json();
      renderMindmap(data.mindmap);
      if (cachedItem) {
        cachedItem.files = cachedItem.files || {};
        cachedItem.files.mindmap = cachedItem.files.mindmap || `/api/library/files/${cachedItem.key}/mindmap`;
        renderProjectFiles(cachedItem);
      }
      setStatus("mindmapped");
      setMessage("思维导图已生成并保存到本地输出目录。");
    } catch (err) {
      setStatus("failed");
      setMessage(`生成导图失败：${err.message}`);
    } finally {
      mindmapButton.disabled = false;
    }
  }

  async function generateFrames() {
    framesButton.disabled = true;
    setStatus("framing");
    setMessage("正在从本地视频抽取关键图...");
    try {
      const form = new FormData();
      form.append("url", cleanBilibiliUrl());
      const response = await fetch(`${apiBase}/api/library/frames`, {
        method: "POST",
        body: form,
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `${response.status} ${response.statusText}`);
      }
      const data = await response.json();
      renderFrames(data.frames);
      if (cachedItem) {
        cachedItem.files = cachedItem.files || {};
        cachedItem.files.frames = cachedItem.files.frames || `/api/library/files/${cachedItem.key}/frames`;
        renderProjectFiles(cachedItem);
      }
      setStatus("framed");
      setMessage("关键图已生成并保存到本地输出目录。");
    } catch (err) {
      setStatus("failed");
      setMessage(`生成关键图失败：${err.message}`);
    } finally {
      framesButton.disabled = false;
    }
  }

  async function sendChatQuestion() {
    const question = (chatInputEl.value || "").trim();
    if (!question) {
      setMessage("请输入问题。");
      return;
    }
    chatSendButton.disabled = true;
    setStatus("chatting");
    setMessage("正在基于当前视频回答...");
    try {
      const form = new FormData();
      form.append("url", cleanBilibiliUrl());
      form.append("question", question);
      form.append("api_key", apiKeyEl.value || "");
      const response = await fetch(`${apiBase}/api/library/chat`, {
        method: "POST",
        body: form,
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `${response.status} ${response.statusText}`);
      }
      const data = await response.json();
      renderChatSession(data.session);
      if (cachedItem) {
        cachedItem.files = cachedItem.files || {};
        cachedItem.files.chat = cachedItem.files.chat || `/api/library/files/${cachedItem.key}/chat`;
        renderProjectFiles(cachedItem);
      }
      chatInputEl.value = "";
      setStatus("chat-ready");
      setMessage("回答已保存到本地默认会话。");
      setActiveTab("chat");
    } catch (err) {
      setStatus("failed");
      setMessage(`对话失败：${err.message}`);
    } finally {
      chatSendButton.disabled = false;
    }
  }

  async function clearChatSession() {
    if (!cachedItem) {
      setMessage("没有可清空的本地对话。");
      return;
    }
    clearChatButton.disabled = true;
    setStatus("clearing-chat");
    try {
      const form = new FormData();
      form.append("url", cleanBilibiliUrl());
      const response = await fetch(`${apiBase}/api/library/chat/clear`, {
        method: "POST",
        body: form,
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `${response.status} ${response.statusText}`);
      }
      const data = await response.json();
      renderChatSession(data.session);
      if (cachedItem) {
        cachedItem.files = cachedItem.files || {};
        cachedItem.files.chat = cachedItem.files.chat || `/api/library/files/${cachedItem.key}/chat`;
        renderProjectFiles(cachedItem);
      }
      setStatus("chat-cleared");
      setMessage("当前视频的本地对话已清空。");
      setActiveTab("chat");
    } catch (err) {
      setStatus("failed");
      setMessage(`清空对话失败：${err.message}`);
    } finally {
      clearChatButton.disabled = false;
    }
  }

  async function pollJob(jobId) {
    const state = await fetchJson(`${apiBase}/api/jobs/${jobId}`);
    setStatus(state.status);
    setLog(state.log || "");
    const phaseMessage = messageForJobState(state);
    if (phaseMessage) {
      setMessage(phaseMessage);
    }
    if (Array.isArray(state.segments) && state.segments.length) {
      renderSegments(state.segments);
      setMessage(`正在显示实时字幕，已生成 ${state.segments.length} 段...`);
      if (!liveTranscriptActivated) {
        setActiveTab("transcript");
        liveTranscriptActivated = true;
      }
    }
    if (state.status === "done") {
      analyzeButton.disabled = false;
      const segments = await loadTranscript(jobId);
      renderSegments(segments);
      summarizeButton.hidden = false;
      mindmapButton.hidden = false;
      framesButton.hidden = false;
      setActiveTab("transcript");
      return;
    }
    if (state.status === "failed") {
      analyzeButton.disabled = false;
      setMessage(state.error || "分析失败，请查看日志。");
      return;
    }
    setTimeout(() => pollJob(jobId), 1200);
  }

  async function analyzeCurrentVideo() {
    analyzeButton.disabled = true;
    setStatus("checking");
    setMessage("正在连接本地 FunASR 服务...");
    setLog("");
    listEl.textContent = "";
    liveTranscriptActivated = false;

    if (!findVideo()) {
      setMessage("没有找到 video 元素，请确认当前是 B站视频播放页。");
      analyzeButton.disabled = false;
      setStatus("no-video");
      return;
    }

    try {
      const detected = await detectApiBase();
      setMessage(`已连接本地服务：${detected}`);
      const form = new FormData();
      form.append("url", cleanBilibiliUrl());
      form.append("use_gpu", "true");
      form.append("keep_media", "true");

      const res = await fetch(`${apiBase}/api/jobs`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `${res.status} ${res.statusText}`);
      }
      const body = await res.json();
      setStatus("queued");
      loadCacheButton.hidden = true;
      cachedItem = null;
      setMessage(`Job ${body.job_id} 已创建，等待转写完成。`);
      pollJob(body.job_id);
    } catch (err) {
      analyzeButton.disabled = false;
      setStatus("offline");
      setMessage(`本地服务不可用或创建任务失败：${err.message}`);
    }
  }

  analyzeButton.addEventListener("click", analyzeCurrentVideo);
  loadCacheButton.addEventListener("click", loadCachedResult);
  summarizeButton.addEventListener("click", showSummaryPromptPanel);
  summaryConfirmButton.addEventListener("click", summarizeCurrentVideo);
  summaryCancelButton.addEventListener("click", hideSummaryPromptPanel);
  mindmapButton.addEventListener("click", generateMindmap);
  framesButton.addEventListener("click", generateFrames);
  chatSendButton.addEventListener("click", sendChatQuestion);
  clearChatButton.addEventListener("click", clearChatSession);
  chatInputEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      sendChatQuestion();
    }
  });
  searchEl.addEventListener("input", () => renderSegmentRows(filterSegments(searchEl.value || "")));
  for (const tab of host.querySelectorAll("[data-tab]")) {
    tab.addEventListener("click", () => setActiveTab(tab.dataset.tab));
  }
  collapseButton.addEventListener("click", () => {
    host.style.display = "none";
  });
  document.addEventListener("timeupdate", highlightCurrentSegment, true);
  lookupCachedResult();
})();
