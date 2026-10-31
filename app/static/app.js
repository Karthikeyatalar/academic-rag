/**
 * Academic RAG Assistant — Frontend Controller
 * Pure Vanilla JavaScript (ES6+)
 */

document.addEventListener("DOMContentLoaded", () => {
  // =========================================================================
  // State
  // =========================================================================
  let sessionId = localStorage.getItem("academic_rag_session_id") || "";
  let stagedFiles = [];
  let indexedFiles = JSON.parse(localStorage.getItem(`rag_files_${sessionId}`) || "[]");
  let totalChunks = parseInt(localStorage.getItem(`rag_chunks_${sessionId}`) || "0", 10);
  let queryHistory = [];

  // =========================================================================
  // DOM Elements
  // =========================================================================
  const sessionIdDisplay = document.getElementById("session-id-display");
  const copySessionBtn = document.getElementById("copy-session-btn");
  const newSessionBtn = document.getElementById("new-session-btn");
  const resetSessionBtn = document.getElementById("reset-session-btn");

  const ollamaPill = document.getElementById("ollama-pill");
  const ollamaDot = document.getElementById("ollama-dot");
  const ollamaText = document.getElementById("ollama-text");

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const stagedContainer = document.getElementById("staged-files-container");
  const stagedCount = document.getElementById("staged-count");
  const stagedList = document.getElementById("staged-files-list");
  const clearStagedBtn = document.getElementById("clear-staged-btn");
  const uploadBtn = document.getElementById("upload-btn");
  const uploadBtnText = document.getElementById("upload-btn-text");
  const uploadSpinner = document.getElementById("upload-spinner");

  const chunkCounterBadge = document.getElementById("chunk-counter-badge");
  const indexedFilesContainer = document.getElementById("indexed-files-container");
  const indexedFilesList = document.getElementById("indexed-files-list");

  const queryInput = document.getElementById("query-input");
  const submitQueryBtn = document.getElementById("submit-query-btn");
  const queryBtnText = document.getElementById("query-btn-text");
  const querySpinner = document.getElementById("query-spinner");
  const quickPromptChips = document.querySelectorAll(".quick-prompt-chip");

  const resultsCard = document.getElementById("results-card");
  const resultStatusBadge = document.getElementById("result-status-badge");
  const stage1Pill = document.getElementById("stage1-pill");
  const stage2Pill = document.getElementById("stage2-pill");
  const answerText = document.getElementById("answer-text");
  const citationContainer = document.getElementById("citation-container");
  const verbatimQuoteText = document.getElementById("verbatim-quote-text");
  const citationSource = document.getElementById("citation-source");
  const citationPage = document.getElementById("citation-page");
  const refusalExplanation = document.getElementById("refusal-explanation");
  const refusalReason = document.getElementById("refusal-reason");

  const historyEmpty = document.getElementById("history-empty");
  const historyList = document.getElementById("history-list");
  const clearHistoryBtn = document.getElementById("clear-history-btn");

  // =========================================================================
  // Initialize
  // =========================================================================
  async function init() {
    checkSystemStatus();

    if (!sessionId) {
      await createNewSession(false);
    } else {
      updateSessionDisplay();
      renderIndexedFiles();
    }

    bindEvents();
  }

  // =========================================================================
  // System Health & Telemetry
  // =========================================================================
  async function checkSystemStatus() {
    try {
      const res = await fetch("/api/status");
      if (res.ok) {
        const data = await res.json();
        if (data.ollama_connected) {
          ollamaDot.className = "status-dot green";
          ollamaText.textContent = "Ollama Active";
          ollamaPill.title = "Ollama is online (llama3:latest)";
        } else {
          ollamaDot.className = "status-dot amber";
          ollamaText.textContent = "Extractive Fallback";
          ollamaPill.title = "Ollama offline. Running in deterministic extractive fallback mode";
        }
      }
    } catch {
      ollamaDot.className = "status-dot amber";
      ollamaText.textContent = "Offline Mode";
    }
  }

  // =========================================================================
  // Session Management
  // =========================================================================
  async function createNewSession(notify = true) {
    try {
      const res = await fetch("/session/create", { method: "POST" });
      if (!res.ok) throw new Error("Failed to create session");
      const data = await res.json();
      sessionId = data.session_id;
      localStorage.setItem("academic_rag_session_id", sessionId);

      indexedFiles = [];
      totalChunks = 0;
      queryHistory = [];
      saveSessionData();

      updateSessionDisplay();
      renderIndexedFiles();
      renderHistory();
      resultsCard.classList.add("hidden");

      if (notify) showToast(`New isolated session created: ${sessionId}`, "success");
    } catch (err) {
      showToast(err.message, "error");
    }
  }

  async function resetSession() {
    if (!sessionId) return;
    if (!confirm(`Are you sure you want to drop session '${sessionId}' and all indexed vectors?`)) return;

    try {
      const res = await fetch(`/session/${sessionId}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Failed to delete collection");

      showToast(`Session ${sessionId} dropped from SQLite database`, "info");
      await createNewSession(true);
    } catch (err) {
      showToast(err.message, "error");
    }
  }

  function updateSessionDisplay() {
    sessionIdDisplay.textContent = sessionId;
  }

  function saveSessionData() {
    localStorage.setItem(`rag_files_${sessionId}`, JSON.stringify(indexedFiles));
    localStorage.setItem(`rag_chunks_${sessionId}`, totalChunks.toString());
  }

  // =========================================================================
  // Drag and Drop & File Staging
  // =========================================================================
  function bindEvents() {
    // Session buttons
    copySessionBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(sessionId);
      showToast("Session ID copied to clipboard!", "info");
    });
    newSessionBtn.addEventListener("click", () => createNewSession(true));
    resetSessionBtn.addEventListener("click", resetSession);

    // Dropzone
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("drag-active");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-active"));
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag-active");
      if (e.dataTransfer.files?.length) {
        handleFileSelection(e.dataTransfer.files);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files?.length) {
        handleFileSelection(e.target.files);
      }
    });

    clearStagedBtn.addEventListener("click", () => {
      stagedFiles = [];
      renderStagedFiles();
      fileInput.value = "";
    });

    uploadBtn.addEventListener("click", uploadFiles);

    // Query Submission
    submitQueryBtn.addEventListener("click", submitQuery);
    queryInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        submitQuery();
      }
    });

    // Quick prompts
    quickPromptChips.forEach((chip) => {
      chip.addEventListener("click", () => {
        queryInput.value = chip.dataset.query;
        queryInput.focus();
      });
    });

    clearHistoryBtn.addEventListener("click", () => {
      queryHistory = [];
      renderHistory();
      showToast("History cleared", "info");
    });
  }

  function handleFileSelection(files) {
    const validExtensions = ["pdf", "docx", "txt"];
    let addedCount = 0;

    for (const file of files) {
      const ext = file.name.split(".").pop().toLowerCase();
      if (!validExtensions.includes(ext)) {
        showToast(`Skipped '${file.name}': Unsupported format (use .pdf, .docx, or .txt)`, "error");
        continue;
      }
      // Avoid duplicate staging
      if (!stagedFiles.some((f) => f.name === file.name && f.size === file.size)) {
        stagedFiles.push(file);
        addedCount++;
      }
    }

    if (addedCount > 0) {
      renderStagedFiles();
    }
  }

  function renderStagedFiles() {
    if (stagedFiles.length === 0) {
      stagedContainer.classList.add("hidden");
      stagedCount.textContent = "0";
      stagedList.innerHTML = "";
      return;
    }

    stagedContainer.classList.remove("hidden");
    stagedCount.textContent = stagedFiles.length;
    stagedList.innerHTML = "";

    stagedFiles.forEach((file, index) => {
      const li = document.createElement("li");
      li.className = "staged-item";
      li.innerHTML = `
        <div class="staged-item-info">
          <span class="badge badge-neutral">${file.name.split(".").pop().toUpperCase()}</span>
          <span class="staged-item-name" title="${file.name}">${file.name}</span>
          <span class="staged-item-size">(${formatBytes(file.size)})</span>
        </div>
        <button class="icon-btn remove-staged-btn" data-index="${index}" title="Remove file">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </button>
      `;
      stagedList.appendChild(li);
    });

    stagedList.querySelectorAll(".remove-staged-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        const idx = parseInt(e.currentTarget.dataset.index, 10);
        stagedFiles.splice(idx, 1);
        renderStagedFiles();
      });
    });
  }

  // =========================================================================
  // Upload & Document Ingestion
  // =========================================================================
  async function uploadFiles() {
    if (stagedFiles.length === 0) return;

    setUploadLoading(true);
    const formData = new FormData();
    formData.append("session_id", sessionId);

    stagedFiles.forEach((file) => {
      formData.append("files", file);
    });

    try {
      const res = await fetch("/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }

      const data = await res.json();
      const chunksIndexed = data.chunks_indexed;
      const filesProcessed = data.files_processed;

      stagedFiles.forEach((f) => {
        if (!indexedFiles.some((item) => item.name === f.name)) {
          indexedFiles.push({ name: f.name, size: f.size });
        }
      });

      totalChunks += chunksIndexed;
      saveSessionData();

      stagedFiles = [];
      fileInput.value = "";
      renderStagedFiles();
      renderIndexedFiles();

      showToast(`Successfully indexed ${chunksIndexed} chunks from ${filesProcessed} document(s)!`, "success");
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      setUploadLoading(false);
    }
  }

  function renderIndexedFiles() {
    chunkCounterBadge.textContent = `${totalChunks} Chunks Indexed`;

    if (indexedFiles.length === 0) {
      indexedFilesContainer.classList.remove("hidden");
      indexedFilesList.classList.add("hidden");
      indexedFilesList.innerHTML = "";
      return;
    }

    indexedFilesContainer.classList.add("hidden");
    indexedFilesList.classList.remove("hidden");
    indexedFilesList.innerHTML = "";

    indexedFiles.forEach((file) => {
      const li = document.createElement("li");
      li.className = "indexed-file-item";
      li.innerHTML = `
        <div class="indexed-file-info">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
          </svg>
          <span class="indexed-file-name">${file.name}</span>
        </div>
        <span class="badge badge-success">Grounded & Searchable</span>
      `;
      indexedFilesList.appendChild(li);
    });
  }

  function setUploadLoading(isLoading) {
    uploadBtn.disabled = isLoading;
    if (isLoading) {
      uploadBtnText.textContent = "Indexing Chunks...";
      uploadSpinner.classList.remove("hidden");
    } else {
      uploadBtnText.textContent = "Ingest & Index Chunks";
      uploadSpinner.classList.add("hidden");
    }
  }

  // =========================================================================
  // Query Execution & Guardrails Verification
  // =========================================================================
  async function submitQuery() {
    const question = queryInput.value.trim();
    if (!question) {
      showToast("Please enter an academic inquiry first.", "info");
      queryInput.focus();
      return;
    }

    if (totalChunks === 0 && indexedFiles.length === 0) {
      showToast("Please upload at least one document before querying.", "info");
      return;
    }

    setQueryLoading(true);

    try {
      const res = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          question: question,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Query failed");
      }

      const result = await res.json();
      displayResult(question, result);

      queryHistory.unshift({ question, result, timestamp: new Date() });
      renderHistory();
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      setQueryLoading(false);
    }
  }

  function displayResult(question, data) {
    resultsCard.classList.remove("hidden");

    const isAnswered = data.status === "answered";

    if (isAnswered) {
      // Status badge
      resultStatusBadge.className = "badge badge-success";
      resultStatusBadge.textContent = "ANSWERED & VERIFIED";

      // Stage pills
      stage1Pill.className = "guardrail-pill pill-pass";
      stage1Pill.innerHTML = '<span class="guardrail-icon">✓</span><span>Stage-1: Relevance Verified (Distance &le; 0.60)</span>';

      stage2Pill.className = "guardrail-pill pill-pass";
      stage2Pill.innerHTML = '<span class="guardrail-icon">✓</span><span>Stage-2: Verbatim Substring Citation Verified (&ge; 10 chars)</span>';

      // Answer & Citation
      answerText.textContent = data.answer;
      verbatimQuoteText.textContent = data.verbatim_quote || "...";
      citationSource.textContent = data.source_document || "document";
      citationPage.textContent = data.cited_page || "1";

      citationContainer.classList.remove("hidden");
      refusalExplanation.classList.add("hidden");
    } else {
      // Refused state
      resultStatusBadge.className = "badge badge-danger";
      resultStatusBadge.textContent = "REFUSED: GUARDRAIL TRIGGERED";

      stage1Pill.className = "guardrail-pill pill-fail";
      stage1Pill.innerHTML = '<span class="guardrail-icon">✕</span><span>Stage-1: Pre-Retrieval Cosine Distance Cutoff (> 0.60)</span>';

      stage2Pill.className = "guardrail-pill pill-fail";
      stage2Pill.innerHTML = '<span class="guardrail-icon">✕</span><span>Stage-2: Citation Not Grounded in Context</span>';

      answerText.textContent = data.answer || "The uploaded documents do not contain information to answer this question.";

      citationContainer.classList.add("hidden");
      refusalExplanation.classList.remove("hidden");
      refusalReason.textContent =
        "The query was rejected by the anti-hallucination guardrail: either the dense cosine distance exceeded the 0.60 threshold (out of scope), or a verbatim exact citation could not be verified in the source text.";
    }

    resultsCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function setQueryLoading(isLoading) {
    submitQueryBtn.disabled = isLoading;
    if (isLoading) {
      queryBtnText.textContent = "Verifying...";
      querySpinner.classList.remove("hidden");
    } else {
      queryBtnText.textContent = "Submit Query";
      querySpinner.classList.add("hidden");
    }
  }

  // =========================================================================
  // History
  // =========================================================================
  function renderHistory() {
    if (queryHistory.length === 0) {
      historyEmpty.classList.remove("hidden");
      historyList.classList.add("hidden");
      historyList.innerHTML = "";
      return;
    }

    historyEmpty.classList.add("hidden");
    historyList.classList.remove("hidden");
    historyList.innerHTML = "";

    queryHistory.forEach((item, index) => {
      const li = document.createElement("li");
      li.className = "history-item";
      const isAns = item.result.status === "answered";
      li.innerHTML = `
        <span class="history-question">${escapeHtml(item.question)}</span>
        <span class="badge ${isAns ? "badge-success" : "badge-danger"}">${isAns ? "ANSWERED" : "REFUSED"}</span>
      `;
      li.addEventListener("click", () => {
        displayResult(item.question, item.result);
      });
      historyList.appendChild(li);
    });
  }

  // =========================================================================
  // Toast Notifications & Helpers
  // =========================================================================
  function showToast(message, type = "info") {
    const container = document.getElementById("toast-container");
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(10px)";
      toast.style.transition = "all 0.2s ease";
      setTimeout(() => toast.remove(), 200);
    }, 3500);
  }

  function formatBytes(bytes) {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Run initialization
  init();
});

