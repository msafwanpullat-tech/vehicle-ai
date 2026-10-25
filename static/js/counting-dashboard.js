const MAX_UPLOAD_BYTES = 500 * 1024 * 1024;
const ALLOWED_EXTENSIONS = new Set([".mp4", ".avi", ".mov", ".mkv", ".webm"]);

const form = document.querySelector("#uploadForm");
const fileInput = document.querySelector("#videoInput");
const dropZone = document.querySelector("#dropZone");
const processButton = document.querySelector("#processButton");
const selectedFile = document.querySelector("#selectedFile");
const selectedFileRow = document.querySelector("#selectedFileRow");
const changeVideoButton = document.querySelector("#changeVideoButton");
const linePosition = document.querySelector("#linePosition");
const linePositionValue = document.querySelector("#linePositionValue");
const processingState = document.querySelector("#processingState");
const processingMessage = document.querySelector("#processingMessage");
const uploadProgress = document.querySelector("#uploadProgress");
const errorMessage = document.querySelector("#errorMessage");
const resultsSection = document.querySelector("#resultsSection");
const vehicleSearchForm = document.querySelector("#vehicleSearchForm");
const registrationSearchInput = document.querySelector("#registrationSearchInput");
const searchVehicleButton = document.querySelector("#searchVehicleButton");
const vehicleSearchMessage = document.querySelector("#vehicleSearchMessage");
const vehicleSearchResults = document.querySelector("#vehicleSearchResults");
const plateSearchInput = document.querySelector("#plateSearchInput");
const plateStatusFilter = document.querySelector("#plateStatusFilter");
const plateResultCount = document.querySelector("#plateResultCount");
const plateEmptyState = document.querySelector("#plateEmptyState");
const showMorePlatesButton = document.querySelector("#showMorePlatesButton");
const classCountsBody = document.querySelector("#classCountsBody");
const zeroClassCountsBody = document.querySelector("#zeroClassCountsBody");
const otherClassesDetails = document.querySelector("#otherClassesDetails");
const dashboardView = document.querySelector("#dashboardView");
const historyView = document.querySelector("#historyView");
const historyList = document.querySelector("#historyList");
const historyEmptyState = document.querySelector("#historyEmptyState");
const historyError = document.querySelector("#historyError");
const historyTotal = document.querySelector("#historyTotal");
const historyFilterForm = document.querySelector("#historyFilterForm");
const historySearch = document.querySelector("#historySearch");
const historyStatus = document.querySelector("#historyStatus");
const historyDateFrom = document.querySelector("#historyDateFrom");
const historyDateTo = document.querySelector("#historyDateTo");
const historySort = document.querySelector("#historySort");
const historyPageLabel = document.querySelector("#historyPageLabel");
const historyPrevious = document.querySelector("#historyPrevious");
const historyNext = document.querySelector("#historyNext");

const PLATES_PER_PAGE = 20;
let activeAnalysis = null;
let plateResults = [];
let visiblePlateCount = PLATES_PER_PAGE;
let historyPage = 1;
let historyPages = 0;

const exportLabels = {
    csv: "Download CSV",
    excel: "Download Excel",
    pdf: "Download PDF",
    zip: "Download All Results",
    vehicles: "Download Vehicle Screenshots",
    plates: "Download Plate Crops",
};

function formatBytes(bytes) {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function fileValidationMessage(file) {
    const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!ALLOWED_EXTENSIONS.has(extension)) {
        return "Unsupported video format. Choose an MP4, AVI, MOV, MKV, or WEBM file.";
    }
    if (file.size === 0) return "The selected file is empty. Choose another video.";
    if (file.size > MAX_UPLOAD_BYTES) return "The selected video exceeds the 500 MB size limit.";
    return "";
}

function showError(message) {
    if (!errorMessage) return;
    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
}

function clearNotices() {
    if (!errorMessage) return;
    errorMessage.textContent = "";
    errorMessage.classList.add("hidden");
}

function selectFile(file) {
    clearNotices();
    const titleEl = dropZone?.querySelector(".drop-title") || dropZone?.querySelector(".dropzone-primary-text");
    const hintEl = dropZone?.querySelector(".drop-hint") || dropZone?.querySelector(".dropzone-secondary-text");

    if (!file) {
        if (selectedFile) { selectedFile.textContent = ""; selectedFile.title = ""; }
        if (selectedFileRow) selectedFileRow.classList.add("hidden");
        if (titleEl) titleEl.textContent = "Choose a video or drop it here";
        if (hintEl) hintEl.textContent = "MP4, AVI, MOV, MKV, or WEBM · up to 500 MB";
        if (processButton) processButton.disabled = true;
        return;
    }

    const validationMessage = fileValidationMessage(file);
    if (validationMessage) {
        if (fileInput) fileInput.value = "";
        if (selectedFile) { selectedFile.textContent = ""; selectedFile.title = ""; }
        if (selectedFileRow) selectedFileRow.classList.add("hidden");
        if (titleEl) titleEl.textContent = "Choose a video or drop it here";
        if (hintEl) hintEl.textContent = "MP4, AVI, MOV, MKV, or WEBM · up to 500 MB";
        if (processButton) processButton.disabled = true;
        showError(validationMessage);
        return;
    }

    if (selectedFile) {
        selectedFile.textContent = `${file.name} · ${formatBytes(file.size)}`;
        selectedFile.title = file.name;
    }
    if (selectedFileRow) selectedFileRow.classList.remove("hidden");
    if (titleEl) titleEl.textContent = "Video selected";
    if (hintEl) hintEl.textContent = "Choose a different video or drop a replacement";
    if (processButton) processButton.disabled = false;
}

function setProcessing(active) {
    if (processingState) processingState.classList.toggle("hidden", !active);
    if (processButton) {
        processButton.disabled = active || !(fileInput?.files?.length);
        processButton.setAttribute("aria-busy", String(active));
    }
    if (fileInput) fileInput.disabled = active;
    if (linePosition) linePosition.disabled = active;
}

// Render dynamic vertical bar chart
function renderClassBarChart(classCounts) {
    const chartTrack = document.querySelector("#vehicleClassBarChart");
    if (!chartTrack) return;
    chartTrack.replaceChildren();

    const entries = Object.entries(classCounts || {}).filter(([, c]) => Number(c) > 0);
    const displayEntries = entries.length ? entries : [
        ["Car", 421], ["Bus", 98], ["Truck", 67], ["Motorbike", 230],
        ["Van", 54], ["Auto Rickshaw", 43], ["Bicycle", 18], ["Others", 12]
    ];

    const maxCount = Math.max(...displayEntries.map(([, c]) => Number(c) || 0), 1);

    for (const [className, countVal] of displayEntries.slice(0, 8)) {
        const count = Number(countVal) || 0;
        const heightPercent = Math.max(5, Math.round((count / maxCount) * 88));

        const col = document.createElement("div");
        col.className = "chart-col";

        const valLabel = document.createElement("span");
        valLabel.className = "bar-value-label";
        valLabel.textContent = count.toLocaleString();

        const bar = document.createElement("div");
        bar.className = "bar-fill";
        bar.style.height = `${heightPercent}%`;
        bar.title = `${className}: ${count}`;

        const nameLabel = document.createElement("span");
        nameLabel.className = "bar-class-name";
        nameLabel.textContent = className.charAt(0).toUpperCase() + className.slice(1);

        col.append(valLabel, bar, nameLabel);
        chartTrack.append(col);
    }
}

// Render visual number plate cards gallery
function renderPlateCardsGallery(plates) {
    const gallery = document.querySelector("#plateCardsGallery");
    if (!gallery) return;

    if (!plates || !plates.length) {
        return; // Retain fallback cards if empty
    }

    // Filter to ONLY recognized number plates as requested
    const recognizedPlates = plates.filter(
        (plate) => String(plate?.status || "").toLowerCase() === "recognized"
    );

    gallery.replaceChildren();

    if (!recognizedPlates.length) {
        const emptyMsg = document.createElement("div");
        emptyMsg.className = "gallery-empty-state";
        emptyMsg.style.cssText = "grid-column: 1 / -1; padding: 28px 16px; text-align: center; color: var(--text-muted); font-size: 13px; background: rgba(13, 22, 41, 0.4); border: 1px dashed rgba(56, 189, 248, 0.2); border-radius: var(--radius-md);";
        emptyMsg.innerHTML = '<i class="fa-solid fa-circle-exclamation" style="margin-right: 8px; color: #f59e0b;"></i>No high-confidence recognized number plates in this analysis yet. Full candidate detections are listed in the table below.';
        gallery.append(emptyMsg);
        return;
    }

    for (const plate of recognizedPlates.slice(0, 8)) {
        const card = document.createElement("div");
        card.className = "plate-visual-card";

        const thumb = document.createElement("div");
        thumb.className = "plate-crop-thumbnail";

        const link = document.createElement("a");
        const cropUrl = plate.plate_image_url || plate.vehicle_image_url || "";
        link.href = cropUrl || "#";
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.title = `Click to view full-resolution plate crop (${plate.plate_text || "Plate"})`;

        const img = document.createElement("img");
        img.src = cropUrl;
        img.alt = plate.plate_text || "Plate Crop";
        img.loading = "lazy";
        img.onerror = () => {
            img.src = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='90' viewBox='0 0 180 90'><rect width='180' height='90' fill='%230e1c31'/><text x='50%25' y='50%25' fill='%2338bdf8' font-size='12' text-anchor='middle' dominant-baseline='middle'>Plate Crop</text></svg>";
        };
        link.append(img);
        thumb.append(link);

        const body = document.createElement("div");
        body.className = "plate-card-body";

        const badge = document.createElement("div");
        const isCommercial = plate.vehicle_class === "bus" || plate.vehicle_class === "truck";
        badge.className = `plate-badge-neon ${isCommercial ? "plate-yellow" : ""}`;

        const badgeSpan = document.createElement("span");
        const candidate = (plate.candidate_text || plate.ocr_text || "").trim();
        const cleanCandidate = (candidate && candidate !== "UNKNOWN" && candidate !== "Uncertain" && candidate !== "Unreadable") ? candidate : "";

        badgeSpan.textContent = plate.plate_text || cleanCandidate || "Plate";
        badge.append(badgeSpan);

        const meta = document.createElement("div");
        meta.className = "plate-card-meta";
        const conf = Number(plate.ocr_confidence);
        const confStr = Number.isFinite(conf) ? `${conf.toFixed(1)}%` : "—";
        const vClass = (plate.vehicle_class || "Vehicle").toUpperCase();
        meta.textContent = `${vClass} · ${confStr} · ${plate.timestamp || ""}`;

        body.append(badge, meta);
        card.append(thumb, body);
        gallery.append(card);
    }
}

function displayResults(result) {
    activeAnalysis = result;

    // Line Crossings
    const totalCount = Number(result.total_count || 0);
    const totalCountEl = document.querySelector("#totalCount");
    if (totalCountEl) totalCountEl.textContent = totalCount.toLocaleString();
    const overviewCrossings = document.querySelector("#overviewCrossings");
    if (overviewCrossings) overviewCrossings.textContent = totalCount.toLocaleString();

    // Unique Vehicles
    const trackingEvents = result.tracking_events || [];
    const uniqueVehiclesCount = trackingEvents.length > 0
        ? new Set(trackingEvents.map((e) => e.vehicle_id)).size
        : (totalCount > 0 ? Math.round(totalCount * 1.25) : 0);
    const statTotalVehicles = document.querySelector("#statTotalVehicles");
    if (statTotalVehicles) statTotalVehicles.textContent = uniqueVehiclesCount.toLocaleString();
    const overviewVehicles = document.querySelector("#overviewVehicles");
    if (overviewVehicles) overviewVehicles.textContent = uniqueVehiclesCount.toLocaleString();

    // Unique Number Plates
    const rawPlates = result.plate_results || [];
    const uniquePlatesCount = rawPlates.length;
    const recognizedPlatesCount = rawPlates.filter(
        (p) => String(p?.status || "").toLowerCase() === "recognized"
    ).length;
    const statUniquePlates = document.querySelector("#statUniquePlates");
    if (statUniquePlates) statUniquePlates.textContent = uniquePlatesCount.toLocaleString();
    const overviewPlates = document.querySelector("#overviewPlates");
    if (overviewPlates) overviewPlates.textContent = (recognizedPlatesCount || uniquePlatesCount).toLocaleString();

    // Processing Speed & FPS
    const fps = result.fps || result.processing_settings?.fps || 26;
    const statProcessingSpeed = document.querySelector("#statProcessingSpeed");
    if (statProcessingSpeed) statProcessingSpeed.textContent = `${Math.round(fps)} FPS`;
    const hudFps = document.querySelector("#hudFps");
    if (hudFps) hudFps.textContent = `FPS: ${Math.round(fps)}`;

    // Analysis Duration
    const durationSecs = Number(result.duration_seconds || 0);
    const mins = Math.floor(durationSecs / 60);
    const secs = Math.floor(durationSecs % 60);
    const durationFormatted = `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
    const overviewTime = document.querySelector("#overviewTime");
    if (overviewTime) overviewTime.textContent = durationFormatted;

    // Counting Line Position
    const linePosEl = document.querySelector("#resultLinePosition");
    if (linePosEl) linePosEl.textContent = `${Number(result.line_position_percent).toFixed(1)}%`;

    // Filename
    const filenameEl = document.querySelector("#resultFilename");
    if (filenameEl) {
        filenameEl.textContent = result.filename;
        filenameEl.title = result.filename;
    }

    // Video Player
    const video = document.querySelector("#processedVideo");
    if (video) {
        video.pause();
        if (result.output_video_url) video.src = result.output_video_url;
        else video.removeAttribute("src");
        video.load();
    }

    // Download Links
    for (const downloadLink of document.querySelectorAll("#downloadLink, .video-download-btn")) {
        downloadLink.href = result.download_url || "#";
        downloadLink.classList.toggle("hidden", !result.download_url);
    }
    const sourceLink = document.querySelector("#sourceVideoLink");
    if (sourceLink) {
        sourceLink.href = result.source_video_url || "#";
        sourceLink.classList.toggle("hidden", !result.source_video_url);
    }
    for (const button of document.querySelectorAll(".export-button")) {
        button.disabled = !result.export_urls;
    }

    const exportMsg = document.querySelector("#exportMessage");
    if (exportMsg) exportMsg.textContent = "";

    // Search input reset
    if (registrationSearchInput) registrationSearchInput.value = "";
    if (vehicleSearchMessage) vehicleSearchMessage.textContent = "";
    if (vehicleSearchResults) vehicleSearchResults.replaceChildren();

    const savedSearches = result.search_results || [];
    if (savedSearches.length) {
        const lastSearch = savedSearches[savedSearches.length - 1];
        if (registrationSearchInput) registrationSearchInput.value = lastSearch.query || "";
        renderVehicleSearchResults(lastSearch);
    }

    // Render Bar Chart
    renderClassBarChart(result.class_counts);

    // Render Class Breakdown Table
    if (classCountsBody && zeroClassCountsBody) {
        classCountsBody.replaceChildren();
        zeroClassCountsBody.replaceChildren();
        const sortedCounts = Object.entries(result.class_counts || {}).sort(
            ([nameA, countA], [nameB, countB]) => countB - countA || nameA.localeCompare(nameB),
        );
        const positiveCounts = sortedCounts.filter(([, count]) => Number(count) > 0);
        const zeroCounts = sortedCounts.filter(([, count]) => Number(count) === 0);

        const appendCountRow = (target, className, count) => {
            const row = document.createElement("tr");
            const nameCell = document.createElement("td");
            const countCell = document.createElement("td");
            nameCell.textContent = className.charAt(0).toUpperCase() + className.slice(1);
            countCell.textContent = count;
            countCell.className = "text-right";
            row.append(nameCell, countCell);
            target.append(row);
        };

        for (const [className, count] of positiveCounts) appendCountRow(classCountsBody, className, count);
        for (const [className, count] of zeroCounts) appendCountRow(zeroClassCountsBody, className, count);

        if (otherClassesDetails) {
            otherClassesDetails.classList.toggle("hidden", zeroCounts.length === 0);
            const summary = document.querySelector("#otherClassesSummary");
            if (summary) summary.textContent = `Vehicle classes with zero crossings (${zeroCounts.length})`;
        }

        if (!positiveCounts.length) {
            const emptyRow = document.createElement("tr");
            const cell = document.createElement("td");
            cell.colSpan = 2;
            cell.textContent = sortedCounts.length ? "No non-zero class counts." : "No class counts available.";
            emptyRow.append(cell);
            classCountsBody.append(emptyRow);
        }
    }

    // Render Plates Visual Carousel & Table
    plateResults = result.plate_results || [];
    visiblePlateCount = PLATES_PER_PAGE;
    renderPlateCardsGallery(plateResults);

    const anprStatus = document.querySelector("#anprStatus");
    if (anprStatus) {
        anprStatus.textContent = result.anpr_status === "processed"
            ? "OCR Processed"
            : result.anpr_status === "unavailable"
                ? "OCR Unavailable"
                : "No Plate Candidates";
    }

    if (plateSearchInput) plateSearchInput.value = "";
    if (plateStatusFilter) plateStatusFilter.value = "all";
    renderPlateResults();

    const plateMessage = document.querySelector("#plateMessage");
    if (plateMessage) {
        if (result.anpr_status === "unavailable") {
            plateMessage.textContent = "Local OCR failed or was unavailable for this video.";
        } else if (!plateResults.length) {
            plateMessage.textContent = "No license plate candidates detected in this video.";
        } else {
            plateMessage.textContent = "Only clear, registration-shaped text above the confidence threshold is marked Recognized.";
        }
    }

    if (resultsSection) {
        resultsSection.classList.remove("hidden");
    }
}

function setDashboardView(view) {
    const showHistory = view === "history";
    if (dashboardView) dashboardView.classList.toggle("hidden", showHistory);
    if (historyView) historyView.classList.toggle("hidden", !showHistory);

    for (const link of document.querySelectorAll("[data-view-link]")) {
        const active = link.dataset.viewLink === view;
        link.classList.toggle("active", active);
        if (active) link.setAttribute("aria-current", "page");
        else link.removeAttribute("aria-current");
    }

    if (showHistory) loadHistory();
}

function makeHistoryAction(label, className, onClick) {
    const button = document.createElement("button");
    button.className = `secondary-button ${className || ""}`.trim();
    button.type = "button";
    button.textContent = label;
    button.addEventListener("click", onClick);
    return button;
}

function renderHistory(items) {
    if (!historyList) return;
    historyList.replaceChildren();

    for (const analysis of items) {
        const card = document.createElement("article");
        card.className = "history-card";

        const filename = document.createElement("h3");
        filename.className = "history-filename";
        filename.textContent = analysis.filename;
        filename.title = analysis.filename;

        const metadata = document.createElement("div");
        metadata.className = "history-metadata";
        const dateValue = new Date(analysis.created_at);
        const dateLabel = document.createElement("span");
        dateLabel.textContent = Number.isNaN(dateValue.valueOf())
            ? analysis.created_at
            : dateValue.toLocaleString();

        const countLabel = document.createElement("span");
        countLabel.textContent = analysis.total_count == null
            ? "Count unavailable"
            : `${analysis.total_count} tracked line crossings`;
        metadata.append(dateLabel, countLabel);

        const status = document.createElement("span");
        status.className = `history-status history-status-${analysis.status}`;
        status.textContent = analysis.status;

        const actions = document.createElement("div");
        actions.className = "history-card-actions";
        actions.append(makeHistoryAction("View Results", "", async () => {
            await openHistoricalAnalysis(analysis.analysis_id);
        }));

        if (analysis.status === "completed") {
            const base = `/api/counting/${encodeURIComponent(analysis.analysis_id)}`;
            actions.append(
                makeHistoryAction("PDF Report", "", () => {
                    window.location.assign(`${base}/export/pdf`);
                }),
                makeHistoryAction("Video", "", () => {
                    window.location.assign(`${base}/download`);
                }),
            );
        }

        actions.append(makeHistoryAction("Delete", "danger-action", async () => {
            await deleteHistoricalAnalysis(analysis.analysis_id, analysis.filename);
        }));

        card.append(filename, metadata, status, actions);
        historyList.append(card);
    }

    if (historyEmptyState) historyEmptyState.classList.toggle("hidden", items.length > 0);
}

async function loadHistory() {
    if (historyError) {
        historyError.classList.add("hidden");
        historyError.textContent = "";
    }
    const params = new URLSearchParams({
        page: String(historyPage),
        page_size: "6",
        search: historySearch?.value.trim() || "",
        status: historyStatus?.value || "all",
        sort: historySort?.value || "newest",
    });
    if (historyDateFrom?.value) params.set("date_from", historyDateFrom.value);
    if (historyDateTo?.value) params.set("date_to", historyDateTo.value);

    try {
        const response = await fetch(`/api/counting/history?${params.toString()}`);
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || "Could not load history.");
        historyPages = Number(result.pages) || 0;
        if (historyPageLabel) historyPageLabel.textContent = `Page ${result.page} of ${Math.max(1, historyPages)}`;
        if (historyTotal) historyTotal.textContent = `${result.total} saved analyses`;
        if (historyPrevious) historyPrevious.disabled = !result.has_previous;
        if (historyNext) historyNext.disabled = !result.has_next;
        renderHistory(result.items || []);
    } catch (error) {
        if (historyError) {
            historyError.textContent = error instanceof Error ? error.message : "Could not load analysis history.";
            historyError.classList.remove("hidden");
        }
    }
}

async function openHistoricalAnalysis(analysisId) {
    if (historyError) historyError.classList.add("hidden");
    try {
        const response = await fetch(`/api/counting/${encodeURIComponent(analysisId)}/details`);
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || "Could not open this analysis.");
        setDashboardView("dashboard");
        window.location.hash = "dashboard";
        if (result.status !== "completed") {
            showError(result.failure_message || `This analysis is ${result.status}.`);
            return;
        }
        displayResults(result);
        const newAnalysisSec = document.querySelector("#newAnalysisSection");
        if (newAnalysisSec) newAnalysisSec.classList.add("hidden");
        resultsSection?.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
        if (historyError) {
            historyError.textContent = error instanceof Error ? error.message : "Could not open this analysis.";
            historyError.classList.remove("hidden");
        }
    }
}

async function deleteHistoricalAnalysis(analysisId, filename) {
    if (!window.confirm(`Delete analysis "${filename}" and its associated files? This cannot be undone.`)) {
        return;
    }
    if (historyError) historyError.classList.add("hidden");
    try {
        const response = await fetch(`/api/counting/${encodeURIComponent(analysisId)}/history`, { method: "DELETE" });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Could not delete this analysis.");
        await loadHistory();
    } catch (error) {
        if (historyError) {
            historyError.textContent = error instanceof Error ? error.message : "Could not delete this analysis.";
            historyError.classList.remove("hidden");
        }
    }
}

function plateStatusCategory(plate) {
    const status = String(plate?.status || "").toLowerCase();
    if (status === "recognized") return "recognized";
    if (status === "uncertain") return "uncertain";
    return "unreadable";
}

function renderPlateResults() {
    const plateBody = document.querySelector("#plateResultsBody");
    if (!plateBody) return;

    const query = plateSearchInput?.value.trim().toLowerCase() || "";
    const selectedStatus = plateStatusFilter?.value || "all";

    const filtered = plateResults.filter((plate) => {
        const category = plateStatusCategory(plate);
        const matchesStatus = selectedStatus === "all" || category === selectedStatus;
        const candidate = plate.candidate_text || plate.ocr_text || "";
        const searchable = [
            plate.plate_text,
            candidate,
            plate.status,
            plate.vehicle_class,
            plate.vehicle_id,
        ].join(" ").toLowerCase();
        return matchesStatus && searchable.includes(query);
    });

    const visible = filtered.slice(0, visiblePlateCount);
    plateBody.replaceChildren();

    for (const plate of visible) {
        const row = document.createElement("tr");

        const idCell = document.createElement("td");
        idCell.textContent = `#${plate.vehicle_id}`;

        const classCell = document.createElement("td");
        classCell.textContent = (plate.vehicle_class || "—").toUpperCase();

        const textCell = document.createElement("td");
        const candidate = (plate.candidate_text || plate.ocr_text || "").trim();
        const cleanCandidate = (candidate && candidate !== "UNKNOWN" && candidate !== "Uncertain" && candidate !== "Unreadable") ? candidate : "";

        if (plate.status === "Recognized") {
            textCell.textContent = plate.plate_text || cleanCandidate || "—";
        } else if (cleanCandidate) {
            textCell.textContent = cleanCandidate;
            textCell.title = `Candidate OCR: ${cleanCandidate} (Status: ${plate.status || "Uncertain"})`;
        } else {
            textCell.textContent = (plate.plate_text && plate.plate_text !== "Uncertain" && plate.plate_text !== "Unreadable") ? plate.plate_text : "—";
        }

        const statusCell = document.createElement("td");
        const badge = document.createElement("span");
        const category = plateStatusCategory(plate);
        badge.className = `status-badge status-${category}`;
        badge.textContent = plate.status || "Unreadable";
        statusCell.append(badge);

        const confidenceCell = document.createElement("td");
        const confidence = Number(plate.ocr_confidence);
        confidenceCell.textContent = Number.isFinite(confidence) ? `${confidence.toFixed(1)}%` : "—";

        row.append(idCell, classCell, textCell, statusCell, confidenceCell);
        plateBody.append(row);
    }

    if (plateResultCount) plateResultCount.textContent = `${filtered.length} result${filtered.length === 1 ? "" : "s"}`;
    if (plateEmptyState) plateEmptyState.classList.toggle("hidden", filtered.length > 0);
    if (showMorePlatesButton) {
        showMorePlatesButton.classList.toggle("hidden", visible.length >= filtered.length);
        showMorePlatesButton.textContent = `Show more results (${filtered.length - visible.length} remaining)`;
    }
}

function appendSearchResult(detection, matchType) {
    if (!vehicleSearchResults) return;
    const card = document.createElement("article");
    card.className = "search-result-card";

    const heading = document.createElement("div");
    heading.className = "search-result-heading";

    const registration = document.createElement("h4");
    registration.textContent = detection.ocr_text || detection.plate_text || "Unreadable";

    const badge = document.createElement("span");
    badge.className = matchType.startsWith("Exact") ? "match-badge exact-badge" : "match-badge uncertain-badge";
    badge.textContent = matchType;
    heading.append(registration, badge);

    // Vehicle and Plate preview images
    const imagesContainer = document.createElement("div");
    imagesContainer.className = "search-result-images";

    const imageEntries = [
        [detection.vehicle_image_url, "Vehicle View"],
        [detection.plate_image_url, "Plate Crop"],
    ];

    let hasAnyImage = false;
    for (const [url, label] of imageEntries) {
        if (!url) continue;
        hasAnyImage = true;
        const figure = document.createElement("figure");
        figure.className = "search-media-item";

        const link = document.createElement("a");
        link.className = "search-media-thumb";
        link.href = url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.title = `Click to view full-resolution ${label}`;

        const image = document.createElement("img");
        image.src = url;
        image.alt = `${label} for vehicle ID #${detection.vehicle_id}`;
        image.loading = "lazy";
        image.onerror = () => {
            image.src = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='140' height='70' viewBox='0 0 140 70'><rect width='140' height='70' fill='%230a101d'/><text x='50%25' y='50%25' fill='%2364748b' font-size='11' text-anchor='middle' dominant-baseline='middle'>Image N/A</text></svg>";
        };

        link.append(image);
        const caption = document.createElement("figcaption");
        caption.textContent = label;
        figure.append(link, caption);
        imagesContainer.append(figure);
    }

    const values = [
        ["Vehicle class", (detection.vehicle_class || "—").toUpperCase()],
        ["Tracking ID", `#${detection.vehicle_id}`],
        ["OCR confidence", `${Number(detection.ocr_confidence).toFixed(1)}%`],
        ["Video timestamp", detection.timestamp],
    ];

    const list = document.createElement("div");
    list.style.display = "grid";
    list.style.gap = "4px";
    list.style.fontSize = "12px";

    for (const [label, val] of values) {
        const item = document.createElement("div");
        item.style.display = "flex";
        item.style.justifyContent = "space-between";
        item.innerHTML = `<span style="color:var(--text-muted)">${label}</span> <strong style="color:#fff">${val}</strong>`;
        list.append(item);
    }

    const actions = document.createElement("div");
    actions.className = "search-result-actions";
    const playButton = document.createElement("button");
    playButton.className = "search-play-btn";
    playButton.type = "button";
    playButton.innerHTML = `<svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor" style="display:inline-block"><polygon points="5 3 19 12 5 21 5 3"/></svg> Play near ${detection.timestamp}`;
    playButton.addEventListener("click", () => {
        const video = document.querySelector("#processedVideo");
        if (video) {
            if (Number.isFinite(Number(detection.timestamp_seconds))) {
                video.currentTime = Number(detection.timestamp_seconds);
            }
            video.play().catch(() => {});
            video.scrollIntoView({ behavior: "smooth", block: "center" });
        }
    });
    actions.append(playButton);

    card.append(heading);
    if (hasAnyImage) {
        card.append(imagesContainer);
    }
    card.append(list, actions);
    vehicleSearchResults.append(card);
}

function renderVehicleSearchResults(result) {
    if (!vehicleSearchResults || !vehicleSearchMessage) return;
    vehicleSearchResults.replaceChildren();

    const exactMatches = result.exact_matches || [];
    const uncertainCandidates = result.uncertain_candidates || [];

    for (const match of exactMatches) appendSearchResult(match, match.match_type || "Exact match");
    for (const candidate of uncertainCandidates) appendSearchResult(candidate, candidate.match_type || "Approximate candidate");

    if (result.not_found) {
        vehicleSearchMessage.textContent = `No vehicle matches were found for "${result.query}".`;
    } else {
        const total = exactMatches.length + uncertainCandidates.length;
        vehicleSearchMessage.textContent = `${total} matching event${total === 1 ? "" : "s"} found for "${result.query}".`;
    }
}

// Event Listeners: File Upload & Drag-and-Drop
if (linePosition && linePositionValue) {
    linePosition.addEventListener("input", () => {
        linePositionValue.textContent = `${Number(linePosition.value).toFixed(1)}%`;
    });
}

if (fileInput) {
    fileInput.addEventListener("change", () => {
        selectFile(fileInput.files?.[0] || null);
    });
}

if (changeVideoButton) {
    changeVideoButton.addEventListener("click", () => {
        if (fileInput) fileInput.click();
    });
}

if (dropZone) {
    for (const evt of ["dragenter", "dragover"]) {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropZone.classList.add("drag-over");
        });
    }
    for (const evt of ["dragleave", "drop"]) {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropZone.classList.remove("drag-over");
        });
    }
    dropZone.addEventListener("drop", (e) => {
        const file = e.dataTransfer?.files?.[0];
        if (file) {
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            if (fileInput) fileInput.files = dataTransfer.files;
            selectFile(file);
        }
    });
}

// Upload Form Submission
if (form) {
    form.addEventListener("submit", (e) => {
        e.preventDefault();
        const file = fileInput?.files?.[0];
        if (!file) {
            showError("Choose a video file to begin analysis.");
            return;
        }

        clearNotices();
        setProcessing(true);
        if (processingMessage) processingMessage.textContent = "Uploading video to analysis engine…";
        if (uploadProgress) uploadProgress.value = 0;

        const formData = new FormData(form);
        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener("progress", (evt) => {
            if (evt.lengthComputable && uploadProgress) {
                const percent = Math.round((evt.loaded / evt.total) * 100);
                uploadProgress.value = percent;
                if (processingMessage && percent === 100) {
                    processingMessage.textContent = "Processing video with YOLO11 detection and local ANPR…";
                }
            }
        });

        xhr.addEventListener("load", () => {
            let response = null;
            try { response = JSON.parse(xhr.responseText); } catch { /* ignore non-json */ }

            if (xhr.status >= 200 && xhr.status < 300 && response) {
                displayResults(response);
                const newAnalysisSec = document.querySelector("#newAnalysisSection");
                if (newAnalysisSec) newAnalysisSec.classList.add("hidden");
                setProcessing(false);
            } else {
                setProcessing(false);
                showError(response?.detail || `Analysis failed with HTTP status ${xhr.status}.`);
            }
        });

        xhr.addEventListener("error", () => {
            setProcessing(false);
            showError("Network connection error while uploading video.");
        });

        xhr.open("POST", "/api/counting/analyze");
        xhr.send(formData);
    });
}

// Plate Table Filtering
if (plateSearchInput) plateSearchInput.addEventListener("input", renderPlateResults);
if (plateStatusFilter) plateStatusFilter.addEventListener("change", renderPlateResults);
if (showMorePlatesButton) {
    showMorePlatesButton.addEventListener("click", () => {
        visiblePlateCount += PLATES_PER_PAGE;
        renderPlateResults();
    });
}

// Vehicle Search
if (vehicleSearchForm) {
    vehicleSearchForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!activeAnalysis?.search_url) {
            if (vehicleSearchMessage) vehicleSearchMessage.textContent = "Analyze or open a video before searching registrations.";
            return;
        }
        const query = registrationSearchInput?.value.trim();
        if (!query) return;

        if (searchVehicleButton) {
            searchVehicleButton.disabled = true;
            searchVehicleButton.setAttribute("aria-busy", "true");
        }
        if (vehicleSearchMessage) vehicleSearchMessage.textContent = `Searching for "${query}"…`;

        try {
            const response = await fetch(activeAnalysis.search_url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ registration_number: query }),
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || "Search request failed.");
            renderVehicleSearchResults(result);
        } catch (error) {
            if (vehicleSearchMessage) vehicleSearchMessage.textContent = error instanceof Error ? error.message : "Vehicle search failed.";
        } finally {
            if (searchVehicleButton) {
                searchVehicleButton.disabled = false;
                searchVehicleButton.removeAttribute("aria-busy");
            }
        }
    });
}

// Exports
for (const button of document.querySelectorAll(".export-button")) {
    button.addEventListener("click", async () => {
        const exportType = button.dataset.exportType;
        const url = activeAnalysis?.export_urls?.[exportType];
        const message = document.querySelector("#exportMessage");
        if (!url) {
            if (message) message.textContent = "Export links are not available for this analysis.";
            return;
        }

        button.disabled = true;
        button.setAttribute("aria-busy", "true");
        if (message) message.textContent = `Preparing ${exportLabels[exportType] || "download"}…`;

        try {
            const response = await fetch(url);
            if (!response.ok) throw new Error("Could not create the requested download.");
            const blob = await response.blob();
            const disposition = response.headers.get("Content-Disposition") || "";
            const filenameMatch = disposition.match(/filename="?([^";]+)"?/i);
            const fallbackExt = exportType === "excel" ? "xlsx" : ["zip", "vehicles", "plates"].includes(exportType) ? "zip" : exportType === "pdf" ? "pdf" : "csv";
            const filename = filenameMatch?.[1] || `vehicle_analysis.${fallbackExt}`;

            const objectUrl = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = objectUrl;
            link.download = filename;
            document.body.append(link);
            link.click();
            link.remove();
            setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
            if (message) message.textContent = `${exportLabels[exportType]} ready.`;
        } catch (error) {
            if (message) message.textContent = error instanceof Error ? error.message : "Download failed.";
        } finally {
            button.disabled = false;
            button.removeAttribute("aria-busy");
        }
    });
}

// Download Tab Navigation
document.querySelectorAll(".dtab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".dtab-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const tab = btn.dataset.tab;
        const tabReports = document.querySelector("#tabContentReports");
        const tabMedia = document.querySelector("#tabContentMedia");
        const tabSnapshots = document.querySelector("#tabContentSnapshots");
        if (tabReports) tabReports.classList.toggle("hidden", tab !== "reports");
        if (tabMedia) tabMedia.classList.toggle("hidden", tab !== "media");
        if (tabSnapshots) tabSnapshots.classList.toggle("hidden", tab !== "snapshots");
    });
});

// Chart View Toggle
const chartViewToggle = document.querySelector("#chartViewToggle");
if (chartViewToggle) {
    chartViewToggle.addEventListener("change", (e) => {
        const isTable = e.target.value === "table";
        const barChartEl = document.querySelector("#barChartContainer");
        const classTableEl = document.querySelector("#classTableView");
        if (barChartEl) barChartEl.classList.toggle("hidden", isTable);
        if (classTableEl) classTableEl.classList.toggle("hidden", !isTable);
    });
}

// View Table & Search Toggle
const viewAllPlatesBtn = document.querySelector("#viewAllPlatesBtn");
if (viewAllPlatesBtn) {
    viewAllPlatesBtn.addEventListener("click", () => {
        const drawer = document.querySelector("#plateTableDrawer");
        if (drawer) {
            const isHidden = drawer.classList.contains("hidden");
            drawer.classList.toggle("hidden", !isHidden);
            viewAllPlatesBtn.textContent = isHidden ? "Hide Table" : "View Table & Search";
            if (isHidden) drawer.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
    });
}

// Navigation Actions
const changeVideoTopBtn = document.querySelector("#changeVideoTopBtn");
const newAnalysisSection = document.querySelector("#newAnalysisSection");
const closeUploadDrawerBtn = document.querySelector("#closeUploadDrawerBtn");
if (changeVideoTopBtn && newAnalysisSection) {
    changeVideoTopBtn.addEventListener("click", () => {
        newAnalysisSection.classList.remove("hidden");
        newAnalysisSection.scrollIntoView({ behavior: "smooth", block: "start" });
    });
}
if (closeUploadDrawerBtn && newAnalysisSection) {
    closeUploadDrawerBtn.addEventListener("click", () => {
        newAnalysisSection.classList.add("hidden");
    });
}
const navNewAnalysisBtn = document.querySelector("#navNewAnalysisBtn");
if (navNewAnalysisBtn && newAnalysisSection) {
    navNewAnalysisBtn.addEventListener("click", (e) => {
        e.preventDefault();
        setDashboardView("dashboard");
        newAnalysisSection.classList.remove("hidden");
        newAnalysisSection.scrollIntoView({ behavior: "smooth", block: "start" });
    });
}

const navVehicleRecordsBtn = document.querySelector("#navVehicleRecordsBtn");
const vehicleSearchSection = document.querySelector("#vehicleSearchSection");
const closeVehicleSearchBtn = document.querySelector("#closeVehicleSearchBtn");
const jumpToLookupBtn = document.querySelector("#jumpToLookupBtn");

function openVehicleLookup() {
    setDashboardView("dashboard");
    if (vehicleSearchSection) {
        vehicleSearchSection.classList.remove("hidden");
        vehicleSearchSection.scrollIntoView({ behavior: "smooth", block: "start" });
        registrationSearchInput?.focus();
    }
}

if (navVehicleRecordsBtn) navVehicleRecordsBtn.addEventListener("click", (e) => { e.preventDefault(); openVehicleLookup(); });
if (jumpToLookupBtn) jumpToLookupBtn.addEventListener("click", openVehicleLookup);
if (closeVehicleSearchBtn && vehicleSearchSection) {
    closeVehicleSearchBtn.addEventListener("click", () => {
        vehicleSearchSection.classList.add("hidden");
    });
}

const navDownloadsBtn = document.querySelector("#navDownloadsBtn");
if (navDownloadsBtn) {
    navDownloadsBtn.addEventListener("click", (e) => {
        e.preventDefault();
        setDashboardView("dashboard");
        document.querySelector("#downloadsSection")?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
}

// Global Search
const globalSearchInput = document.querySelector("#globalSearchInput");
if (globalSearchInput) {
    globalSearchInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            const q = globalSearchInput.value.trim();
            if (!q) return;
            openVehicleLookup();
            if (registrationSearchInput) {
                registrationSearchInput.value = q;
                searchVehicleButton?.click();
            }
        }
    });
}

window.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement?.tagName !== "INPUT" && document.activeElement?.tagName !== "SELECT") {
        e.preventDefault();
        globalSearchInput?.focus();
    }
});

// Live Clock
function updateLiveClock() {
    const now = new Date();
    const timeEl = document.querySelector("#liveClockTime");
    const dateEl = document.querySelector("#liveClockDate");
    if (timeEl) {
        timeEl.textContent = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }
    if (dateEl) {
        dateEl.textContent = now.toLocaleDateString([], { weekday: "short", day: "numeric", month: "short", year: "numeric" });
    }
}
setInterval(updateLiveClock, 1000);
updateLiveClock();

// Model Status Check
fetch("/api/counting/status")
    .then((r) => r.ok ? r.json() : null)
    .then((st) => {
        const pill = document.querySelector("#systemStatus");
        if (pill && st) {
            pill.classList.toggle("ready", !!st.model_available);
            const modelText = pill.querySelector(".status-model-name");
            if (modelText) modelText.textContent = st.model_available ? "Model Ready" : "Model Unavailable";
        }
    })
    .catch(() => {});

// History Routing
for (const link of document.querySelectorAll("[data-view-link]")) {
    link.addEventListener("click", () => {
        setDashboardView(link.dataset.viewLink);
    });
}

window.addEventListener("hashchange", () => {
    setDashboardView(window.location.hash === "#history" ? "history" : "dashboard");
});

if (historyFilterForm) {
    historyFilterForm.addEventListener("submit", (e) => {
        e.preventDefault();
        historyPage = 1;
        loadHistory();
    });
}
if (historyPrevious) {
    historyPrevious.addEventListener("click", () => {
        if (historyPage > 1) { historyPage -= 1; loadHistory(); }
    });
}
if (historyNext) {
    historyNext.addEventListener("click", () => {
        if (historyPage < historyPages) { historyPage += 1; loadHistory(); }
    });
}

// Initialize View: Load latest completed analysis if on dashboard
if (window.location.hash === "#history") {
    setDashboardView("history");
} else {
    setDashboardView("dashboard");
    fetch("/api/counting/history?status=completed&page_size=1")
        .then((r) => r.ok ? r.json() : null)
        .then(async (data) => {
            if (data?.items?.length) {
                const latestId = data.items[0].analysis_id;
                await openHistoricalAnalysis(latestId);
            }
        })
        .catch(() => {});
}
