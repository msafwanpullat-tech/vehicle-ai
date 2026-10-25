document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("dropZone");
    const videoInput = document.getElementById("videoInput");
    const lineRatioSelect = document.getElementById("lineRatioSelect");
    const fileInfo = document.getElementById("fileInfo");
    const fileName = document.getElementById("fileName");
    const fileSize = document.getElementById("fileSize");
    const processBtn = document.getElementById("processBtn");
    const uploadForm = document.getElementById("uploadForm");
    
    const processingState = document.getElementById("processingState");
    const errorMessage = document.getElementById("errorMessage");
    const resultsSection = document.getElementById("resultsSection");

    const processedVideo = document.getElementById("processedVideo");
    const originalVideo = document.getElementById("originalVideo");
    const tabBtns = document.querySelectorAll(".tab-btn");

    const statLineCrossings = document.getElementById("statLineCrossings");
    const statDirectional = document.getElementById("statDirectional");
    const statUniqueVehicles = document.getElementById("statUniqueVehicles");
    const statFps = document.getElementById("statFps");

    const countCar = document.getElementById("countCar");
    const countMotorcycle = document.getElementById("countMotorcycle");
    const countBus = document.getElementById("countBus");
    const countTruck = document.getElementById("countTruck");

    const platesTableBody = document.getElementById("platesTableBody");
    const searchInput = document.getElementById("searchInput");
    const searchBtn = document.getElementById("searchBtn");
    const searchResults = document.getElementById("searchResults");

    let selectedFile = null;

    dropZone.addEventListener("click", () => videoInput.click());

    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("dragover");
        }, false);
    });

    dropZone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    videoInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith("video/")) {
            showError("Please select a valid video file (.mp4, .avi, .mov, .mkv).");
            return;
        }

        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + " MB";
        
        fileInfo.classList.remove("hidden");
        processBtn.disabled = false;
        hideError();
    }

    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!selectedFile) return;

        processBtn.disabled = true;
        processingState.classList.remove("hidden");
        resultsSection.classList.add("hidden");
        hideError();

        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("line_ratio", lineRatioSelect.value);

        try {
            const response = await fetch("/api/process-video", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "Video processing & OCR failed.");
            }

            renderResults(data);

        } catch (err) {
            showError(err.message || "An error occurred while processing the video.");
        } finally {
            processingState.classList.add("hidden");
            processBtn.disabled = false;
        }
    });

    function renderResults(data) {
        processedVideo.src = data.output_video_url;
        originalVideo.src = data.original_video_url;

        processedVideo.load();
        originalVideo.load();

        statLineCrossings.textContent = data.total_line_crossings !== undefined ? data.total_line_crossings : 0;
        statDirectional.textContent = `IN: ${data.in_count || 0} | OUT: ${data.out_count || 0}`;
        statUniqueVehicles.textContent = data.total_unique_vehicles !== undefined ? data.total_unique_vehicles : 0;
        statFps.textContent = (data.processing_fps || 0) + " FPS";

        const counts = data.class_counts || {};
        countCar.textContent = counts.car || 0;
        countMotorcycle.textContent = counts.motorcycle || 0;
        countBus.textContent = counts.bus || 0;
        countTruck.textContent = counts.truck || 0;

        renderPlatesTable(data.recognized_plates || []);

        resultsSection.classList.remove("hidden");
        resultsSection.scrollIntoView({ behavior: "smooth" });
    }

    function renderPlatesTable(plates) {
        if (!plates || plates.length === 0) {
            platesTableBody.innerHTML = `<tr><td colspan="6" class="text-center">No license plates recognized. Try uploading a video with visible plates.</td></tr>`;
            return;
        }

        platesTableBody.innerHTML = plates.map(p => `
            <tr>
                <td><strong>#${p.track_id}</strong></td>
                <td><span class="phase-badge">${p.vehicle_type.toUpperCase()}</span></td>
                <td><span class="plate-badge">${p.registration_number}</span></td>
                <td>${p.confidence}%</td>
                <td>${p.timestamp}</td>
                <td>
                    ${p.crop_url ? `<img src="${p.crop_url}" class="plate-img-thumb" alt="Plate Crop">` : 'N/A'}
                </td>
            </tr>
        `).join("");
    }

    // Search Registration Logic
    searchBtn.addEventListener("click", () => handleSearch());
    searchInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") handleSearch();
    });

    async function handleSearch() {
        const query = searchInput.value.trim();
        if (!query) return;

        searchResults.classList.remove("hidden");
        searchResults.innerHTML = `<div class="text-center">Searching for <strong>${query}</strong>...</div>`;

        try {
            const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.detail || "Search request failed.");
            }

            if (!data.results || data.results.length === 0) {
                searchResults.innerHTML = `
                    <div class="error-alert">
                        <strong>No Match Found:</strong> ${data.message || 'No matching registration number was found.'}
                    </div>
                `;
                return;
            }

            searchResults.innerHTML = data.results.map(r => `
                <div class="match-card">
                    <div class="match-details">
                        <span class="match-badge">MATCH FOUND: ${r.registration_number}</span>
                        <p><strong>Vehicle ID:</strong> #${r.track_id}</p>
                        <p><strong>Vehicle Type:</strong> ${r.vehicle_type.toUpperCase()}</p>
                        <p><strong>OCR Confidence:</strong> ${r.confidence}%</p>
                        <p><strong>Timestamp:</strong> ${r.timestamp} (Frame #${r.frame_number})</p>
                    </div>
                    ${r.frame_url ? `<img src="${r.frame_url}" class="match-frame-img" alt="Matched Frame Snapshot">` : ''}
                </div>
            `).join("");

        } catch (err) {
            searchResults.innerHTML = `<div class="error-alert">${err.message}</div>`;
        }
    }

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            const target = btn.dataset.target;
            if (target === "processedVideo") {
                processedVideo.classList.remove("hidden");
                originalVideo.classList.add("hidden");
                originalVideo.pause();
            } else {
                originalVideo.classList.remove("hidden");
                processedVideo.classList.add("hidden");
                processedVideo.pause();
            }
        });
    });

    function showError(msg) {
        errorMessage.textContent = msg;
        errorMessage.classList.remove("hidden");
    }

    function hideError() {
        errorMessage.textContent = "";
        errorMessage.classList.add("hidden");
    }
});
