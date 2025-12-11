const motorStatusEl = document.getElementById("motor-status");
const detectionsBody = document.getElementById("detections-body");
const latestDetectionEl = document.getElementById("latest-detection");
const motorButtons = document.querySelectorAll("[data-motor-action]");

async function fetchJSON(url, options = {}) {
    const response = await fetch(url, {
        headers: { "Content-Type": "application/json" },
        ...options,
    });
    if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.message || `Request failed (${response.status})`);
    }
    return response.json();
}

function setMotorButtonsDisabled(disabled) {
    motorButtons.forEach((button) => {
        button.disabled = disabled;
        if (disabled) {
            button.classList.add("opacity-50", "cursor-not-allowed");
        } else {
            button.classList.remove("opacity-50", "cursor-not-allowed");
        }
    });
}

async function refreshStatus() {
    try {
        const status = await fetchJSON("/api/status");
        const motorControl = status.motor_control || {};
        const enabled = Boolean(motorControl.enabled);
        if (!enabled) {
            if (motorStatusEl) {
                motorStatusEl.textContent = "Motor control disabled. Check Firebase configuration.";
            }
            setMotorButtonsDisabled(true);
        } else {
            if (motorStatusEl && (!motorStatusEl.textContent || motorStatusEl.textContent.includes("disabled"))) {
                motorStatusEl.textContent = "Awaiting command.";
            }
            setMotorButtonsDisabled(false);
        }

        if (latestDetectionEl) {
            const latest = status.latest_detection;
            if (latest) {
                const confidence = latest.confidence != null ? `${(latest.confidence * 100).toFixed(1)}%` : "n/a";
                latestDetectionEl.textContent = `${latest.class_name} (${confidence}) at ${latest.captured_at}`;
            } else {
                latestDetectionEl.textContent = "No detections available yet.";
            }
        }
    } catch (error) {
        console.error(error);
        if (latestDetectionEl) {
            latestDetectionEl.textContent = `Status unavailable: ${error.message}`;
        }
    }
}

async function refreshDetections() {
    try {
        const detections = await fetchJSON("/api/detections");
        if (!detections.length) {
            detectionsBody.innerHTML = '<tr><td colspan="6" class="px-4 py-6 text-center text-slate-400">No detections yet.</td></tr>';
            return;
        }
        detectionsBody.innerHTML = detections
            .map((det) => {
                const confidence = det.confidence != null ? `${(det.confidence * 100).toFixed(1)}%` : "";
                const preview = det.cloudinary_url
                    ? `<img src="${det.cloudinary_url}" alt="${det.class_name ?? ""}" class="h-16 w-16 object-cover rounded"/>`
                    : "";
                const documentId = det.id || det.firebase_document_id || "";
                const cloudinaryId = det.cloudinary_public_id || "";
                const actions =
                    documentId
                        ? `<button class="px-3 py-1 rounded bg-rose-600 text-white text-xs hover:bg-rose-700"
                                   data-action="delete-detection"
                                   data-id="${documentId}"
                                   data-cloudinary="${cloudinaryId}">
                               Delete
                           </button>`
                        : "";
                return `
                    <tr>
                        <td>${det.class_name ?? ""}</td>
                        <td>${confidence}</td>
                        <td>${det.captured_at ?? ""}</td>
                        <td>${preview}</td>
                        <td>${cloudinaryId}</td>
                        <td>${actions}</td>
                    </tr>
                `;
            })
            .join("\n");
    } catch (error) {
        console.error(error);
        alert(`Failed to load detections: ${error.message}`);
    }
}

function motorPayloadFromAction(action) {
    return {
        forward: action === "forward",
        reverse: action === "reverse",
        left: action === "left",
        right: action === "right",
        stop: action === "stop",
    };
}

async function sendMotorCommand(action) {
    if (!action) {
        return;
    }
    setMotorButtonsDisabled(true);
    if (motorStatusEl) {
        motorStatusEl.textContent = `Sending "${action}" command...`;
    }
    try {
        await fetchJSON("/api/motor-control", {
            method: "POST",
            body: JSON.stringify(motorPayloadFromAction(action)),
        });
        if (motorStatusEl) {
            motorStatusEl.textContent = `Last command: ${action}`;
        }
    } catch (error) {
        console.error(error);
        if (motorStatusEl) {
            motorStatusEl.textContent = `Failed to send command: ${error.message}`;
        }
        alert(`Failed to update motor command: ${error.message}`);
    } finally {
        setMotorButtonsDisabled(false);
    }
}

motorButtons.forEach((button) => {
    button.addEventListener("click", (event) => {
        const action = event.currentTarget.getAttribute("data-motor-action");
        sendMotorCommand(action);
    });
});

document.getElementById("refresh-detections").addEventListener("click", () => refreshDetections());

detectionsBody.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-action='delete-detection']");
    if (!button) {
        return;
    }
    const documentId = button.getAttribute("data-id");
    const cloudinaryId = button.getAttribute("data-cloudinary");
    if (!documentId) {
        return;
    }
    const confirmDelete = window.confirm("Delete this detection from Firestore and Cloudinary?");
    if (!confirmDelete) {
        return;
    }
    try {
        await fetchJSON(`/api/detections/${encodeURIComponent(documentId)}`, {
            method: "DELETE",
            body: JSON.stringify({ cloudinary_id: cloudinaryId }),
        });
        await refreshDetections();
    } catch (error) {
        console.error(error);
        alert(`Failed to delete detection: ${error.message}`);
    }
});

refreshStatus();
refreshDetections();
setInterval(refreshStatus, 3000);
setInterval(refreshDetections, 3000);
