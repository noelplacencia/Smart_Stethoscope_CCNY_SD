async function fetchData() {
    const response = await fetch("/api/data");
    return await response.json();
}

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function getHeartRate(latest) {
    return latest.heart_rate ?? latest.ecg_hr ?? latest.hr_ppg ?? "--";
}

function updateStatus(ai) {
    const statusEl = document.getElementById("patientStatus");
    const alertEl = document.getElementById("patientAlert");
    const confEl = document.getElementById("confidence");

    if (!statusEl || !ai) return;

    statusEl.textContent = ai.status || "Normal";
    statusEl.className = "status";

    if (ai.status === "Normal") {
        statusEl.classList.add("normal");
    } else if (ai.risk_level === "high") {
        statusEl.classList.add("critical");
    } else {
        statusEl.classList.add("warning");
    }

    if (alertEl) alertEl.textContent = (ai.alerts || ["No active alerts"]).join(", ");
    if (confEl) confEl.textContent = Math.round((ai.confidence || 0.96) * 100) + "%";
}

function plotLine(divId, yData) {
    const div = document.getElementById(divId);
    if (!div) return;

    const trace = {
        y: yData,
        mode: "lines",
        line: { width: 2 }
    };

    const layout = {
        margin: { l: 35, r: 10, t: 10, b: 30 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: { showgrid: true, zeroline: false },
        yaxis: { showgrid: true, zeroline: false },
        showlegend: false
    };

    Plotly.react(divId, [trace], layout, {displayModeBar: false, responsive: true});
}

async function updateDashboard() {
    try {
        const data = await fetchData();
        const latest = data.latest || {};

        setText("hr", getHeartRate(latest));
        setText("spo2", latest.spo2 ?? "--");
        setText("resp", latest.respiration ?? "--");
        setText("temp", latest.temperature ?? "--");

        setText("pressure", latest.pressure ?? "--");
        setText("piezo1", latest.piezo1 ?? "--");
        setText("piezo2", latest.piezo2 ?? "--");
        setText("imux", latest.imu_x ?? "--");
        setText("imuy", latest.imu_y ?? "--");
        setText("imuz", latest.imu_z ?? "--");
        setText("updated", latest.timestamp ?? "--");

        setText("bleStatus", latest.timestamp ? "Connected" : "Waiting");

        updateStatus(latest.ai_result);

        plotLine("patientEcg", data.ecg_wave || []);
        plotLine("doctorEcg", data.ecg_wave || []);
        plotLine("doctorHeart", data.heart_wave || []);
        plotLine("spo2Trend", data.spo2_trend || []);
        plotLine("respTrend", data.resp_trend || []);

    } catch (err) {
        console.log("Dashboard update error:", err);
        setText("bleStatus", "Disconnected");
    }
}

function getValue(id) {
    const el = document.getElementById(id);
    return el ? el.value : "";
}

function setValue(id, value) {
    const el = document.getElementById(id);
    if (el) el.value = value || "";
}

async function saveNote() {
    const note = {
        assessment: getValue("assessment"),
        plan: getValue("plan")
    };

    await fetch("/api/notes", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(note)
    });

    setValue("assessment", "");
    setValue("plan", "");
    await loadNotes();
    alert("Doctor notes saved.");
}

async function loadNotes() {
    const list = document.getElementById("notesList");
    if (!list) return;

    try {
        const res = await fetch("/api/notes");
        const notes = await res.json();

        list.innerHTML = "";

        notes.forEach(note => {
            const div = document.createElement("div");
            div.className = "note-item";
            div.innerHTML = `
                <strong>${note.timestamp}</strong><br>
                Assessment: ${note.assessment || ""}<br>
                Plan: ${note.plan || ""}
            `;
            list.appendChild(div);
        });
    } catch (err) {
        console.log("Notes load error:", err);
    }
}

setInterval(updateDashboard, 500);
updateDashboard();
loadNotes();
