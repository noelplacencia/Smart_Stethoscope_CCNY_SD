async function fetchData() {
    const response = await fetch("/api/data");
    return await response.json();
}

function setText(id, value) {
    const element = document.getElementById(id);
    if (element) element.textContent = value;
}

function updateStatus(ai) {
    const statusEl = document.getElementById("patientStatus");
    const alertEl = document.getElementById("patientAlert");
    const confEl = document.getElementById("confidence");

    if (!statusEl || !ai) return;

    statusEl.textContent = ai.status;
    statusEl.className = "status";

    if (ai.status === "Normal") {
        statusEl.classList.add("normal");
    } else {
        statusEl.classList.add("warning");
    }

    if (alertEl) alertEl.textContent = ai.alerts.join(", ");
    if (confEl) confEl.textContent = Math.round(ai.confidence * 100) + "%";
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

    Plotly.react(divId, [trace], layout, {displayModeBar: false});
}

async function updateDashboard() {
    try {
        const data = await fetchData();
        const latest = data.latest || {};

        setText("hr", latest.heart_rate ?? "--");
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

        updateStatus(latest.ai_result);

        plotLine("patientEcg", data.ecg_wave || []);
        plotLine("doctorEcg", data.ecg_wave || []);
        plotLine("doctorHeart", data.heart_wave || []);
        plotLine("spo2Trend", data.spo2_trend || []);
        plotLine("respTrend", data.resp_trend || []);

    } catch (err) {
        console.log("Dashboard update error:", err);
    }
}

setInterval(updateDashboard, 500);
updateDashboard();
