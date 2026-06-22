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

    if (ai.risk_level === "high") {
        statusEl.classList.add("critical");
    } else if (ai.status === "Normal") {
        statusEl.classList.add("normal");
    } else {
        statusEl.classList.add("warning");
    }

    if (alertEl) {
        alertEl.textContent =
            (ai.alerts || ["No active alerts"]).join(", ");
    }

    if (confEl) {
        confEl.textContent =
            Math.round((ai.confidence || 0.96) * 100) + "%";
    }
}

function generateDemoWave(length = 250) {
    const arr = [];

    for (let i = 0; i < length; i++) {

        let v =
            Math.sin(i * 0.12) * 0.8 +
            Math.sin(i * 0.03) * 0.2;

        if (i % 45 === 0) {
            v += 2.5;
        }

        arr.push(v);
    }

    return arr;
}

function plotLine(divId, yData) {

    const div = document.getElementById(divId);

    if (!div) return;

    if (!yData || yData.length === 0) {
        yData = generateDemoWave();
    }

    Plotly.react(
        divId,
        [{
            y: yData,
            mode: "lines",
            line: {
                width: 2
            }
        }],
        {
            margin: {
                l: 25,
                r: 5,
                t: 5,
                b: 20
            },
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            showlegend: false,
            xaxis: {
                showgrid: true,
                zeroline: false
            },
            yaxis: {
                showgrid: true,
                zeroline: false
            }
        },
        {
            displayModeBar: false,
            responsive: true
        }
    );
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

        setText(
            "bleStatus",
            latest.timestamp ? "Connected" : "Waiting"
        );

        updateStatus(latest.ai_result);

        plotLine(
            "patientEcg",
            data.ecg_wave
        );

        plotLine(
            "doctorEcg",
            data.ecg_wave
        );

        plotLine(
            "doctorHeart",
            data.heart_wave
        );

        plotLine(
            "spo2Trend",
            data.spo2_trend
        );

        plotLine(
            "respTrend",
            data.resp_trend
        );

    } catch (err) {

        console.log(err);

        setText(
            "bleStatus",
            "Disconnected"
        );

        plotLine(
            "doctorEcg",
            generateDemoWave()
        );

        plotLine(
            "doctorHeart",
            generateDemoWave()
        );

        plotLine(
            "patientEcg",
            generateDemoWave()
        );
    }
}

async function saveNote() {

    const note = {
        assessment:
            document.getElementById(
                "assessment"
            )?.value || "",

        plan:
            document.getElementById(
                "plan"
            )?.value || ""
    };

    await fetch("/api/notes", {
        method: "POST",
        headers: {
            "Content-Type":
                "application/json"
        },
        body: JSON.stringify(note)
    });

    await loadNotes();
}

async function loadNotes() {

    const list =
        document.getElementById(
            "notesList"
        );

    if (!list) return;

    try {

        const response =
            await fetch("/api/notes");

        const notes =
            await response.json();

        list.innerHTML = "";

        notes.forEach(note => {

            const div =
                document.createElement(
                    "div"
                );

            div.className =
                "note-item";

            div.innerHTML = `
                <strong>${note.timestamp}</strong><br>
                Assessment: ${note.assessment}<br>
                Plan: ${note.plan}
            `;

            list.appendChild(div);

        });

    } catch (err) {
        console.log(err);
    }
}

setInterval(
    updateDashboard,
    500
);

updateDashboard();
loadNotes();
