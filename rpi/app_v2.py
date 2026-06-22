from flask import Flask, render_template, jsonify, request, send_file
from config import HOST, PORT
from ble_receiver import store, start_data_receiver
from database import init_db, get_profile, save_note, get_notes
from report_generator import generate_report

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/doctor_v2")
def doctor_v2():
    return render_template("doctor_v2.html")


@app.route("/patient_v2")
def patient_v2():
    return render_template("patient_v2.html")


@app.route("/telehealth_v2")
def telehealth_v2():
    return render_template("telehealth_v2.html")


@app.route("/api/data")
def api_data():
    return jsonify(store.get_dashboard_data())


@app.route("/api/profile")
def api_profile():
    return jsonify(get_profile())


@app.route("/api/notes", methods=["GET", "POST"])
def api_notes():
    if request.method == "POST":
        save_note(request.json or {})
        return jsonify({"ok": True})
    return jsonify(get_notes())


@app.route("/api/report_v2")
def api_report_v2():
    data = store.get_dashboard_data()
    latest = data.get("latest", {})
    ai = latest.get("ai_result", {})

    patient_data = {
        "Patient": get_profile().get("name", "Demo Patient"),
        "Heart Rate": latest.get("heart_rate") or latest.get("ecg_hr") or latest.get("hr_ppg") or "--",
        "SpO2": latest.get("spo2", "--"),
        "Respiration": latest.get("respiration", "--"),
        "Temperature": latest.get("temperature", "--"),
        "AI Status": ai.get("status", "Normal"),
        "AI Confidence": str(round(ai.get("confidence", 0.96) * 100)) + "%",
        "Alerts": ", ".join(ai.get("alerts", ["No active alerts"])),
        "Last Updated": latest.get("timestamp", "--"),
    }

    path = generate_report(patient_data, "reports/smart_stethoscope_report.pdf")
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    init_db()
    start_data_receiver()
    app.run(host=HOST, port=PORT, debug=False)
