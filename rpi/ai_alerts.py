def analyze_vitals(data):
    """
    Simple AI/alert placeholder.
    Later, this file can be replaced with a trained ML model.
    """
    alerts = []
    status = "Normal"
    risk_level = "low"

    hr = data.get("heart_rate", 0)
    spo2 = data.get("spo2", 0)
    rr = data.get("respiration", 0)
    temp = data.get("temperature", 0)
    motion = abs(data.get("imu_x", 0)) + abs(data.get("imu_y", 0))

    if spo2 and spo2 < 92:
        alerts.append("Low oxygen level detected")
        status = "Warning"
        risk_level = "medium"

    if hr and (hr < 50 or hr > 120):
        alerts.append("Abnormal heart rate detected")
        status = "Warning"
        risk_level = "medium"

    if rr and (rr < 10 or rr > 24):
        alerts.append("Abnormal respiration rate detected")
        status = "Warning"
        risk_level = "medium"

    if temp and temp >= 38.0:
        alerts.append("High temperature detected")
        status = "Warning"
        risk_level = "medium"

    if motion > 0.8:
        alerts.append("Motion artifact possible")
        status = "Check Signal"
        risk_level = "medium"

    if len(alerts) == 0:
        alerts.append("No active alerts")

    return {
        "status": status,
        "risk_level": risk_level,
        "confidence": 0.96 if status == "Normal" else 0.82,
        "alerts": alerts
    }
