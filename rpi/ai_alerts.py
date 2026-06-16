def analyze_vitals(data: dict, ml_result: dict = None) -> dict:
    """
    Combine threshold-based vitals alerts with the latest CNN inference result.

    Parameters
    ----------
    data      : latest vitals packet from the ESP32
    ml_result : dict returned by InferenceEngine.run(), or None if not yet available
    """
    alerts = []
    status = "Normal"
    risk_level = "low"

    hr     = data.get("heart_rate", 0)
    spo2   = data.get("spo2", 0)
    rr     = data.get("respiration", 0)
    temp   = data.get("temperature", 0)
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

    # ── CNN inference result ──────────────────────────────────────────────────
    ml_label      = None
    ml_confidence = 0.0
    ml_mode       = None

    if ml_result and ml_result.get("label") not in (None, "unavailable", "error"):
        ml_label      = ml_result["label"]
        ml_confidence = ml_result.get("confidence", 0.0)
        ml_mode       = ml_result.get("mode", "")

        if ml_mode == "heart":
            if ml_label == "present":
                alerts.append(f"Murmur detected ({ml_confidence:.0%} confidence)")
                status = "Warning"
                risk_level = "high"
            else:
                alerts.append(f"No murmur detected ({ml_confidence:.0%} confidence)")

        elif ml_mode == "lung":
            if ml_label == "normal":
                alerts.append(f"Lung sounds normal ({ml_confidence:.0%} confidence)")
            else:
                label_map = {"crackle": "Crackles", "wheeze": "Wheeze",
                             "both": "Crackles and wheeze"}
                display = label_map.get(ml_label, ml_label.capitalize())
                alerts.append(f"{display} detected ({ml_confidence:.0%} confidence)")
                status = "Warning"
                risk_level = "high" if ml_confidence >= 0.75 else "medium"

    if not alerts:
        alerts.append("No active alerts")

    # Confidence shown in the dashboard: use ML confidence when available,
    # otherwise a fixed high value for "vitals normal" or lower for "vitals warning"
    if ml_confidence > 0:
        confidence = ml_confidence
    else:
        confidence = 0.96 if status == "Normal" else 0.82

    return {
        "status":     status,
        "risk_level": risk_level,
        "confidence": round(confidence, 2),
        "alerts":     alerts,
        "ml_label":   ml_label,
        "ml_mode":    ml_mode,
    }
