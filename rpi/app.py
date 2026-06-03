from flask import Flask, render_template, jsonify
from config import HOST, PORT
from ble_receiver import store, start_data_receiver

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/doctor")
def doctor():
    return render_template("doctor.html")

@app.route("/patient")
def patient():
    return render_template("patient.html")

@app.route("/api/data")
def api_data():
    return jsonify(store.get_dashboard_data())

if __name__ == "__main__":
    start_data_receiver()
    app.run(host=HOST, port=PORT, debug=False)
