from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Replace with your OpenWeatherMap API key
API_KEY = "89d614ff35929d456eb2019352f9422f"
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_weather", methods=["POST"])
def get_weather():
    city = request.json["city"]
    params = {"q": city, "appid": API_KEY, "units": "metric"}

    response = requests.get(BASE_URL, params=params)
    weather_data = response.json()

    if response.status_code == 200:
        return jsonify({
            "city": weather_data["name"],
            "temperature": weather_data["main"]["temp"],
            "description": weather_data["weather"][0]["description"],
            "humidity": weather_data["main"]["humidity"]
        })
    else:
        return jsonify({"error": "City not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
