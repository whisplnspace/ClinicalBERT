import streamlit as st
import openai
import pandas as pd
import plotly.express as px
import random
from deep_translator import GoogleTranslator
from geopy.geocoders import Nominatim
from langdetect import detect
import requests
import os
import geocoder
import folium
from streamlit_folium import folium_static
from huggingface_hub import login
from transformers import pipeline  # Importing pipeline properly

# Load API Key from Environment
GEMINI_API = os.getenv("GEMINI_API")

# Streamlit Page Config
st.set_page_config(page_title="MediSense AI", layout="wide")
st.title("🩺 MediSense AI - Your Personal Health Assistant")

# Sidebar Menu
st.sidebar.title("Menu")
menu_option = st.sidebar.radio("Choose an option",
                               ["IoT Sensor Predictions", "Symptom Checker", "Health Dashboard", "Chatbot",
                                "BMI Calculator", "Nearby Healthcare Centers"])

# Health Tips
st.sidebar.markdown("### Health Tips 🏥")
health_tips = [
    "💧 Stay hydrated by drinking at least 8 glasses of water daily.",
    "🥦 Eat a balanced diet rich in fruits and vegetables.",
    "🏋️‍♂️ Exercise for at least 30 minutes daily.",
    "😴 Ensure 7-9 hours of quality sleep every night.",
    "🧘 Manage stress through meditation or relaxation techniques.",
]
for tip in health_tips:
    st.sidebar.write(tip)


# Initialize ClinicalBERT Pipeline Once
@st.cache_resource
def load_pipeline():
    return pipeline("fill-mask", model="medicalai/ClinicalBERT")


pipe = load_pipeline()

# IoT Sensor Predictions
if menu_option == "IoT Sensor Predictions":
    st.subheader("📟 IoT Sensor Data & Health Prediction")


    # Function to simulate IoT sensor data
    def get_mock_iot_data():
        return {
            "heart_rate": random.randint(60, 100),  # BPM
            "SpO2": round(random.uniform(95, 100), 1),  # Oxygen level %
            "temperature": round(random.uniform(36.0, 37.5), 1)  # Celsius
        }


    # Function to predict health status
    def predict_health_status(sensor_data):
        sentence = (
            f"The patient has a heart rate of {sensor_data['heart_rate']} BPM, "
            f"SpO2 at {sensor_data['SpO2']}%, and a temperature of {sensor_data['temperature']}°C. "
            f"Possible condition: [MASK]."
        )
        prediction = pipe(sentence)
        return prediction[:3]  # Top 3 predictions


    sensor_data = get_mock_iot_data()
    st.write("📊 **Live IoT Sensor Readings:**", sensor_data)

    if st.button("Predict Health Condition"):
        predictions = predict_health_status(sensor_data)
        st.write("🤖 **AI Predictions:**")
        for pred in predictions:
            st.write(f"- {pred['sequence']} (Confidence: {pred['score']:.2f})")

# Symptom Checker
elif menu_option == "Symptom Checker":
    st.subheader("🤒 Enter Your Symptoms")
    symptoms = st.text_area("Describe your symptoms:")
    if st.button("Check Diagnosis"):
        response = "Possible conditions: Cold, Flu, or Allergies."  # Replace with AI model
        st.success(response)

# Health Insights Dashboard
elif menu_option == "Health Dashboard":
    st.subheader("📊 Your Health Insights")
    data = pd.DataFrame({
        "Date": pd.date_range(start='2024-02-01', periods=10, freq='D'),
        "Heart Rate": [72, 75, 78, 80, 76, 74, 77, 79, 81, 73],
        "Temperature": [36.5, 36.6, 36.7, 36.8, 36.5, 36.4, 36.7, 36.6, 36.9, 36.5]
    })
    fig = px.line(data, x="Date", y=["Heart Rate", "Temperature"], markers=True, title="Health Trends")
    st.plotly_chart(fig)

import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Configure Gemini AI
if GEMINI_API:
    genai.configure(api_key=GEMINI_API)
else:
    st.error("⚠️ API Key is missing. Please set your GEMINI_API key.")

# Chatbot (MediSenseBot)
if menu_option == "Chatbot":
    st.subheader("💬 Chat with MediSenseBot")

    with st.form(key="chatbot_form"):
        user_input = st.text_input("Ask a health-related question:", "What are the common symptoms of flu?")
        submit_button = st.form_submit_button("Get Response")  # Makes Enter key work

    if submit_button:
        if GEMINI_API:
            try:
                model = genai.GenerativeModel("gemini-pro", generation_config={"temperature": 0.7})
                response = model.generate_content(user_input)

                # Ensure the response is not None before accessing `.text`
                reply = response.text if response and hasattr(response, "text") else "⚠️ Error: No response received."

                # Fix Markdown '*' issue by using `st.write()`
                st.write(reply)

            except Exception as e:
                st.error(f"⚠️ API Error: {str(e)}")

        else:
            st.error("⚠️ API key is missing! Please configure your Gemini API key.")

if menu_option == "BMI Calculator":
    st.subheader("⚖️ BMI Calculator")  # This line should be indented properly

    # Add your BMI calculation logic here
    weight = st.number_input("Enter weight (kg):", min_value=1.0, format="%.1f")
    height = st.number_input("Enter height (cm):", min_value=50.0, format="%.1f")

    if st.button("Calculate BMI"):
        if weight and height:
            bmi = weight / ((height / 100) ** 2)
            st.write(f"Your BMI is: **{bmi:.2f}**")

            if bmi < 18.5:
                st.warning("🔹 Underweight")
            elif 18.5 <= bmi < 24.9:
                st.success("✅ Normal weight")
            elif 25 <= bmi < 29.9:
                st.warning("⚠️ Overweight")
            else:
                st.error("🚨 Obese")



# Function to fetch nearby hospitals using Overpass API
def get_nearby_hospitals(lat, lon):
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (
      node["amenity"="hospital"](around:5000,{lat},{lon});
      node["amenity"="pharmacy"](around:5000,{lat},{lon});
    );
    out;
    """
    response = requests.get(overpass_url, params={'data': query})
    data = response.json()
    hospitals = []

    if "elements" in data:
        for element in data["elements"]:
            name = element.get("tags", {}).get("name", "Unknown Hospital/Pharmacy")
            lat = element.get("lat")
            lon = element.get("lon")
            hospitals.append({"name": name, "lat": lat, "lon": lon})
    return hospitals


# Nearby Healthcare Centers
if menu_option == "Nearby Healthcare Centers":
    st.subheader("📍 Find Nearby Hospitals & Pharmacies")

    # Get user's real-time location
    g = geocoder.ip('me')  # Fetch user's location via IP
    if g.latlng:
        user_lat, user_lon = g.latlng
    else:
        # Fallback if IP location fails
        geolocator = Nominatim(user_agent="geoapi")
        location = geolocator.geocode("Kolkata, India")
        if location:
            user_lat, user_lon = location.latitude, location.longitude
        else:
            st.error("Could not fetch location data. Please enter your location manually.")
            user_lat, user_lon = None, None

    if user_lat and user_lon:
        hospitals = get_nearby_hospitals(user_lat, user_lon)

        if hospitals:
            st.success(f"Found {len(hospitals)} hospitals/pharmacies near you!")
            for hospital in hospitals[:5]:
                st.write(f"🏥 **{hospital['name']}** (📍 {hospital['lat']}, {hospital['lon']})")

            # Show locations on an interactive map
            m = folium.Map(location=[user_lat, user_lon], zoom_start=12)
            folium.Marker([user_lat, user_lon], tooltip="Your Location", icon=folium.Icon(color="blue")).add_to(m)

            for hospital in hospitals:
                folium.Marker([hospital["lat"], hospital["lon"]], tooltip=hospital["name"],
                              icon=folium.Icon(color="red", icon="plus-square")).add_to(m)

            folium_static(m)  # Display the map in Streamlit
        else:
            st.warning("No hospitals or pharmacies found nearby.")
