A modern Flask web application for air pollution detection with dynamic UI, real-time predictions, and interactive visualizations.

✨ Features
🎨 Modern & Interactive UI

Sleek glassmorphism + gradient design

Animated drag-and-drop image upload zone with glowing hover effect

Real-time prediction display with smooth transitions and animated progress bars

Interactive Grad-CAM visualization with before/after comparison slider

Dynamic AQI gauge with color-coded health zones (green/yellow/red)

Animated pollution metrics cards showing PM2.5, PM10, and AQI with pulse effects

Beautiful history timeline with hover animations

Animated probability indicators showing prediction confidence

Mobile-first responsive design

Dark mode toggle with persistent theme preference

🧠 Machine Learning Integration

Integrates a pre-trained CNN model for pollution detection

Grad-CAM-based explainability highlights influential image regions

Real-time predictions rendered instantly

Supports probability distribution visualization for model outputs

🧩 Technical Stack

Frontend: HTML5, CSS3, JavaScript (ES6), Jinja2 Templates

Backend: Python 3.x, Flask, Flask-CORS

Model Serving: TensorFlow / Keras / PyTorch

Visualization: OpenCV, Matplotlib, Chart.js / D3.js

Styling Framework: TailwindCSS / Bootstrap (optional)

Animations: Anime.js, AOS, GSAP

Data Handling: NumPy, Pandas

Deployment: Gunicorn + Nginx / Render / AWS / Heroku

🏗️ Architecture Overview

User Interface

Drag & drop upload zone

AQI gauge & Grad-CAM visualization

Animated metric cards & history timeline

Flask Backend

Handles routes & uploads

Calls ML model for inference

Returns JSON predictions

ML Model & Processing

CNN-based air pollution detection

Grad-CAM visualization

AQI computation logic

Visualization & Analytics

Graphs & gauges

History timeline

Probability indicators

📋 Requirements
Python Dependencies

flask - Backend web framework

flask-cors - Cross-origin resource sharing

numpy - Data processing

pandas - Data handling

tensorflow / torch - ML model

opencv-python - Image preprocessing

matplotlib - Grad-CAM visualization

chart.js / plotly - Data visualization

gunicorn - WSGI server

🚀 Quick Start
Installation

Clone the repository

git clone https://github.com/Dharsan2024/Dharsan2024.github.io.git
cd Dharsan2024.github.io


Create and activate a virtual environment

python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac


Install dependencies

pip install -r requirements.txt

Running the Application
python app.py


Then open your browser at: http://127.0.0.1:5000
