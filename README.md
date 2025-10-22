A modern Flask web application for air quality analysis from images, with dynamic UI, real-time predictions, and interactive visualizations.

✨ Features

🎨 Modern & Interactive UI
- Sleek glassmorphism + gradient design
- Animated drag-and-drop image upload zone with glowing hover effect
- Interactive Grad-CAM visualization with before/after comparison slider
- Dynamic AQI gauge with color-coded health zones (green/yellow/red)
- Animated pollution metrics cards showing PM2.5, PM10, and AQI with pulse effects
- Beautiful history timeline with hover animations
- Dark mode toggle with persistent theme preference

🧠 Machine Learning Integration
- Integrates a pre-trained CNN model for pollution detection
- Grad-CAM-based explainability highlights influential image regions
- Real-time predictions rendered instantly
- Supports probability distribution visualization for model outputs

🧩 Technical Stack
- **Frontend**: HTML5, CSS3, JavaScript (ES6), Jinja2 Templates
- **Backend**: Python 3.x, Flask, Flask-CORS
- **Model Serving**: TensorFlow / Keras / PyTorch
- **Visualization**: OpenCV, Matplotlib, Chart.js / D3.js
- **Styling Framework**: TailwindCSS / Bootstrap (optional)
- **Animations**: Anime.js, AOS, GSAP
- **Data Handling**: NumPy, Pandas

📋 Requirements

Python Dependencies

- `flask` - Backend web framework
- `flask-cors` - Cross-origin resource sharing
- `numpy` - Data processing
- `pandas` - Data handling
- `tensorflow / torch` - ML model
- `opencv-python` - Image preprocessing
- `matplotlib` - Grad-CAM visualization
- `chart.js / plotly` - Data visualization
- `gunicorn` - WSGI server

🚀 Quick Start

Installation
1.Clone the repository
  ```bash
git clone https://github.com/Dharsan2024/Dharsan2024.github.io.git
cd Dharsan2024.github.io
 ```

2.Create and activate a virtual environment

- python -m venv venv
- venv\Scripts\activate      # Windows
- source venv/bin/activate   # Linux/Mac

3.Install dependencies
```bash
pip install -r requirements.txt
 ```

4.Running the Application
```bash
python app.py
 ```
Then open your browser at: http://127.0.0.1:5000



