# 🚗 AI-Powered License Plate Recognition System

An **AI-based License Plate Recognition (LPR) system** that detects vehicle number plates from images using **YOLO**, extracts the plate text using **OCR**, and stores the results in **MongoDB**. The application includes an interactive **Streamlit dashboard** for viewing records and performing analytics.

---

# 📌 Project Overview

This project automatically:

1. Uploads a vehicle image
2. Detects the license plate using YOLO
3. Crops the detected plate region
4. Extracts text using OCR
5. Saves plate number, timestamp, and image path to MongoDB
6. Displays records and analytics in a Streamlit dashboard

---

# 🧠 Technologies Used

* **Python**
* **YOLOv8** – Object detection
* **EasyOCR** – Text extraction
* **OpenCV** – Image processing
* **Streamlit** – Web dashboard
* **MongoDB Atlas** – Cloud database
* **Pandas** – Data analysis
* **Matplotlib / Seaborn** – Data visualization

---

# 📂 Project Structure

```
License-Plate-Recognition
│
├── app.py                # Streamlit application
├── best.pt              # Trained YOLO model
├── uploads/              # Uploaded vehicle images
├── plates/               # Cropped license plate images
├── requirements.txt      # Python dependencies
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/license-plate-recognition.git
cd license-plate-recognition
```

---

## 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Run the application

```bash
streamlit run app.py
```

The app will start at:

```
http://localhost:8501
```

---

# 🖼️ Application Features

### 🚗 Vehicle Image Upload

Users can upload vehicle images to detect number plates.

### 🔍 Automatic Plate Detection

YOLO model detects the plate region with bounding boxes.

### 🔤 OCR Plate Extraction

Extracts license plate text using EasyOCR.

### 💾 Data Storage

Stores:

* Plate Number
* Timestamp
* Image Path

in **MongoDB Atlas**.

### 📊 Analytics Dashboard

Includes multiple data insights:

* Most frequent vehicles
* Hourly traffic trend
* Daily visit trend
* Vehicle distribution by state
* Unique vs repeat vehicle analysis

---

# 📊 Example Output

```
Plate Number: TN09AB1234
Time: 2026-03-14 10:45:22
Image Path: uploads/car_12.jpg
```

---

# 📈 Example Dashboard Insights

* 🚗 Top frequent vehicles
* ⏰ Peak traffic hours
* 📅 Daily traffic analysis
* 🌍 State-wise vehicle distribution
* 🔁 Repeat visitor detection

---

# ☁️ MongoDB Database Schema

Example document stored in MongoDB:

```json
{
  "plate_number": "TN09AB1234",
  "time": "2026-03-14 10:45:22",
  "file_path": "uploads/car_12.jpg"
}
```

---

# 🚀 Future Improvements

* Live CCTV vehicle detection
* Real-time traffic monitoring
* Suspicious vehicle alert system
* Vehicle blacklist detection
* GPS-based vehicle tracking

---

# 👨‍💻 Author

**Rengaraj**

Data Science & AI Enthusiast

---

# ⭐ If you like this project

Give the repository a **star ⭐ on GitHub**.
