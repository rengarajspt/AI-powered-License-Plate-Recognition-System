
import streamlit as st
import pandas as pd
import cv2
import base64
import numpy as np
import easyocr
from ultralytics import YOLO
from PIL import Image
import datetime
import os
import re
from pymongo import MongoClient
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------- MongoDB ----------------------

connection_string = "mongodb+srv://rengarajdhaswin:raju7366@mongodb.oxsxoc5.mongodb.net/?appName=mongodb"

client = MongoClient(connection_string)

db = client["mango"]
database = db["detected_records"]


# ---------------------- Model ----------------------

model = YOLO("best.pt")
reader = easyocr.Reader(['en'])


# ---------------------- OCR Function ----------------------

def read_plate_perfect(plate):

    plate = cv2.resize(plate, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]])

    sharp = cv2.filter2D(gray,-1,kernel)

    thresh = cv2.adaptiveThreshold(
        sharp,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,2
    )

    imgs = [plate, gray, sharp, thresh]

    candidates = []

    for img in imgs:

        result = reader.readtext(
            img,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            paragraph=False
        )

        if result:
            text = "".join([r[1] for r in result])
            conf = np.mean([r[2] for r in result])
            candidates.append((text, conf))

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda x:x[1], reverse=True)
    text = candidates[0][0]

    text = text.replace(" ","").upper()

    corrections = {"O":"0","I":"1","S":"5","B":"8"}
    for k,v in corrections.items():
        text = text.replace(k,v)

    return text


# ---------------------- UI CSS ----------------------

st.markdown("""
<style>

.stApp {
    background-color: #f8f6f2;
}

.main-title {
    color: #8b1c2e;
    font-size: 50px;
    font-weight: 700;
    text-align: center;
}

section[data-testid="stSidebar"] {
    background-color: #f8f6f2;
}

.stButton>button {
    background-color: #8b1c2e;
    color: white;
    font-size: 18px;
    border-radius: 8px;
}

.result-card {
    background-color: white;
    padding: 25px;
    border-radius: 10px;
    border-left: 8px solid #8b1c2e;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# ---------------------- Streamlit Setup ----------------------

st.set_page_config(page_title="Plate Reader", layout="centered")

# Navigation
page = st.sidebar.radio("✳️ 𝐏𝐚𝐠𝐞𝐬",
    ["   𝐑𝐞𝐚𝐝 𝐍𝐮𝐦𝐛𝐞𝐫", "   𝐑𝐞𝐜𝐨𝐫𝐝𝐬"] )


# ---------------------- PAGE 1 ----------------------

if page == "   𝐑𝐞𝐚𝐝 𝐍𝐮𝐦𝐛𝐞𝐫":

    st.markdown('<div class="main-title">𝐀𝐈-𝐩𝐨𝐰𝐞𝐫𝐞𝐝 𝐋𝐢𝐜𝐞𝐧𝐬𝐞 𝐏𝐥𝐚𝐭𝐞 🚗 𝐑𝐞𝐜𝐨𝐠𝐧𝐢𝐭𝐢𝐨𝐧 𝐒𝐲𝐬𝐭𝐞𝐦</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("𝐔𝐩𝐥𝐨𝐚𝐝 𝐢𝐦𝐚𝐠𝐞", type=["jpg","jpeg","png"])

    if uploaded_file is not None:
      try:
        os.makedirs("uploads", exist_ok=True)

        file_path = "uploads/" + uploaded_file.name

        # read image
        image = Image.open(uploaded_file)
        image = np.array(image)

        h, w, _ = image.shape

        # YOLO prediction
        results = model(image)

        plate_found = False

        for r in results:

            boxes = r.boxes.xyxy.cpu().numpy()

            for box in boxes:

                x1, y1, x2, y2 = map(int, box)

                # expand bounding box
                x_pad = int((x2-x1)*0.1)
                y_pad = int((y2-y1)*0.1)

                x1 = max(0, x1-x_pad)
                y1 = max(0, y1-y_pad)
                x2 = min(w, x2+x_pad)
                y2 = min(h, y2+y_pad)

                plate = image[y1:y2, x1:x2]

                plate_text = read_plate_perfect(plate)

                if plate_text:
                    plate_found = True

                    # convert to base64
                    _, buffer = cv2.imencode('.jpg', plate)
                    img_base64 = base64.b64encode(buffer).decode()

                    st.markdown(f"""
                    <div class="result-card">
                        <h1>𝗟𝗶𝗰𝗲𝗻𝘀𝗲 𝗣𝗹𝗮𝘁𝗲 𝗡𝘂𝗺𝗯𝗲𝗿</h1>
                        <img src="data:image/jpg;base64,{img_base64}" 
                        style="width:220px;margin:10px;border-radius:8px;">
                        <h1 style="color:#FFD700; font-weight:900; letter-spacing:3px;">
                        {plate_text}
                        </h1>
                    </div>
                    """, unsafe_allow_html=True)

                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    data = {
                        "plate_number": plate_text,
                        "time": current_time,
                        "file_path": file_path
                    }

                    database.insert_one(data)

        if not plate_found:
            st.warning("⚠️ No license plate detected. Please upload a clearer image.")

      except Exception as e:
        st.error("🚨 Error processing image. Please try another image.")
        st.write("Error details:", e)


# ---------------------- PAGE 2 ----------------------

if page == "   𝐑𝐞𝐜𝐨𝐫𝐝𝐬":

    st.markdown('<div class="main-title">𝐃𝐞𝐭𝐞𝐜𝐭𝐞𝐝 𝐑𝐞𝐜𝐨𝐫𝐝𝐬</div>', unsafe_allow_html=True)

    plateno=[]
    tim=[]
    path=[]

    for i in database.find():

        plateno.append(i.get('plate_number'))
        tim.append(i.get('time'))
        path.append(i.get('file_path'))

    df = pd.DataFrame({
        "plate_number":plateno,
        "time":tim,
        "file_path":path
    })

    st.dataframe(df)

    st.download_button(
        label="𝐃𝐨𝐰𝐧𝐥𝐨𝐚𝐝 𝐜𝐬𝐯",
        data=df.to_csv(index=False),
        file_name="plate_records.csv",
        mime="text/csv"
    )

# ---------------------------------------------------------------------------------------------------------------------------------------


    st.markdown('<div class="main-title">𝐑𝐞𝐜𝐨𝐫𝐝𝐞𝐝 𝐃𝐚𝐭𝐚 𝐀𝐧𝐚𝐥𝐲𝐬𝐢𝐬</div>', unsafe_allow_html=True)

    analysis_option = st.selectbox("𝗦𝗲𝗹𝗲𝗰𝘁 𝗔𝗻𝗮𝗹𝘆𝘀𝗶𝘀",
    [
        "Most Frequent Vehicles (Repeat Visitors)",
        "Vehicle Visits by Hour",
        "Daily Visit Trend",
        "Vehicle Distribution by State",
        "Unique vs Repeat Vehicles"
    ] )


    df["time"] = pd.to_datetime(df["time"])

                                                         # 1️⃣ Most Frequent Vehicles
    if analysis_option == "Most Frequent Vehicles (Repeat Visitors)":

      st.markdown(""" <h1 style='text-align: center;color:#2E86C1;'>Most Frequent Vehicles (Repeat Visitors)</h3> """, unsafe_allow_html=True)

      top_vehicles = df["plate_number"].value_counts().head(10)

      fig, ax = plt.subplots()
      top_vehicles.plot(kind="bar", ax=ax)
      ax.set_xlabel("Plate Number")
      ax.set_ylabel("Visit Count")
      st.pyplot(fig)


                                                          # 2️⃣ Visits by Hour
    elif analysis_option == "Vehicle Visits by Hour":

      st.markdown(""" <h1 style='text-align: center;color:#2E86C1;'>Vehicle Visits by Hour</h3> """, unsafe_allow_html=True)


      df["hour"] = df["time"].dt.hour
      hour_counts = df["hour"].value_counts().sort_index()

      fig, ax = plt.subplots()
      hour_counts.plot(kind="line", marker="o", ax=ax)
      ax.set_xlabel("Hour")
      ax.set_ylabel("Number of Vehicles")
      st.pyplot(fig)


                                                           # 3️⃣ Daily Visit Trend


    elif analysis_option == "Daily Visit Trend":

      st.markdown(""" <h1 style='text-align: center;color:#2E86C1;'>Daily Visit Trend</h3> """, unsafe_allow_html=True)

      df["date"] = df["time"].dt.date
      daily_counts = df["date"].value_counts()

      fig, ax = plt.subplots()
      daily_counts.plot(kind="bar", ax=ax)
      ax.set_xlabel("Date")
      ax.set_ylabel("Number of Vehicles")
      st.pyplot(fig)


                                                            # 4️⃣ Vehicle Distribution by State
    elif analysis_option == "Vehicle Distribution by State":

      st.markdown(""" <h1 style='text-align: center;color:#2E86C1;'>Vehicle Distribution by State</h3> """, unsafe_allow_html=True)

      df["state"] = df["plate_number"].str[:2]
      state_counts = df["state"].value_counts().head(10)

      fig, ax = plt.subplots()
      state_counts.plot(kind="pie", autopct="%1.1f%%", ax=ax)
      ax.set_ylabel("")
      st.pyplot(fig)


                                                            # 5️⃣ Unique vs Repeat Vehicles
    elif analysis_option == "Unique vs Repeat Vehicles":

      st.markdown(""" <h1 style='text-align: center;color:#2E86C1;'>Unique vs Repeat Vehicles</h3> """, unsafe_allow_html=True)

      unique_vehicles = df["plate_number"].nunique()
      repeat_visits = len(df) - unique_vehicles

      fig, ax = plt.subplots()
      ax.pie(
        [unique_vehicles, repeat_visits],
        labels=["Unique Vehicles", "Repeat Visits"],
        autopct="%1.1f%%" )

      st.pyplot(fig)





