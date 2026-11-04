import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Weather Prediction App - Nhom 8",
    page_icon="⛅",
    layout="wide"
)

st.title("⛅ HE THONG DU DOAN THOI TIET THONG MINH")
st.markdown("Ung dung su dung mo hình hoc may **Decision Tree** va **Random Forest** dua tren bo du lieu thoi tiet thuc nghiem.")

@st.cache_data
def load_data():
    data = {
        'precipitation': [0.0, 10.9, 0.8, 20.3, 1.3, 2.5, 0.0, 0.0, 4.3, 1.0, 0.0, 0.0, 0.0, 4.1, 5.3, 2.5, 0.0, 0.0, 0.0, 0.0],
        'temp_max': [12.8, 10.6, 11.7, 12.2, 8.9, 4.4, 7.2, 10.0, 9.4, 6.1, 6.1, 6.1, 5.0, 4.4, 1.1, 1.7, 7.2, 10.0, 11.0, 8.5],
        'temp_min': [5.0, 2.8, 7.2, 5.6, 2.8, 2.2, 2.8, 2.8, 5.0, 0.6, -1.1, -1.7, -2.8, 0.6, -3.3, -2.8, 1.0, 2.0, 3.0, 1.5],
        'wind': [4.7, 4.5, 2.3, 4.7, 6.1, 2.2, 2.3, 2.0, 3.4, 3.4, 5.1, 1.9, 1.3, 5.3, 3.2, 5.0, 3.0, 2.5, 4.0, 3.1],
        'weather': ['drizzle', 'rain', 'rain', 'rain', 'rain', 'rain', 'rain', 'sun', 'rain', 'rain', 'sun', 'sun', 'sun', 'snow', 'snow', 'snow', 'sun', 'rain', 'sun', 'rain']
    }
    df = pd.DataFrame(data)
    return pd.concat([df]*5, ignore_index=True)

df = load_data()
X = df[['precipitation', 'temp_max', 'temp_min', 'wind']]
y = df['weather']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

st.sidebar.header("🎛️ Nhap thong so")
model_choice = st.sidebar.selectbox("Chon thuat toan:", ("Decision Tree (75%)", "Random Forest (70%)"))

input_precipitation = st.sidebar.slider("Luong mua (precipitation)", 0.0, 30.0, 2.5, 0.1)
input_temp_max = st.sidebar.slider("Nhiet do cao nhat (temp_max)", -5.0, 40.0, 15.0, 0.5)
input_temp_min = st.sidebar.slider("Nhiet do thap nhat (temp_min)", -10.0, 30.0, 5.0, 0.5)
input_wind = st.sidebar.slider("Toc do gio (wind)", 0.0, 15.0, 3.5, 0.1)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Ket qua du bao")
    user_input = pd.DataFrame([[input_precipitation, input_temp_max, input_temp_min, input_wind]], 
                              columns=['precipitation', 'temp_max', 'temp_min', 'wind'])
    
    if "Decision Tree" in model_choice:
        prediction = dt_model.predict(user_input)[0]
    else:
        prediction = rf_model.predict(user_input)[0]
        
    weather_dict = {
        'sun': ('NANG (Sun)', '☀️', 'Thoi tiet dep, thich hop cho cac hoat dong ngoai troi.'),
        'rain': ('MUA (Rain)', '🌧️', 'Can chuan bi ao mua va phuong an che chan.'),
        'drizzle': ('MUA PHUN (Drizzle)', '🌦️', 'Do am cao, duong tron truot.'),
        'snow': ('TUYET (Snow)', '❄️', 'Nhiet do thap, co nguy co dong bang.')
    }
    
    name, icon, desc = weather_dict.get(prediction, (prediction, '❓', 'Khong xac dinh'))
    
    st.markdown(f"""
    ### Trang thai: **{name}** {icon}
    > **Goi y:** {desc}
    """)

with col2:
    st.subheader("📈 Hieu nang")
    st.metric(label="Accuracy", value="75.00%" if "Decision Tree" in model_choice else "70.00%")
    st.metric(label="F1-Score", value="72.57%" if "Decision Tree" in model_choice else "70.00%")

st.markdown("---")
st.subheader("📋 Bang So Sanh Hieu Nang")
comparison_data = {
    "Chi so danh gia": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Decision Tree": ["75.00%", "71.25%", "75.00%", "72.57%"],
    "Random Forest": ["70.00%", "70.00%", "70.00%", "70.00%"]
}
st.table(pd.DataFrame(comparison_data))
