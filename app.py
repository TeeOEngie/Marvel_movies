import streamlit as st
import pandas as pd
import joblib

# ============================================================
# โหลดโมเดลที่เทรนไว้ (ต้องอยู่ path เดียวกับ app.py)
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load("svm_student_grade_model.pkl")

model = load_model()

# ============================================================
# ตั้งค่าหน้าเว็บ
# ============================================================
st.set_page_config(page_title="ทำนายเกรดนักเรียน", page_icon="🎓", layout="centered")
st.title("🎓 ระบบทำนายเกรดนักเรียนด้วย SVM")
st.write("กรอกข้อมูลนักเรียนด้านล่างเพื่อทำนายเกรดสุดท้าย (Final Grade)")

st.divider()

# ============================================================
# ฟอร์มรับ input จากผู้ใช้
# ============================================================
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("เพศ (Gender)", ["Male", "Female"])
    study_time_hours = st.slider("ชั่วโมงอ่านหนังสือ/วัน", 0.0, 10.0, 3.5, 0.1)
    attendance_percent = st.slider("เปอร์เซ็นต์การเข้าเรียน (%)", 0.0, 100.0, 85.0, 0.1)
    sleep_hours = st.slider("ชั่วโมงนอน/วัน", 0.0, 12.0, 6.5, 0.1)
    previous_grade = st.slider("คะแนนเกรดเทอมก่อน", 0.0, 100.0, 70.0, 0.1)

with col2:
    parental_education = st.selectbox(
        "ระดับการศึกษาของผู้ปกครอง",
        ["High School", "Bachelors", "Masters", "PhD", "Unknown"]
    )
    internet_access = st.selectbox("มีอินเทอร์เน็ตใช้ที่บ้านหรือไม่", ["Yes", "No"])
    extracurricular_activities = st.selectbox("ทำกิจกรรมนอกหลักสูตรหรือไม่", ["Yes", "No"])
    part_time_job = st.selectbox("ทำงานพาร์ทไทม์หรือไม่", ["Yes", "No"])

st.divider()

# ============================================================
# ปุ่มทำนาย
# ============================================================
if st.button("🔮 ทำนายเกรด", use_container_width=True):
    input_data = pd.DataFrame([{
        "gender": gender,
        "study_time_hours": study_time_hours,
        "attendance_percent": attendance_percent,
        "sleep_hours": sleep_hours,
        "parental_education": parental_education,
        "internet_access": internet_access,
        "extracurricular_activities": extracurricular_activities,
        "part_time_job": part_time_job,
        "previous_grade": previous_grade
    }])

    prediction = model.predict(input_data)[0]

    # ถ้าโมเดลรองรับ predict_proba (SVC(probability=True) ตอนเทรน)
    try:
        proba = model.predict_proba(input_data)[0]
        proba_df = pd.DataFrame({
            "เกรด": model.classes_,
            "ความน่าจะเป็น": proba
        }).sort_values("ความน่าจะเป็น", ascending=False)
    except Exception:
        proba_df = None

    st.success(f"ผลการทำนาย: นักเรียนมีแนวโน้มได้เกรด **{prediction}**")

    if proba_df is not None:
        st.write("ความน่าจะเป็นของแต่ละเกรด:")
        st.bar_chart(proba_df.set_index("เกรด"))

st.caption("โมเดล: Support Vector Machine (SVM) | Dataset: student_performance_dataset.csv")