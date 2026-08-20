import streamlit as st
import pandas as pd
import joblib

# --- การตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Marvel Box Office Oracle", page_icon="🎬", layout="wide")

# --- Custom CSS ตกแต่งให้สวยงาม ---
st.markdown("""
<style>
    /* พื้นหลังและฟอนต์หลัก */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e94560;
    }
    /* หัวข้อหลัก */
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #e94560, #f39c12);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .sub-title {
        text-align: center;
        color: #a0a0a0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    /* การ์ดผลลัพธ์ */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        border: 2px solid #e94560;
        box-shadow: 0 8px 32px 0 rgba(233, 69, 96, 0.37);
        margin-top: 20px;
    }
    .result-value {
        font-size: 3rem;
        font-weight: bold;
        color: #f39c12;
        margin: 10px 0;
    }
    /* ปรับแต่งปุ่ม */
    .stButton>button {
        background: linear-gradient(90deg, #e94560, #f39c12);
        color: white;
        border-radius: 30px;
        padding: 10px 30px;
        border: none;
        font-weight: bold;
        font-size: 1.1rem;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(233, 69, 96, 0.5);
    }
    /* ปรับแต่ง Sidebar */
    [data-testid=stSidebar] {
        background: rgba(22, 33, 62, 0.95);
    }
    /* ซ่อนเมนู Streamlit เดิม */
    #MainMenu, header, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- หัวข้อแอป ---
st.markdown('<p class="main-title">🎬 Marvel Box Office Oracle 🦸‍♂️</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">ทำนายรายได้ทั่วโลกของภาพยนตร์ Marvel ด้วยพลังของ Ensemble Random Forest</p>', unsafe_allow_html=True)

# --- โหลดโมเดล ---
@st.cache_resource
def load_model():
    return joblib.load('rf_model.pkl')

model = load_model()

# --- รายการข้อมูลสำหรับ Dropdown (ดึงจาก Dataset) ---
directors = ['Jon Favreau', 'Louis Leterrier', 'Kenneth Branagh', 'Joe Johnston', 'Joss Whedon', 
             'Shane Black', 'Alan Taylor', 'Anthony Russo', 'Joe Russo', 'James Gunn', 'Peyton Reed', 
             'Scott Derrickson', 'Jon Watts', 'Taika Waititi', 'Ryan Coogler', 'Anna Boden', 
             'Ryan Fleck', 'Cate Shortland', 'Destin Daniel Cretton', 'Chloé Zhao', 'Sam Raimi', 
             'Nia DaCosta', 'Shawn Levy']

actors = ['Robert Downey Jr.', 'Edward Norton', 'Chris Hemsworth', 'Chris Evans', 'Chris Pratt', 
          'Paul Rudd', 'Benedict Cumberbatch', 'Tom Holland', 'Chadwick Boseman', 'Brie Larson', 
          'Scarlett Johansson', 'Simu Liu', 'Gemma Chan', 'Letitia Wright', 'Ryan Reynolds']

# --- Sidebar สำหรับรับค่า Input ---
st.sidebar.header("🎛️ สร้างภาพยนตร์ Marvel ของคุณ")
st.sidebar.markdown("---")

col1, col2 = st.sidebar.columns(2)
with col1:
    budget = st.number_input("งบประมาณสร้าง (ล้าน $)", min_value=50, max_value=500, value=200, step=10)
    runtime = st.number_input("ความยาวภาพยนตร์ (นาที)", min_value=90, max_value=200, value=120, step=1)
    imdb = st.slider("คะแนน IMDb", 0.0, 10.0, 7.5, 0.1)
    rt = st.slider("คะแนน Rotten Tomatoes (%)", 0, 100, 80, 1)

with col2:
    year = st.number_input("ปี ที่เข้าฉาย", min_value=2008, max_value=2030, value=2024, step=1)
    phase = st.selectbox("Phase", [1, 2, 3, 4, 5], index=2)
    meta = st.slider("คะแนน Metacritic", 0, 100, 70, 1)
    
st.sidebar.markdown("---")
director = st.sidebar.selectbox("ผู้กำกับ", directors)
actor = st.sidebar.selectbox("นักแสดงนำ", actors)

# --- ปุ่มทำนาย ---
st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🚀 ทำนายรายได้เลย!")

# --- หน้าหลักแสดงผลลัพธ์ ---
if predict_btn:
    # เตรียมข้อมูลให้ตรงกับตอนที่เทรน
    input_data = pd.DataFrame({
        'production_budget_millions': [budget],
        'runtime_minutes': [runtime],
        'release_year': [year],
        'phase_number': [phase],
        'imdb_rating': [imdb],
        'rotten_tomatoes': [rt],
        'metacritic_score': [meta],
        'director': [director],
        'lead_actor': [actor]
    })
    
    # ทำนาย
    prediction = model.predict(input_data)[0]
    
    # แสดงผลแบบสวยๆ
    st.markdown(f"""
    <div class="result-card">
        <h2 style="color: #fff; margin-top: 0;">📊 ผลการทำนายรายได้ทั่วโลก</h2>
        <p class="result-value">${prediction:,.2f} M</p>
        <p style="color: #a0a0a0; font-size: 1.1rem;">
            ภาพยนตร์เรื่องใหม่ของคุณนำแสดงโดย <b style="color:#e94560;">{actor}</b> 
            และกำกับโดย <b style="color:#e94560;">{director}</b>
        </p>
        <hr style="border-color: rgba(255,255,255,0.1);">
        <p style="color: #a0a0a0; font-size: 0.9rem;">
            💡 <i>โมเดลวิเคราะห์จากงบประมาณ {budget}M$, ความยาว {runtime} นาที, 
            และคะแนนรีวิวระดับตำนาน!</i>
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; color: #a0a0a0;">
        <h3>👈 ปรับแต่งพารามิเตอร์ที่เมนูด้านซ้าย แล้วกดปุ่มทำนาย!</h3>
        <p>ดูซิว่าภาพยนตร์ Marvel ในจินตนาการของคุณจะทำเงินได้เท่าไหร่</p>
    </div>
    """, unsafe_allow_html=True)