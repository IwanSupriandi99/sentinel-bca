import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sqlalchemy import create_engine
import plotly.express as px
from preprocessing import bersihkan_teks

# 1. Konfigurasi Halaman Web
st.set_page_config(
    page_title="SENTINEL - Sentiment Insight & Evaluator", 
    layout="wide", 
    page_icon="🦅"
)

# --- KODE LOAD MODEL ---
@st.cache_resource 
def load_models():
    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('svm_sentimen.pkl', 'rb') as f:
        svm_sentimen = pickle.load(f)
    with open('svm_aspek.pkl', 'rb') as f:
        svm_aspek = pickle.load(f)
    return tfidf, svm_sentimen, svm_aspek

# 2. Inisialisasi Session State
if 'sudah_login' not in st.session_state:
    st.session_state['sudah_login'] = False
if 'data_mentah' not in st.session_state:
    st.session_state['data_mentah'] = None
if 'data_bersih' not in st.session_state: 
    st.session_state['data_bersih'] = None
if 'halaman_saat_ini' not in st.session_state: 
    st.session_state['halaman_saat_ini'] = 1

# Fungsi untuk Logout
def logout():
    st.session_state['sudah_login'] = False
    st.rerun()

# ==========================================
# FUNGSI UI: TABEL 
# ==========================================
def tampilkan_tabel_cantik(df):
    css = """
    <style>
    .tabel-container {
        max-height: 400px;
        overflow-y: auto;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-top: 10px;
    }
    .tabel-sentinel {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 14px;
    }
    .tabel-sentinel th {
        background-color: #7dd3fc !important;
        color: #0f172a !important;
        padding: 14px 16px;
        position: sticky;
        top: 0;
        z-index: 2;
        font-weight: 800;
        text-align: center !important; /* MEMBUAT JUDUL KOLOM KE TENGAH */
        border-bottom: 2px solid #38bdf8;
    }
    .tabel-sentinel td {
        padding: 12px 16px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
        text-align: left; /* Teks ulasan tetap rata kiri agar enak dibaca */
        vertical-align: middle; /* Teks seimbang di tengah atas-bawah */
    }
    .tabel-sentinel tbody tr {
        transition: all 0.2s ease-in-out;
    }
    .tabel-sentinel tbody tr:hover {
        background-color: #fef08a !important; /* WARNA HOVER KUNING PUCAT (Highlighter) */
        color: #0f172a !important; /* Teks otomatis jadi gelap saat disorot agar mudah dibaca */
    }
    </style>
    """
    
    # 1. Ubah dataframe Pandas ke bentuk tabel HTML
    html_table = df.to_html(classes='tabel-sentinel', index=False, escape=False)
    
    # 2. Gabungkan CSS dan Tabel
    full_html = f"{css}<div class='tabel-container'>{html_table}</div>"
    
    # 3. JURUS ANTI-BOCOR: Hapus semua "Enter/Baris Baru"
    full_html_aman = full_html.replace('\n', '')
    
    # 4. Tampilkan secara paksa sebagai HTML murni
    st.markdown(full_html_aman, unsafe_allow_html=True)

# =====================================================================
# --- HALAMAN PINTU MASUK (LOGIN) ---
# =====================================================================
if not st.session_state.get('sudah_login', False):
    st.markdown("""
        <style>
        /* Reset & Hide Streamlit Elements */
        [data-testid="stHeader"] { display: none; }
        .block-container { max-width: 100% !important; padding-top: 2rem !important; }
        
        /* ---------------- SISI KIRI  ---------------- */
        .login-container { padding: 5% 10%; display: flex; flex-direction: column; justify-content: center; height: 100%; }
        .login-title { font-size: 38px; font-weight: 900; margin-bottom: 5px; color: var(--text-color); letter-spacing: -1px; }
        .login-sub { font-size: 15px; color: gray; margin-bottom: 35px; line-height: 1.5; font-weight: 500; }
        
        /* Animasi Tangan Melambai */
        .wave-hand { display: inline-block; animation: wave 2.5s infinite; transform-origin: 70% 70%; }
        @keyframes wave { 0% { transform: rotate(0deg); } 10% { transform: rotate(14deg); } 20% { transform: rotate(-8deg); } 30% { transform: rotate(14deg); } 40% { transform: rotate(-4deg); } 50% { transform: rotate(10deg); } 60% { transform: rotate(0deg); } 100% { transform: rotate(0deg); } }
        
        /* Efek Glow BCA saat form diklik (Fokus) */
        .stTextInput > div > div > input { 
            border-radius: 10px !important; border: 1.5px solid rgba(128,128,128,0.2) !important; 
            padding: 14px 18px !important; font-size: 15px !important; transition: all 0.3s ease;
        }
        .stTextInput > div > div > input:focus {
            border: 1.5px solid #00A2E9 !important;
            box-shadow: 0 0 15px rgba(0, 162, 233, 0.25) !important;
        }
        .stTextInput > label { font-size: 14px !important; font-weight: 700 !important; color: gray !important; margin-bottom: 8px !important;}
        
        /* Tombol Biru BCA Streamlit + Animasi Shine */
        div.stButton > button { 
            background: linear-gradient(135deg, #005AAA 0%, #00A2E9 100%) !important; 
            color: white !important; border: none !important; font-weight: 800 !important; 
            border-radius: 10px !important; padding: 12px 24px !important; font-size: 16px !important;
            margin-top: 10px; transition: transform 0.3s ease, box-shadow 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 90, 170, 0.3);
            position: relative; overflow: hidden; /* Syarat efek kilap */
        }
        div.stButton > button:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0, 90, 170, 0.5); }
        
        /* Efek Kilap (Shine) pada tombol */
        div.stButton > button::after {
            content: ''; position: absolute; top: 0; left: -150%; width: 50%; height: 100%;
            background: linear-gradient(to right, rgba(255,255,255,0) 0%, rgba(255,255,255,0.4) 50%, rgba(255,255,255,0) 100%);
            transform: skewX(-25deg); animation: buttonShine 5s infinite; z-index: 1;
        }
        @keyframes buttonShine { 0% { left: -150%; } 20% { left: 250%; } 100% { left: 250%; } }

        /* Garis Pemisah (Divider) Ala SaaS Modern */
        .divider-s { 
            display: flex; align-items: center; text-align: center; margin: 25px 0; 
            color: gray; font-size: 12px; font-weight: 700; letter-spacing: 1px; opacity: 0.6;
        }
        .divider-s::before, .divider-s::after { content: ''; flex: 1; border-bottom: 1px solid rgba(128,128,128,0.3); }
        .divider-s:not(:empty)::before { margin-right: 15px; }
        .divider-s:not(:empty)::after { margin-left: 15px; }

        /* SISI KANAN (BRANDING)  */
        .branding-card {
            background: linear-gradient(135deg, #0B1120 0%, #152A4A 100%);
            border-radius: 24px; padding: 50px; display: flex; flex-direction: column; justify-content: space-between;
            box-shadow: 0 20px 50px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.05); min-height: 80vh;
        }
        .feature-box { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 16px 20px; border-radius: 14px; margin-bottom: 15px; font-weight: 600; display: flex; align-items: center; color: white !important; transition: transform 0.3s ease, background 0.3s ease; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
        .feature-box:hover { transform: translateX(10px); background: rgba(255,255,255,0.08); }
        
        /* Animasi Muncul Bertahap saat halaman di-load */
        .fade-in-up { animation: fadeInUp 0.8s cubic-bezier(0.165, 0.84, 0.44, 1) forwards; opacity: 0; }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
        </style>
    """, unsafe_allow_html=True)

    col_kiri, col_kanan = st.columns([1.1, 0.9], gap="large")
    
    # --- SISI KIRI  ---
    with col_kiri:
        st.markdown("<div class='login-container fade-in-up' style='animation-delay: 0.1s;'>", unsafe_allow_html=True)
        
        st.markdown("<div class='login-title'>Selamat Datang <span class='wave-hand'>👋</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='login-sub'>Masukkan kredensial admin untuk mengakses kontrol panel eksekutif <b>SENTINEL</b>.</div>", unsafe_allow_html=True)
        
        # Social Login dengan Hover Modern
        c_soc1, c_soc2 = st.columns(2)
        with c_soc1: 
            if st.button("🌐 Masuk dengan Google", use_container_width=True):
                st.toast("Fitur SSO Google dalam tahap pengembangan.", icon="ℹ️")
        with c_soc2: 
            if st.button("🍏 Masuk dengan Apple", use_container_width=True):
                st.toast("Fitur SSO Apple dalam tahap pengembangan.", icon="ℹ️")
        # Divider Super Elegan
        st.markdown("<div class='divider-s'>ATAU GUNAKAN AKUN</div>", unsafe_allow_html=True)

        # Input Form
        username = st.text_input("Username", placeholder="Masukkan username")
        password = st.text_input("Password", type="password", placeholder="Masukkan password")
        
        # Tombol Submit Utama
        if st.button("Masuk Ke Dashboard", use_container_width=True):
            if username == "admin" and password == "admin123":
                st.session_state['sudah_login'] = True
                st.rerun()
            else:
                st.error("❌ Username atau Password salah! Coba lagi.")
                
        st.markdown("<p style='text-align: center; margin-top:35px; font-size:12px; color:gray;'>2026 SENTINEL by Iwan Supriandi. All rights Reserved.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_kanan:
        # RAW HTML Rata Kiri 100% (Dilarang ada spasi di awal baris agar tidak jadi Code Block)
        st.markdown("""
<div class="branding-card">
<div>
<h2 style="font-weight: 900; font-size: 28px; margin:0; color:white;">🦅 SENTINEL</h2>
<p style="opacity: 0.7; font-size: 13px; font-weight:600; color:white;">Sentiment Insight & Evaluator</p>
</div>

<div style="flex-grow: 1; margin-top: 30px;">
<p style="color:#64FFDA !important; font-weight:800; font-size:12px; letter-spacing:2px; margin-bottom: 10px;">📡 ANALISIS SENTIMEN ADVANCED</p>
<h1 style="font-size: 38px; line-height: 1.2; font-weight: 900; margin-bottom: 20px; color:white;">Revolusi Pemantauan Ulasan m-Banking BCA</h1>
<p style="opacity: 0.7; margin-bottom: 35px; font-size: 15px; color:white; line-height: 1.6;">Dapatkan wawasan mendalam dan objektif dari ribuan suara pengguna. SENTINEL membantu Tim BCA mengambil keputusan strategis lebih cepat dan presisi.</p>

<div class="feature-box"><span style="margin-right:15px; font-size:22px;">⚡</span> Deteksi Masalah Real-Time</div>
<div class="feature-box"><span style="margin-right:15px; font-size:22px;">🧠</span> Mesin Support Vector Machine Lanjutan</div>
<div class="feature-box"><span style="margin-right:15px; font-size:22px;">🚨</span> Tindakan Strategis Berbasis Data</div>
</div>

<div style="margin-top: 30px; border-top: 1px solid rgba(255,255,255,0.1); padding-top:25px; display:flex; justify-content:space-between; align-items:center;">
<span style="font-size: 12px; opacity: 0.6; color:white; font-weight: 600;">Dukungan Ekosistem:</span>
<span style="background:black; color:#112240 !important; padding:6px 14px; border-radius:8px; font-weight:900; font-size:12px; box-shadow: 0 4px 10px rgba(0,0,0,0.2);">🛡️ UMSU Fikti-SI</span>
</div>
</div>
        """, unsafe_allow_html=True)

# =====================================================================
# --- HALAMAN UTAMA (JIKA SUDAH LOGIN) ---
# =====================================================================
else:
    st.markdown("""
        <style>
        html, body, [class*="st-"] { text-rendering: optimizeLegibility; -webkit-font-smoothing: antialiased; }
        [data-testid="stHeader"] { background-color: rgba(0,0,0,0) !important; }
        .block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; max-width: 96% !important;}

        [data-testid="stSidebar"] { background: linear-gradient(180deg, #0B1120 0%, #1F2937 100%) !important; border-right: 1px solid #1F2937; }
        @media (prefers-color-scheme: light) {
            [data-testid="stSidebar"] { background: linear-gradient(180deg, #005AAA 0%, #009FE3 100%) !important; border-right: none; }
            .stApp { background-color: #F4F7FC !important; }
        }
        [data-testid="stSidebar"] *, [data-testid="stSidebar"] p { color: #FFFFFF !important; }

        @keyframes breatheSentinel { 0%, 100% { transform: scale(1) translateY(0px); opacity: 1; } 50% { transform: scale(1.03) translateY(-5px); opacity: 0.85; } }
        .sidebar-branding { animation: breatheSentinel 4s ease-in-out infinite; text-align: right; margin-bottom: 30px; padding-right: 5px; }

        [data-testid="stSidebar"] button { background-color: rgba(255,255,255,0.1) !important; border: 1px solid rgba(255,255,255,0.4) !important; border-radius: 8px !important; color: white !important; transition: all 0.3s ease; font-weight: 700 !important; }
        [data-testid="stSidebar"] button:hover { background-color: rgba(255,255,255,0.25) !important; border-color: #FFFFFF !important; transform: scale(1.02); }

        div[role="radiogroup"] > label > div:first-of-type { display: none; }
        div[role="radiogroup"] > label { padding: 10px 15px; border-radius: 8px; margin-bottom: 8px; background-color: rgba(255,255,255,0.05); border-left: 3px solid transparent; font-weight: 700; letter-spacing: 0.5px; transition: all 0.3s ease; }
        div[role="radiogroup"] > label:hover { background-color: rgba(255,255,255,0.2) !important; border-left: 4px solid #64FFDA; transform: translateX(5px); }
        div[role="radiogroup"] > label[data-baseweb="radio"] { cursor: pointer; }

        .shimmer-text { background: linear-gradient(90deg, #00A2E9, #64FFDA, #00A2E9); background-size: 200% auto; color: transparent; -webkit-background-clip: text; font-size: 13px; font-weight: 900; letter-spacing: 1px; text-transform: uppercase; margin-bottom: -10px; animation: shimmer 3s linear infinite; }
        @keyframes shimmer { to { background-position: 200% center; } }
        .dash-title { color: var(--text-color); font-weight: 900; font-size: 34px; margin-bottom: 25px; display: flex; align-items: center;}
        .live-dot { height: 12px; width: 12px; background-color: #10B981; border-radius: 50%; display: inline-block; margin-right: 10px; box-shadow: 0 0 8px #10B981; animation: pulse-dot 1.5s infinite; }
        @keyframes pulse-dot { 0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); } 70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); } 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); } }

        .ticker-wrap { width: 100%; overflow: hidden; background: var(--secondary-background-color); padding: 12px 15px; border-radius: 12px; margin-bottom: 25px; border: 1px solid rgba(128,128,128,0.15); display: flex; align-items: center; }
        .ticker-title { font-weight: 900; font-size: 12px; color: #00A2E9; letter-spacing: 1px; padding-right: 15px; border-right: 2px solid rgba(128,128,128,0.2); margin-right: 15px; white-space: nowrap; z-index: 2; }
        .ticker-move { display: inline-block; white-space: nowrap; animation: ticker 45s linear infinite; }
        .ticker-move:hover { animation-play-state: paused; }
        @keyframes ticker { 0% { transform: translateX(100%); } 100% { transform: translateX(-150%); } }

        .grad-card { min-height: 175px; display: flex; flex-direction: column; justify-content: space-between; padding: 22px; border-radius: 16px; color: white !important; position: relative; overflow: hidden; border: 1px solid rgba(255,255,255,0.15); transition: transform 0.3s ease, box-shadow 0.3s ease; }
        .grad-card:hover { transform: translateY(-5px); box-shadow: 0 12px 25px -5px rgba(0,0,0,0.2); }
        .grad-card > div, .grad-card > p, .grad-card > h2 { position: relative; z-index: 3; }
        .watermark { position: absolute; right: -10px; bottom: -20px; font-size: 90px; opacity: 0.15; transform: rotate(-15deg); pointer-events: none; z-index: 1;}
        
        .grad-card::after { content: ''; position: absolute; top: 0; left: -150%; width: 50%; height: 100%; background: linear-gradient(to right, rgba(255,255,255,0) 0%, rgba(255,255,255,0.3) 50%, rgba(255,255,255,0) 100%); transform: skewX(-25deg); animation: cardShine 6s infinite; z-index: 2; pointer-events: none; }
        @keyframes cardShine { 0% { left: -150%; } 20% { left: 250%; } 100% { left: 250%; } }
        .delay-shine-1::after { animation-delay: 0s; } .delay-shine-2::after { animation-delay: 1.5s; } .delay-shine-3::after { animation-delay: 3s; } .delay-shine-4::after { animation-delay: 4.5s; }

        .grad-blue { background: linear-gradient(135deg, #005AAA 0%, #00A2E9 100%); } .grad-green { background: linear-gradient(135deg, #059669 0%, #34D399 100%); } .grad-red { background: linear-gradient(135deg, #DC2626 0%, #F87171 100%); } .grad-purple { background: linear-gradient(135deg, #6D28D9 0%, #A78BFA 100%); }
        .grad-title { white-space: normal !important; line-height: 1.3; font-size: 14px; margin-bottom: 5px; font-weight: 700; opacity: 0.9; }
        .grad-value { font-size: 36px; font-weight: 900; margin: 0; line-height: 1;}
        .grad-badge { background: rgba(255,255,255,0.25); backdrop-filter: blur(5px); font-size: 12px; font-weight: 700; padding: 5px 12px; border-radius: 20px; align-self: flex-start; margin: 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }

        .health-meter-container { background-color: var(--secondary-background-color); padding: 20px 25px; border-radius: 16px; margin-top: 15px; margin-bottom: 25px; border: 1px solid rgba(128,128,128,0.15); box-shadow: 0 4px 15px rgba(0,0,0,0.02); }
        .health-bar { display: flex; height: 12px; border-radius: 10px; overflow: hidden; margin-top: 12px; }
        .hb-pos { background-color: #10B981; transition: width 1.5s ease-in-out; } .hb-net { background-color: #8B5CF6; transition: width 1.5s ease-in-out; } .hb-neg { background-color: #EF4444; transition: width 1.5s ease-in-out; }

        .strategic-alert { background: linear-gradient(90deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.02)); border-left: 5px solid #EF4444; padding: 20px 25px; border-radius: 12px; animation: pulse-border 2s infinite; margin-top: 10px; margin-bottom: 30px; display: flex; align-items: center; border-right: 1px solid rgba(239, 68, 68, 0.1); border-top: 1px solid rgba(239, 68, 68, 0.1); border-bottom: 1px solid rgba(239, 68, 68, 0.1); }
        @keyframes pulse-border { 0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.3); } 70% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); } 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); } }

        .table-container { background: var(--secondary-background-color); padding: 25px; border-radius: 16px; border: 1px solid rgba(128,128,128,0.15); box-shadow: 0 4px 20px rgba(0,0,0,0.03); }

        @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .anim-fade-up { animation: fadeInUp 0.6s ease-out forwards; opacity: 0; }
        .delay-1 { animation-delay: 0.1s; } .delay-2 { animation-delay: 0.2s; } .delay-3 { animation-delay: 0.3s; } .delay-4 { animation-delay: 0.4s; }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""
            <div class="sidebar-branding">
                <h1 style="margin: 0; font-size: 28px; font-weight: 900; letter-spacing: 1px; color: white;">
                    <span style="font-size: 24px; margin-right: 5px;">🦅</span>SENTINEL
                </h1>
                <p style="margin: 0; font-size: 11px; opacity: 0.8; font-weight: 600; color: white;">Sentiment Insight & Evaluator</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='text-align: right; font-size:10px; padding-right:10px; margin-bottom:10px; font-weight:800; opacity:0.6; color: white;'>GENERAL</div>", unsafe_allow_html=True)
        menu = st.radio("Pilih Menu:", ["🏠 Dashboard", "📂 Upload Data", "⚙️ Preprocessing", "🧠 Klasifikasi SVM", "📊 Hasil Analisis" , "🧪 Prediksi Real-Time"], label_visibility="collapsed")
        
        # ==========================================
        # MENU SYSTEM ADMIN (HANYA MUNCUL SETELAH LOGIN)
        # ==========================================
        st.markdown("<div style='text-align: right; font-size:10px; padding-right:10px; margin-top:30px; margin-bottom:10px; font-weight:800; opacity:0.6; color: white;'>SYSTEM ADMIN</div>", unsafe_allow_html=True)
        
        # Danger Zone (Menggunakan logika pengaman buatan Mas Iwan)
        with st.expander("Pembersihan Database ☠️"):
            st.warning("Tindakan ini akan menghancurkan SELURUH data di SQLite secara permanen!")
            konfirmasi_hapus = st.checkbox("Ya, kosongkan database.")
            
            if konfirmasi_hapus:
                if st.button("🔴 Eksekusi Hard Reset", type="primary", use_container_width=True):
                    with st.spinner("Sedang menyapu bersih database..."):
                        try:
                            from sqlalchemy import text, create_engine
                            engine = create_engine('sqlite:///database_sentimen.db')
                            with engine.begin() as conn:
                                conn.execute(text("DROP TABLE IF EXISTS tabel_klasifikasi"))
                            
                            # Bersihkan 'Ingatan Sementara' di RAM aplikasi
                            if 'data_prediksi' in st.session_state: st.session_state['data_prediksi'] = None
                            if 'data_bersih' in st.session_state: st.session_state['data_bersih'] = None
                            if 'data_mentah' in st.session_state: st.session_state['data_mentah'] = None
                            
                            st.success("✅ Boom! Database rata dengan tanah.")
                            import time
                            time.sleep(1.5)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Gagal Reset: {e}")
        
        # Tombol Logout mutlak berada di paling bawah
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Keluar (Logout)", on_click=logout, use_container_width=True)

    # --- MENU 1: DASHBOARD ---
    if menu == "🏠 Dashboard":
        st.markdown("<p style='color: #00A2E9; font-size: 13px; font-weight: 800; letter-spacing: 1px; margin-bottom: -15px;'>✨ SENTINEL ANALYTICS ENGINE</p>", unsafe_allow_html=True)
        st.markdown("<div class='dash-title anim-fade-up'><span class='live-dot'></span> Executive Dashboard</div>", unsafe_allow_html=True)

        try:
            # ==========================================
            # 1. SMART DATA SWITCH 
            # ==========================================
            engine = create_engine('sqlite:///database_sentimen.db')
            
            # Coba ambil dari memori (Session State) dulu
            if 'data_prediksi' in st.session_state and st.session_state['data_prediksi'] is not None:
                df_dash = st.session_state['data_prediksi']
            else:
                # Jika memori kosong, ambil dari MySQL. 
                # Gunakan try-except khusus agar tidak error jika tabel belum ada.
                try:
                    df_dash = pd.read_sql("SELECT * FROM tabel_klasifikasi", con=engine)
                    # Isi juga memori sementara agar sinkron
                    st.session_state['data_prediksi'] = df_dash 
                except:
                    # Jika tabel MySQL tidak ditemukan, buat Dataframe kosong agar aplikasi tidak hancur
                    df_dash = pd.DataFrame()
            
            if not df_dash.empty:
                # --- LIVE FEED TICKER ---
                latest_tweets = df_dash.tail(5).iloc[::-1]
                ticker_html = ""
                for _, row in latest_tweets.iterrows():
                    sent = str(row.get('Prediksi_Sentimen', 'Netral'))
                    teks = str(row.get('teks_bersih', row.get('Tweet Text', '')))[:70] + "..." 
                    color = "#10B981" if "Positif" in sent else "#EF4444" if "Negatif" in sent else "#8B5CF6"
                    icon = "✅" if "Positif" in sent else "🚨" if "Negatif" in sent else "💬"
                    ticker_html += f"<span style='margin-right: 40px;'><span style='color:{color}; font-weight:800;'>{icon} [{sent.upper()}]</span> <span style='color:gray; font-weight:500;'>{teks}</span></span>"

                st.markdown(f'<div class="ticker-wrap anim-fade-up delay-1"><div class="ticker-title">📡 LIVE FEED</div><div style="overflow: hidden; width: 100%;"><div class="ticker-move">{ticker_html}</div></div></div>', unsafe_allow_html=True)

                # --- KALKULASI DATA ---
                tot_data = len(df_dash)
                tot_pos = len(df_dash[df_dash['Prediksi_Sentimen'].str.contains('Positif', case=False, na=False)])
                tot_neg = len(df_dash[df_dash['Prediksi_Sentimen'].str.contains('Negatif', case=False, na=False)])
                tot_net = len(df_dash[df_dash['Prediksi_Sentimen'].str.contains('Netral', case=False, na=False)])

                # ==========================================
                # 2. CSS OVERRIDE UNTUK MEMENDEKKAN KARTU
                # ==========================================
                st.markdown("""
                    <style>
                    /* Memaksa kartu lebih pendek dan rapat */
                    .grad-card { padding: 15px 20px !important; min-height: 120px !important; }
                    .grad-title { font-size: 13px !important; margin-bottom: 5px !important; line-height: 1.2 !important; }
                    .grad-value { font-size: 30px !important; margin: 0 !important; line-height: 1 !important; }
                    .grad-badge { padding: 3px 10px !important; font-size: 10px !important; margin-top: 12px !important; display: inline-block; }
                    .watermark { font-size: 60px !important; top: -5px !important; right: 0px !important; opacity: 0.15 !important; }
                    </style>
                """, unsafe_allow_html=True)

                # --- 4 KARTU DASHBOARD ---
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.markdown(f'<div class="grad-card grad-blue anim-fade-up delay-1 delay-shine-1"><div class="watermark">📊</div><div><p class="grad-title">Total Data<br>Sentimen</p><h2 class="grad-value">{tot_data}</h2></div><p class="grad-badge">⚡ Real-time Sync</p></div>', unsafe_allow_html=True)
                with c2: st.markdown(f'<div class="grad-card grad-green anim-fade-up delay-2 delay-shine-2"><div class="watermark">✨</div><div><p class="grad-title">Sentimen<br>Positif</p><h2 class="grad-value">{tot_pos}</h2></div><p class="grad-badge">📈 Layanan Diapresiasi</p></div>', unsafe_allow_html=True)
                with c3: st.markdown(f'<div class="grad-card grad-red anim-fade-up delay-3 delay-shine-3"><div class="watermark">🚨</div><div><p class="grad-title">Sentimen<br>Negatif</p><h2 class="grad-value">{tot_neg}</h2></div><p class="grad-badge">⚠️ Prioritas Evaluasi</p></div>', unsafe_allow_html=True)
                with c4: st.markdown(f'<div class="grad-card grad-purple anim-fade-up delay-4 delay-shine-4"><div class="watermark">💬</div><div><p class="grad-title">Sentimen<br>Netral</p><h2 class="grad-value">{tot_net}</h2></div><p class="grad-badge">🔍 Opini Objektif</p></div>', unsafe_allow_html=True)

                # --- GRAFIK DAN KESEHATAN SISTEM (SISA KODEMU TETAP AMAN) ---
                pct_pos = (tot_pos / tot_data * 100) if tot_data > 0 else 0
                pct_neg = (tot_neg / tot_data * 100) if tot_data > 0 else 0
                pct_net = (tot_net / tot_data * 100) if tot_data > 0 else 0
                
                st.markdown(f"""
                    <div class="health-meter-container anim-fade-up delay-4">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h4 style="margin:0; font-size:16px; color:var(--text-color); font-weight:800;">Indikator Kesehatan Sistem (Sentiment Health Ratio)</h4>
                            <span style="font-size:13px; color:gray; font-weight:600;">Distribusi Proporsional</span>
                        </div>
                        <div class="health-bar">
                            <div class="hb-pos" style="width: {pct_pos}%;" title="Positif: {pct_pos:.1f}%"></div>
                            <div class="hb-net" style="width: {pct_net}%;" title="Netral: {pct_net:.1f}%"></div>
                            <div class="hb-neg" style="width: {pct_neg}%;" title="Negatif: {pct_neg:.1f}%"></div>
                        </div>
                        <div style="display:flex; gap:15px; margin-top:8px; font-size:12px; font-weight:600;">
                            <span style="color:#10B981;">● Positif {pct_pos:.1f}%</span>
                            <span style="color:#8B5CF6;">● Netral {pct_net:.1f}%</span>
                            <span style="color:#EF4444;">● Negatif {pct_neg:.1f}%</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                col_chart1, col_chart2 = st.columns([4, 6])
                with col_chart1:
                    st.markdown("<div class='anim-fade-up delay-4'>", unsafe_allow_html=True)
                    st.markdown("<h3 style='font-weight: 900; color: var(--text-color); font-size: 20px; margin-bottom: 0px;'>Distribusi Makro</h3>", unsafe_allow_html=True)
                    st.markdown("<p style='color: gray; font-weight: 600; font-size: 12px; margin-bottom: 10px;'>Persentase Keseluruhan Sentimen</p>", unsafe_allow_html=True)
                    
                    sentimen_counts = df_dash['Prediksi_Sentimen'].value_counts().reset_index()
                    sentimen_counts.columns = ['Sentimen', 'Jumlah']
                    fig_pie = px.pie(sentimen_counts, values='Jumlah', names='Sentimen', hole=0.65,
                                     color='Sentimen', color_discrete_map={'Positif':'#10B981', 'Negatif':'#EF4444', 'Netral':'#8B5CF6'})
                    fig_pie.update_traces(textinfo='percent', textfont_size=14, textfont_color='white', hoverinfo="label+value", marker=dict(line=dict(color='rgba(0,0,0,0)', width=0)))
                    fig_pie.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0), height=320,
                        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(color='gray', size=12)),
                        annotations=[dict(text=f"<span style='font-size:30px; font-weight:900; color:#00A2E9;'>{tot_data}</span><br><span style='font-size:12px; color:gray;'>Total Data</span>", x=0.5, y=0.5, showarrow=False)]
                    )
                    st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_chart2:
                    st.markdown("<div class='anim-fade-up delay-4'>", unsafe_allow_html=True)
                    st.markdown("<h3 style='font-weight: 900; color: var(--text-color); font-size: 20px; margin-bottom: 0px;'>Analisis Mikro per Aspek</h3>", unsafe_allow_html=True)
                    st.markdown("<p style='color: gray; font-weight: 600; font-size: 12px; margin-bottom: 10px;'>Perbandingan Sentimen per Kategori Layanan</p>", unsafe_allow_html=True)
                    
                    if 'Prediksi_Aspek' in df_dash.columns and 'Prediksi_Sentimen' in df_dash.columns:
                        df_grouped = df_dash.groupby(['Prediksi_Aspek', 'Prediksi_Sentimen']).size().reset_index(name='Jumlah')
                        fig_bar = px.bar(df_grouped, x='Prediksi_Aspek', y='Jumlah', color='Prediksi_Sentimen', barmode='group', text='Jumlah',
                                         color_discrete_map={'Positif':'#10B981', 'Negatif':'#EF4444', 'Netral':'#8B5CF6'})
                        fig_bar.update_traces(marker_line_width=0, opacity=0.9, textposition='outside', textfont=dict(color='gray', size=12, weight='bold'))
                        fig_bar.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0), height=320,
                            xaxis=dict(showgrid=False, title=None, tickfont=dict(color='gray', size=12, weight='bold')),
                            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.15)', title=None, tickfont=dict(color='gray')),
                            legend=dict(title=None, orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1, font=dict(color='gray', size=12))
                        )
                        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
                    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
                # 3. DUAL ALERT SYSTEM (POSITIF & NEGATIF)
                # ==========================================
                if 'Prediksi_Aspek' in df_dash.columns:
                    df_negatif = df_dash[df_dash['Prediksi_Sentimen'].str.contains('Negatif', case=False, na=False)]
                    df_positif = df_dash[df_dash['Prediksi_Sentimen'].str.contains('Positif', case=False, na=False)]
                    
                    st.markdown("<br>", unsafe_allow_html=True) # Spasi sedikit
                    col_alert1, col_alert2 = st.columns(2)
                    
                    with col_alert1:
                        if not df_negatif.empty:
                            aspek_terburuk = df_negatif['Prediksi_Aspek'].value_counts().idxmax()
                            jumlah_buruk = df_negatif['Prediksi_Aspek'].value_counts().max()
                            
                            st.markdown(f"""
                                <div class="anim-fade-up delay-4" style="background-color: rgba(239, 68, 68, 0.05); border-left: 5px solid #EF4444; padding: 15px; border-radius: 5px; height: 100%;">
                                    <h4 style='color: #EF4444; margin: 0 0 5px 0; font-weight: 800; font-size: 15px;'>🚨 PRIORITAS EVALUASI</h4>
                                    <p style='color: gray; margin: 0; font-size: 13px;'>Keluhan tertinggi ada pada aspek <b>'{aspek_terburuk}'</b> ({jumlah_buruk} ulasan). Direkomendasikan evaluasi teknis pada sektor ini.</p>
                                </div>
                            """, unsafe_allow_html=True)

                    with col_alert2:
                        if not df_positif.empty:
                            aspek_terbaik = df_positif['Prediksi_Aspek'].value_counts().idxmax()
                            jumlah_baik = df_positif['Prediksi_Aspek'].value_counts().max()
                            
                            st.markdown(f"""
                                <div class="anim-fade-up delay-4" style="background-color: rgba(16, 185, 129, 0.05); border-left: 5px solid #10B981; padding: 15px; border-radius: 5px; height: 100%;">
                                    <h4 style='color: #10B981; margin: 0 0 5px 0; font-weight: 800; font-size: 15px;'>✨ LAYANAN DIAPRESIASI</h4>
                                    <p style='color: gray; margin: 0; font-size: 13px;'>Nasabah sangat puas dengan aspek <b>'{aspek_terbaik}'</b> ({jumlah_baik} ulasan). Pertahankan standar pada sektor ini.</p>
                                </div>
                            """, unsafe_allow_html=True)

                # ==========================================
                # 4. TABEL DATA TERKINI (KOLOM DIBERSIHKAN)
                # ==========================================
                st.markdown("<div class='table-container anim-fade-up delay-4' style='margin-top: 30px;'>", unsafe_allow_html=True)
                st.markdown("<h3 style='font-weight: 900; color: var(--text-color); font-size: 20px; margin-bottom: 0px;'>Rekam Data Terkini</h3>", unsafe_allow_html=True)
                st.markdown("<p style='color: gray; font-weight: 500; font-size: 13px; margin-bottom: 15px;'>10 hasil prediksi terbaru dari mesin Support Vector Machine (SVM)</p>", unsafe_allow_html=True)
                
                df_tampil = df_dash.tail(10).iloc[::-1].copy()
                
                # KITA TEMBAK LANGSUNG KE TEKS BERSIH (PASTI ADA ISINYA)
                if 'teks_bersih' in df_tampil.columns and 'Prediksi_Sentimen' in df_tampil.columns:
                    # Ambil 3 kolom inti
                    df_bersih_tabel = df_tampil[['teks_bersih', 'Prediksi_Sentimen', 'Prediksi_Aspek']]
                    
                    # Ganti nama kolom (Perhatikan! Nanti namanya akan berubah di layar)
                    df_bersih_tabel.columns = ['Isi Ulasan (Teks Bersih)', 'Hasil Sentimen', 'Kategori Aspek']
                    
                    # ---> KODE LAMA (st.dataframe) DIGANTI JADI INI <---
                    tampilkan_tabel_cantik(df_bersih_tabel)
                else:
                    # Fallback jika terjadi error
                    # ---> KODE LAMA (st.dataframe) DIGANTI JADI INI <---
                    tampilkan_tabel_cantik(df_tampil)
                    
                st.markdown("</div>", unsafe_allow_html=True)

            else:
                st.info("💡 Database kosong dan belum ada data yang diproses. Silakan mulai di menu Upload Data.")

        except Exception as e:
            st.error(f"Gagal memuat visualisasi: {e}")

    # --- MENU 2: UPLOAD DATA ---
    elif menu == "📂 Upload Data":
        st.markdown("""
            <style>
            /* Reset Spacing */
            .block-container { padding-top: 1.5rem !important; }
            
            /* Modern Header */
            .header-left { margin-bottom: 40px; border-left: 5px solid #00A2E9; padding-left: 20px; }
            .shimmer-mini { color: #00A2E9; font-size: 12px; font-weight: 800; letter-spacing: 2px; text-transform: uppercase; }
            
            /* Aktor Utama: Sleek Uploader */
            .stFileUploader {
                border: 2px dashed rgba(0, 162, 233, 0.2);
                border-radius: 15px; padding: 20px;
                background: rgba(128,128,128,0.02);
                transition: all 0.3s ease;
            }
            .stFileUploader:hover {
                border-color: #00A2E9;
                background: rgba(0, 162, 233, 0.03);
            }

            /* Fitur Baru: Intelligence Panel */
            .intel-card {
                background: white; border-radius: 12px; padding: 20px;
                border: 1px solid rgba(128,128,128,0.1);
                box-shadow: 0 4px 12px rgba(0,0,0,0.03);
                display: flex; align-items: center; gap: 15px;
            }
            @media (prefers-color-scheme: dark) {
                .intel-card { background: #1E293B; border-color: rgba(255,255,255,0.1); }
            }
            .intel-icon { font-size: 24px; }
            .intel-title { font-size: 12px; color: gray; font-weight: 700; text-transform: uppercase; }
            .intel-value { font-size: 16px; font-weight: 800; color: var(--text-color); }

            /* Animasi Fade In */
            .fade-in { animation: fadeIn 0.8s ease-in-out; }
            @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
            </style>
        """, unsafe_allow_html=True)

        # --- HEADER ---
        st.markdown("""
            <div class="header-left fade-in">
                <p class="shimmer-mini">Data Management</p>
                <h1 style="font-weight:900; margin-top:-5px;">Dataset Acquisition</h1>
                <p style="color:gray; margin-top:-10px;">Kelola dan verifikasi dataset ulasan m-Banking sebelum tahap pemrosesan.</p>
            </div>
        """, unsafe_allow_html=True)

        # --- AKTOR UTAMA: UPLOADER ---
        st.markdown("### 📥 Unggah Dataset")
        uploaded_file = st.file_uploader("Pilih file CSV atau Excel", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file, encoding='latin-1')
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state['data_mentah'] = df
                st.session_state['data_bersih'] = None 
                st.session_state['halaman_saat_ini'] = 1 
                st.success(f"Berhasil mengimpor {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error membaca file: {e}")

        # --- FITUR INTELLIGENCE PANEL (HARD-OVERRIDE YELLOW) ---
        st.markdown("""
            <style>
            /* Memaksa warna teks di dalam kartu intel agar tidak pudar */
            .intel-card {
                background: #112240 !important; /* Navy Gelap */
                border: 1.5px solid rgba(255, 215, 0, 0.4) !important;
                border-radius: 15px; padding: 20px; display: flex; align-items: center; gap: 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            }
            .intel-title-yellow {
                font-size: 12px !important;
                color: #FFFFFF !important; /* Judul kita buat Putih Terang */
                font-weight: 700 !important;
                text-transform: uppercase !important;
                letter-spacing: 1px !important;
                margin: 0 !important;
            }
            .intel-value-yellow {
                font-size: 20px !important;
                font-weight: 900 !important;
                color: #FFD700 !important; /* Nilai kita buat KUNING EMAS */
                margin: 0 !important;
                display: block !important;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_inf1, col_inf2, col_inf3 = st.columns(3)

        has_data = st.session_state['data_mentah'] is not None
        df_tmp = st.session_state['data_mentah']

        with col_inf1:
            val = f"{len(df_tmp)} Baris" if has_data else "Menunggu..."
            st.markdown(f"""
                <div class="intel-card">
                    <span style="font-size:24px;">📊</span>
                    <div>
                        <p class="intel-title-yellow">KAPASITAS DATA</p>
                        <p class="intel-value-yellow">{val}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col_inf2:
            val = f"{len(df_tmp.columns)} Kolom" if has_data else "N/A"
            st.markdown(f"""
                <div class="intel-card">
                    <span style="font-size:24px;">🔍</span>
                    <div>
                        <p class="intel-title-yellow">METADATA KOLOM</p>
                        <p class="intel-value-yellow">{val}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col_inf3:
            val = "Terverifikasi ✅" if has_data else "Proses..."
            st.markdown(f"""
                <div class="intel-card">
                    <span style="font-size:24px;">🛡️</span>
                    <div>
                        <p class="intel-title-yellow">INTEGRITAS DATA</p>
                        <p class="intel-value-yellow">{val}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        # --- PREVIEW TABEL ---
        if has_data:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("### 📋 Preview Dataset Mentah")
            st.dataframe(df_tmp.head(10), use_container_width=True)
            
            # Tips Tambahan agar space tidak kosong
            st.info("💡 **Tips:** Lanjutkan ke menu **Preprocessing** untuk membersihkan simbol, angka, dan melakukan stemming pada ulasan di atas.")
        else:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.warning("Silakan upload file untuk mengaktifkan Intelligence Panel dan melihat preview data.")

    # ==========================================
    # 5. MENU 3: PREPROCESSING 
    # ==========================================
    elif menu == "⚙️ Preprocessing":
        st.markdown("""
            <style>
            .block-container { padding-top: 1.5rem !important; }
            
            /* Grid untuk 6 Tahapan agar tidak sesak */
            .step-grid { 
                display: grid; 
                grid-template-columns: repeat(3, 1fr); 
                gap: 15px; 
                margin-bottom: 35px; 
            }
            @media (max-width: 768px) { .step-grid { grid-template-columns: repeat(2, 1fr); } }

            .step-item { 
                background: rgba(128,128,128,0.05); 
                border: 1px solid rgba(128,128,128,0.1);
                padding: 12px; border-radius: 10px; 
                text-align: center; font-size: 11px; 
                font-weight: 800; color: gray;
                transition: 0.3s;
            }
            .step-done { 
                border-color: #00A2E9; 
                color: #00A2E9; 
                background: rgba(0, 162, 233, 0.05);
                box-shadow: 0 4px 10px rgba(0,162,233,0.1);
            }
            
            /* Card Perbandingan Kuning Navy */
            .compare-card {
                background: #112240; border-radius: 15px; padding: 20px;
                border: 1.5px solid rgba(255, 215, 0, 0.3); margin-bottom: 25px;
            }
            .compare-label { font-size: 10px; color: rgba(255,255,255,0.6); font-weight: 800; text-transform: uppercase; }
            .compare-text { font-size: 14px; color: #FFD700; font-weight: 600; margin-top: 5px; }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("<div class='shimmer-text'>⚙️ DATA REFINERY PIPELINE</div>", unsafe_allow_html=True)
        st.markdown("<h1 class='dash-title anim-fade-up'>Preprocessing Laboratory</h1>", unsafe_allow_html=True)

        if st.session_state['data_mentah'] is not None:
            df = st.session_state['data_mentah']
            
            # --- VISUALISASI 6 TAHAP SESUAI PROPOSAL ---
            st.markdown(f"""
                <div class='step-grid anim-fade-up'>
                    <div class='step-item {'step-done' if st.session_state['data_bersih'] is not None else ''}'>1. CLEANING</div>
                    <div class='step-item {'step-done' if st.session_state['data_bersih'] is not None else ''}'>2. CASE FOLDING</div>
                    <div class='step-item {'step-done' if st.session_state['data_bersih'] is not None else ''}'>3. TOKENIZING</div>
                    <div class='step-item {'step-done' if st.session_state['data_bersih'] is not None else ''}'>4. NORMALIZATION</div>
                    <div class='step-item {'step-done' if st.session_state['data_bersih'] is not None else ''}'>5. STOPWORD REMOVAL</div>
                    <div class='step-item {'step-done' if st.session_state['data_bersih'] is not None else ''}'>6. STEMMING</div>
                </div>
            """, unsafe_allow_html=True)

            # Control Panel
            st.markdown("<div class='table-container'>", unsafe_allow_html=True)
            col_ctrl1, col_ctrl2 = st.columns([2, 1])
            with col_ctrl1:
                kolom_teks = st.selectbox("🎯 Pilih Kolom Teks (Input):", df.columns)
            with col_ctrl2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.session_state['data_bersih'] is None:
                    proses_btn = st.button("🚀 Jalankan Pipeline", type="primary", use_container_width=True)
                else:
                    proses_btn = st.button("🔄 Ulangi Preprocessing", use_container_width=True)
            st.markdown("</div><br>", unsafe_allow_html=True)

            if proses_btn:
                with st.spinner('🛠️ Menjalankan 6 tahap algoritma... Mohon tunggu.'):
                    df_bersih = df.copy()
                    # Memanggil fungsi dari preprocessing.py
                    df_bersih['teks_bersih'] = df_bersih[kolom_teks].apply(bersihkan_teks)
                    st.session_state['data_bersih'] = df_bersih
                    st.session_state['halaman_saat_ini'] = 1
                st.success("🎉 Seluruh ulasan telah berhasil diproses sesuai metodologi!")
                st.rerun()

            # Preview Perbandingan
            if st.session_state['data_bersih'] is not None:
                st.markdown("### 🔍 Metodologi Check")
                sample_raw = st.session_state['data_bersih'][kolom_teks].iloc[0]
                sample_clean = st.session_state['data_bersih']['teks_bersih'].iloc[0]
                
                c_samp1, c_samp2 = st.columns(2)
                with c_samp1:
                    st.markdown(f"<div class='compare-card'><p class='compare-label'>TEKS ASLI</p><p class='compare-text' style='color: white;'>\"{sample_raw}\"</p></div>", unsafe_allow_html=True)
                with c_samp2:
                    st.markdown(f"<div class='compare-card'><p class='compare-label'>HASIL FINAL</p><p class='compare-text'>\"{sample_clean}\"</p></div>", unsafe_allow_html=True)

                # Tabel Data
                st.markdown("### 📋 Cleaned Dataset Records")
                df_tampil = st.session_state['data_bersih'][[kolom_teks, 'teks_bersih']].copy()
                df_tampil.index = df_tampil.index + 1
                
                items_per_page = 10
                total_pages = max(1, (len(df_tampil) - 1) // items_per_page + 1)
                start_idx = (st.session_state['halaman_saat_ini'] - 1) * items_per_page
                
                st.dataframe(df_tampil.iloc[start_idx : start_idx + items_per_page], use_container_width=True)
                
                # Pagination
                st.markdown("<br>", unsafe_allow_html=True)
                p1, p2, p3, p4, p5 = st.columns([2, 1, 2, 1, 2])
                with p2:
                    if st.button("⬅️ Prev", disabled=(st.session_state['halaman_saat_ini'] == 1)):
                        st.session_state['halaman_saat_ini'] -= 1
                        st.rerun()
                with p3:
                    st.markdown(f"<p style='text-align: center; font-weight:800; padding-top:10px;'>Halaman {st.session_state['halaman_saat_ini']} / {total_pages}</p>", unsafe_allow_html=True)
                with p4:
                    if st.button("Next ➡️", disabled=(st.session_state['halaman_saat_ini'] == total_pages)):
                        st.session_state['halaman_saat_ini'] += 1
                        st.rerun()
        else:
            st.info("💡 Hubungkan dataset terlebih dahulu melalui menu **📂 Upload Data**.")

# ==========================================
    # 6. MENU 4: KLASIFIKASI SVM 
    # ==========================================
    elif menu == "🧠 Klasifikasi SVM":
        # --- CSS KHUSUS HYBRID ENGINE ---
        st.markdown("""
            <style>
            .block-container { padding-top: 1.5rem !important; }
            
            /* Desain Tab Streamlit Premium */
            .stTabs [data-baseweb="tab-list"] { gap: 20px; }
            .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 10px 10px 0px 0px; padding: 10px 20px; font-weight: 800; color: gray; }
            .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: rgba(0, 162, 233, 0.1); color: #00A2E9; border-bottom: 3px solid #00A2E9; }
            
            /* Metrik Evaluasi Model */
            .eval-card { background: #112240; border: 1px solid rgba(255, 215, 0, 0.3); border-radius: 15px; padding: 20px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
            .eval-title { font-size: 12px; color: rgba(255,255,255,0.7); font-weight: 800; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
            .eval-score { font-size: 28px; font-weight: 900; color: #FFD700; margin: 0; line-height: 1; }
            
            /* Kartu Engine Tab 2 */
            .engine-card { background: #112240; border: 1.5px solid rgba(100, 255, 218, 0.3); border-radius: 16px; padding: 30px; margin-bottom: 30px; }
            .status-ready { color: #FFD700; font-weight: 900; font-size: 12px; background: rgba(255, 215, 0, 0.1); padding: 5px 12px; border-radius: 8px; }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("<div class='shimmer-text' style='text-align:left;'>🧠 SENTINEL HYBRID AI</div>", unsafe_allow_html=True)
        st.markdown("<h1 class='dash-title' style='margin-top:-5px;'>Support Vector Machine Engine</h1>", unsafe_allow_html=True)

        if st.session_state['data_bersih'] is None:
            st.warning("⚠️ Menunggu Data Pipeline: Silakan selesaikan tahap Preprocessing terlebih dahulu.")
        else:
            # MEMBUAT DUA TAB INTERAKTIF
            tab1, tab2 = st.tabs(["📊 Evaluasi Model (Sesuai Proposal)", "🚀 Prediksi Massal (Mode Industri)"])

            # ==========================================
            # TAB 1: EVALUASI MODEL (DUAL SYSTEM: SENTIMEN & ASPEK)
            # ==========================================
            with tab1:
                st.markdown("<br><h3 style='color: var(--text-color);'>🎛️ Evaluasi Model SVM</h3>", unsafe_allow_html=True)
    
                df_eval = st.session_state['data_bersih']
    
                mode_eval = st.radio("🎯 Pilih Fokus Evaluasi:", 
                          ["🧠 Evaluasi Model Sentimen", "🏷️ Evaluasi Model Aspek"], 
                          horizontal=True)
    
                if "Sentimen" in mode_eval:
                    default_col = [c for c in df_eval.columns if 'Sentimen' in c]
                    kolom_label = st.selectbox(
                        "Pilih Kolom Label SENTIMEN:", 
                        df_eval.columns,
                        index=df_eval.columns.tolist().index(default_col[0]) if default_col else 0
                    )
                else:
                    default_col = [c for c in df_eval.columns if 'Aspek' in c]
                    kolom_label = st.selectbox(
                        "Pilih Kolom Label ASPEK:", 
                        df_eval.columns,
                        index=df_eval.columns.tolist().index(default_col[0]) if default_col else 0
                    )
    
                split_ratio = st.slider("Split Data Ratio (% Training)", 50, 90, 80, 5)
                test_size = 100 - split_ratio
                st.info(f"💡 Konfigurasi: **{split_ratio}% Training** / **{test_size}% Testing**")
    
                if st.button(f"⚙️ Evaluasi dengan Model PKL", type="primary", use_container_width=True):
                    with st.spinner("🧠 Mengevaluasi menggunakan model yang sudah dilatih..."):
                        try:
                            from sklearn.model_selection import train_test_split
                            from sklearn.metrics import (accuracy_score, precision_score, 
                                             recall_score, confusion_matrix, 
                                             classification_report)
                            import pickle

                            df_eval = df_eval.dropna(subset=['teks_bersih', kolom_label])
                            X = df_eval['teks_bersih']
                            y = df_eval[kolom_label]

                            # Split data
                            _, X_test, _, y_test = train_test_split(
                                X, y, test_size=test_size/100.0, random_state=42
                            )

                            # ✅ LOAD PKL — pakai model yang sudah ditraining di VS Code
                            with open('tfidf.pkl', 'rb') as f:
                                tfidf_loaded = pickle.load(f)

                            if "Sentimen" in mode_eval:
                                with open('svm_sentimen.pkl', 'rb') as f:
                                    model_loaded = pickle.load(f)
                            else:
                                with open('svm_aspek.pkl', 'rb') as f:
                                    model_loaded = pickle.load(f)

                            # Transform pakai tfidf dari pkl (bukan fit ulang!)
                            X_test_vec = tfidf_loaded.transform(X_test)

                            y_pred = model_loaded.predict(X_test_vec)
                           
                            
                            acc  = accuracy_score(y_test, y_pred)
                            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
                            rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)

                            st.success(f"✅ Evaluasi selesai menggunakan model PKL!")

                            c_ev1, c_ev2, c_ev3 = st.columns(3)
                            with c_ev1:
                                st.markdown(f"<div class='eval-card'><p class='eval-title'>Akurasi Model</p><p class='eval-score'>{acc*100:.2f}%</p></div>", unsafe_allow_html=True)
                            with c_ev2:
                                st.markdown(f"<div class='eval-card'><p class='eval-title'>Tingkat Presisi</p><p class='eval-score'>{prec*100:.2f}%</p></div>", unsafe_allow_html=True)
                            with c_ev3:
                                st.markdown(f"<div class='eval-card'><p class='eval-title'>Tingkat Recall</p><p class='eval-score'>{rec*100:.2f}%</p></div>", unsafe_allow_html=True)

                            # Confusion Matrix & Classification Report
                            st.markdown("<br><h3 style='color: var(--text-color);'>📉 Matrix & Laporan Klasifikasi</h3>", unsafe_allow_html=True)
                           # 🚀 1. RASIO 50:50 YANG PROPORSIONAL
                            c_mat1, c_mat2 = st.columns([1.1, 0.9])

                            with c_mat1:
                                st.markdown("**Visualisasi Confusion Matrix**")
                                
                                # 🚀 2. KUNCI KONSISTENSI LABEL OTOMATIS (ANTI-ERROR)
                                # Membaca langsung dari data lalu diurutkan sesuai abjad
                                labels = sorted(y.astype(str).unique())
                                
                                cm = confusion_matrix(y_test, y_pred, labels=labels)
                                color_scale = 'Blues' if "Sentimen" in mode_eval else 'Purples'
                                
                                fig, ax = plt.subplots(figsize=(6, 5), dpi=300) 
                                
                                fig.patch.set_facecolor('none')
                                ax.patch.set_facecolor('none')
                                
                                heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap=color_scale, 
                                            cbar=True, 
                                            annot_kws={"size": 14, "weight": "bold"}, 
                                            ax=ax)
                                
                                # ======================================================
                                # 🚀 WARNA CHAMELEON (Aman di Light & Dark Mode)
                                # ======================================================
                                target_color = '#64748b' # Slate Gray (Abu-abu kebiruan)
                                
                                ax.set_xlabel("Prediksi Mesin", fontsize=12, fontweight='bold', labelpad=10, color=target_color)
                                ax.set_ylabel("Aktual", fontsize=12, fontweight='bold', labelpad=10, color=target_color)
                                
                                # 🚀 LABEL DIMIRINGKAN 40 DERAJAT AGAR RAPI
                                ax.set_xticklabels(labels, fontsize=10, rotation=40, ha='right', color=target_color)
                                ax.set_yticklabels(labels, fontsize=10, rotation=0, color=target_color)
                                
                                cbar = heatmap.collections[0].colorbar
                                cbar.ax.yaxis.set_tick_params(color=target_color, labelcolor=target_color)
                                
                                ax.tick_params(colors=target_color)
                                
                                # Memastikan layout grafik tidak terpotong
                                fig.tight_layout()
                                
                                st.pyplot(fig, clear_figure=True)

                            with c_mat2:
                                st.markdown("**Tabel Laporan Klasifikasi**")
                                report_dict = classification_report(
                                    y_test, y_pred, labels=labels, output_dict=True, zero_division=0
                                )
                                df_report = pd.DataFrame(report_dict).transpose().round(2)
                                st.dataframe(df_report, use_container_width=True)

                        except FileNotFoundError:
                            st.error("❌ File .pkl tidak ditemukan! Pastikan tfidf.pkl, svm_sentimen.pkl, svm_aspek.pkl ada di folder yang sama.")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")

            # ==========================================
            # TAB 2: PREDIKSI MASSAL 
            # ==========================================
            with tab2:
                jml_data = len(st.session_state['data_bersih'])
                st.markdown(f"""
                    <br>
                    <div class="engine-card">
                        <div style="color:#64FFDA; font-weight:800; margin-bottom:15px;">⚙️ SYSTEM READINESS CHECK</div>
                        <div style="display:flex; justify-content:space-between; margin-bottom:10px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom:10px;">
                            <span style="color:white;">Model Sentimen (.pkl)</span> <span class="status-ready">✅ LOADED</span>
                        </div>
                        <div style="display:flex; justify-content:space-between;">
                            <span style="color:white;">Data Input Pipeline</span> <span style="color:#00A2E9; font-weight:900;">🚀 {jml_data} BARIS SIAP DIEKSEKUSI</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                if st.button("🚀 AKTIFKAN MESIN PREDIKSI", type="primary", use_container_width=True):
                            with st.spinner("🧠 Mesin SVM sedang membedah pola kata dan bobot sentimen..."):
                                import pickle
                                import time
                                try:
                                    time.sleep(1.5) 
                                    
                                    with open('tfidf.pkl', 'rb') as f:
                                        tfidf = pickle.load(f)
                                    with open('svm_sentimen.pkl', 'rb') as f:
                                        svm_sentimen = pickle.load(f)
                                    with open('svm_aspek.pkl', 'rb') as f:
                                        svm_aspek = pickle.load(f)
                                        
                                    df_prediksi = st.session_state['data_bersih'].copy()
                                    X_baru = tfidf.transform(df_prediksi['teks_bersih'])
                                    
                                    df_prediksi['Prediksi_Sentimen'] = svm_sentimen.predict(X_baru)
                                    df_prediksi['Prediksi_Aspek'] = svm_aspek.predict(X_baru)
                                    
                                    # ==========================================================
                                    # LOGIKA "GUDANG CERDAS": GABUNG -> SARING -> SIMPAN
                                    # ==========================================================
                                    from sqlalchemy import create_engine
                                    import pandas as pd
                                    engine = create_engine('sqlite:///database_sentimen.db')
                                    
                                    # 1. Panggil data lama dari Gudang (SQLite)
                                    try:
                                        df_lama = pd.read_sql("SELECT * FROM tabel_klasifikasi", con=engine)
                                    except:
                                        df_lama = pd.DataFrame() # Jika gudang masih kosong
                                        
                                    # 2. Saring data baru HANYA 3 kolom inti
                                    df_baru_inti = df_prediksi[['teks_bersih', 'Prediksi_Sentimen', 'Prediksi_Aspek']].copy()
                                    
                                    # 3. Gabungkan data lama dengan data baru
                                    df_gabungan = pd.concat([df_lama, df_baru_inti], ignore_index=True)
                                    
                                    # 4. Hapus duplikat (Teks ulasan yang sama tidak akan masuk dua kali)
                                    df_bersih_final = df_gabungan.drop_duplicates(subset=['teks_bersih'], keep='first')
                                    
                                    # 5. Simpan permanen ke SQLite (Aman ditimpa karena datanya sudah gabungan)
                                    df_bersih_final.to_sql('tabel_klasifikasi', con=engine, if_exists='replace', index=False)
                                    
                                    # 6. Update memori RAM agar Dashboard langsung sinkron
                                    st.session_state['data_prediksi'] = df_bersih_final
                                        
                                    total_sekarang = len(df_bersih_final)
                                    st.success(f"✅ Selesai! Total {total_sekarang} data berhasil digabungkan dan tersimpan permanen di SQLite!")
                                    st.rerun()
                                    
                                except FileNotFoundError:
                                    st.error("❌ File model (.pkl) tidak ditemukan! Pastikan file training sudah ada di folder aplikasi.")
                                except Exception as e:
                                    st.error(f"❌ Terjadi kesalahan pada sistem: {e}")

                # --- BAGIAN HASIL PREDIKSI DI TAB 2 ---
                if 'data_prediksi' in st.session_state and st.session_state['data_prediksi'] is not None:
                    st.markdown("<hr style='opacity: 0.2; margin: 30px 0;'>", unsafe_allow_html=True)
                    df_res = st.session_state['data_prediksi']
                    
                    # 1. TAMPILKAN METRIK SEMENTARA
                    pos_count = len(df_res[df_res['Prediksi_Sentimen'].str.contains('Positif', case=False, na=False)])
                    neg_count = len(df_res[df_res['Prediksi_Sentimen'].str.contains('Negatif', case=False, na=False)])
                    net_count = len(df_res[df_res['Prediksi_Sentimen'].str.contains('Netral', case=False, na=False)])
                    
                    c_res1, c_res2, c_res3 = st.columns(3)
                    with c_res1: st.metric("✅ Sentimen Positif", f"{pos_count} Data")
                    with c_res2: st.metric("🚨 Sentimen Negatif", f"{neg_count} Data")
                    with c_res3: st.metric("💬 Sentimen Netral", f"{net_count} Data")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # 2. TOMBOL DOWNLOAD (LAPORAN CSV)
                    csv = df_res.to_csv(index=False).encode('utf-8')
                    st.download_button("💾 Download Hasil Prediksi (.CSV) untuk Laporan", data=csv, file_name='hasil_prediksi.csv', mime='text/csv', use_container_width=True)
    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.dataframe(df_res[['teks_bersih', 'Prediksi_Sentimen', 'Prediksi_Aspek']].head(100), use_container_width=True)

    # ==========================================
    # 7. MENU 5: HASIL (DASHBOARD INSIGHT)
    # ==========================================
    elif menu == "📊 Hasil Analisis":
        st.markdown("<div class='shimmer-text' style='text-align:left;'>📊 ANALYTICS HUB</div>", unsafe_allow_html=True)
        st.markdown("<h1 class='dash-title' style='margin-top:-5px;'>Dashboard Hasil Analisis</h1>", unsafe_allow_html=True)

        # Cek apakah data prediksi sudah ada
        if 'data_prediksi' not in st.session_state or st.session_state['data_prediksi'] is None:
            st.warning("⚠️ Data Belum Tersedia: Silakan lakukan Klasifikasi SVM terlebih dahulu di Menu 4.")
        else:
            df_hasil = st.session_state['data_prediksi']
            
            # --- 1. SUMMARY CARDS (ULTIMATE TOUCH) ---
            total_data = len(df_hasil)
            top_aspek = df_hasil['Prediksi_Aspek'].mode()[0]
            pos_rate = (len(df_hasil[df_hasil['Prediksi_Sentimen'] == 'Positif']) / total_data) * 100

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Ulasan", f"{total_data} Tweet")
            with c2:
                st.metric("Aspek Terpopuler", top_aspek)
            with c3:
                st.metric("Sentimen Positif", f"{pos_rate:.1f}%")

            st.markdown("<hr style='opacity: 0.1;'>", unsafe_allow_html=True)

            # --- 2. DISTRIBUSI SENTIMEN BERDASARKAN ASPEK (SESUAI RANCANGAN) ---
            st.markdown("### 📈 Distribusi Sentimen Berdasarkan Aspek")
            
            # Olah data untuk Grafik Stacked Bar
            df_chart = df_hasil.groupby(['Prediksi_Aspek', 'Prediksi_Sentimen']).size().reset_index(name='Jumlah')
            
            fig = px.bar(df_chart, 
                         x="Prediksi_Aspek", 
                         y="Jumlah", 
                         color="Prediksi_Sentimen",
                         color_discrete_map={'Positif': "#5ebe74", 'Netral': "#0787ff", 'Negatif': '#dc3545'},
                         barmode="stack",
                         template="plotly_dark",
                         text_auto=True)
            
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend_title="Sentimen",
                xaxis_title="Kategori Aspek",
                yaxis_title="Jumlah Ulasan"
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- 3. TABEL HASIL KLASIFIKASI (SESUAI RANCANGAN) ---
            st.markdown("### 📋 Tabel Detail Klasifikasi")
            
            # Fitur Filter (Ultimate)
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                filter_sentimen = st.multiselect("Filter Sentimen:", options=['Positif', 'Netral', 'Negatif'], default=['Positif', 'Netral', 'Negatif'])
            with col_f2:
                filter_aspek = st.multiselect("Filter Aspek:", options=df_hasil['Prediksi_Aspek'].unique(), default=df_hasil['Prediksi_Aspek'].unique())

            df_filtered = df_hasil[
                (df_hasil['Prediksi_Sentimen'].isin(filter_sentimen)) & 
                (df_hasil['Prediksi_Aspek'].isin(filter_aspek))
            ]

            # ==========================================
            # TAMPILKAN TABEL CANTIK (KODE BARU)
            # ==========================================
            if not df_filtered.empty:
                # 1. Ambil kolom inti dan ganti nama agar seragam dengan Dashboard
                df_hasil_tampil = df_filtered[['teks_bersih', 'Prediksi_Sentimen', 'Prediksi_Aspek']].copy()
                df_hasil_tampil.columns = ['Isi Ulasan (Teks Bersih)', 'Hasil Sentimen', 'Kategori Aspek']
                
                # 2. Panggil fungsi mesin pembuat tabel
                tampilkan_tabel_cantik(df_hasil_tampil)
            else:
                st.warning("⚠️ Tidak ada data yang cocok dengan filter yang dipilih.")

            st.caption(f"Menampilkan {len(df_filtered)} ulasan hasil filter.")

    # --- MENU 5: PREDIKSI REAL-TIME ---
    elif menu == "🧪 Prediksi Real-Time":
        st.markdown("<h2 style='text-align: center; font-weight: 900; color: var(--text-color);'>⚡ Live Command Center</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>Sistem Pendukung Keputusan & Evaluator Sentimen Mesin SVM</p>", unsafe_allow_html=True)
        st.markdown("---")

        # ==========================================
        # SUNTIKAN CSS ANIMASI KHUSUS TEKS INFORMASI
        # ==========================================
        st.markdown("""
        <style>
        /* Animasi Goyang (Shake) untuk Peringatan Teks Kosong */
        @keyframes goyang {
            0% { transform: translateX(0); }
            25% { transform: translateX(-5px); }
            50% { transform: translateX(5px); }
            75% { transform: translateX(-5px); }
            100% { transform: translateX(0); }
        }
        .teks-peringatan {
            font-size: 13px;
            color: #9f1239;
            background-color: #ffe4e6;
            padding: 8px 12px;
            border-radius: 6px;
            border-left: 4px solid #e11d48;
            animation: goyang 0.4s ease-in-out;
            margin-bottom: 15px;
            display: inline-block;
            font-weight: 500;
        }
        
        /* Animasi Muncul Halus (Fade Up) untuk Log Sistem */
        @keyframes munculLembut {
            0% { opacity: 0; transform: translateY(15px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        .teks-log {
            font-size: 12px;
            color: #9ca3af;
            text-align: right;
            font-style: italic;
            animation: munculLembut 0.8s ease-out;
            margin-top: 15px;
        }
        </style>
        """, unsafe_allow_html=True)

        # ==========================================
        # 1. INISIALISASI MEMORI TUNGGAL
        # ==========================================
        if 'hasil_terakhir' not in st.session_state:
            st.session_state['hasil_terakhir'] = None

        input_user = st.text_area("✍️ Ketik ulasan atau keluhan pelanggan di sini:", height=100, placeholder="Contoh: Aplikasi mobile ini error terus dari pagi, pelayanannya sangat mengecewakan!")
        
        # Tempat untuk memunculkan animasi peringatan
        tempat_peringatan = st.empty()

        # Tombol Eksekusi & Pembersih Layar
        col_btn1, col_btn2 = st.columns([4, 1])
        with col_btn1:
            klik_analisis = st.button("🚀 Analisis Sentimen & Eksekusi", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🧹 Bersihkan Layar", use_container_width=True):
                st.session_state['hasil_terakhir'] = None
                st.rerun()

        # ==========================================
        # 2. LOGIKA MESIN (JALAN SAAT TOMBOL DITEKAN)
        # ==========================================
        if klik_analisis:
            if input_user:
                import time
                tfidf, svm_sentimen, svm_aspek = load_models()
                
                with st.status("🔍 Menjalankan Bedah Forensik NLP...", expanded=True) as status:
                    st.write("⚙️ Membersihkan teks (Casefolding & Cleansing)...")
                    time.sleep(0.3)
                    st.write("✂️ Menghapus kata hubung (Stopword Removal)...")
                    time.sleep(0.3)
                    st.write("🌱 Mengubah ke kata dasar (Stemming)...")
                    time.sleep(0.3)
                    st.write("🧠 Mengukur bobot kata (TF-IDF) & Prediksi SVM...")
                    time.sleep(0.3)
                    
                    teks_bersih = bersihkan_teks(input_user)
                    X_vektor = tfidf.transform([teks_bersih])
                    pred_sentimen = svm_sentimen.predict(X_vektor)[0]
                    pred_aspek = svm_aspek.predict(X_vektor)[0]
                    status.update(label="✅ Bedah Forensik Selesai!", state="complete", expanded=False)

                # SIMPAN HASIL KE MEMORI
                st.session_state['hasil_terakhir'] = {
                    'teks_asli': input_user,
                    'teks_bersih': teks_bersih,
                    'sentimen': pred_sentimen,
                    'aspek': pred_aspek
                }
                
                # Simpan ke SQLite
                try:
                    import pandas as pd
                    from sqlalchemy import create_engine
                    engine = create_engine('sqlite:///database_sentimen.db')
                    df_baru = pd.DataFrame({'teks_bersih': [teks_bersih], 'Prediksi_Sentimen': [pred_sentimen], 'Prediksi_Aspek': [pred_aspek]})
                    df_baru.to_sql('tabel_klasifikasi', con=engine, if_exists='append', index=False)
                except Exception as e:
                    pass
            else:
                # Memanggil Animasi Goyang (Shake)
                tempat_peringatan.markdown("<div class='teks-peringatan'>⚠️ Wah, kotak teksnya masih kosong! Ketik ulasan dulu ya.</div>", unsafe_allow_html=True)

        # ==========================================
        # 3. RENDER UI TUNGGAL (MENEMPEL PERMANEN)
        # ==========================================
        if st.session_state['hasil_terakhir']:
            data = st.session_state['hasil_terakhir']
            
            st.markdown("---")
            st.markdown("### 📊 Hasil Analisis Terkini")
            
            warna_bg = "#d4edda" if data['sentimen'] == "Positif" else "#f8d7da" if data['sentimen'] == "Negatif" else "#fff3cd"
            warna_teks = "#155724" if data['sentimen'] == "Positif" else "#721c24" if data['sentimen'] == "Negatif" else "#856404"
            ikon = "🟢" if data['sentimen'] == "Positif" else "🔴" if data['sentimen'] == "Negatif" else "🟡"
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div style='background-color: {warna_bg}; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid {warna_teks};'><h4 style='color: {warna_teks}; margin:0;'>{ikon} Sentimen: {data['sentimen']}</h4></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div style='background-color: #e0f2fe; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #0284c7;'><h4 style='color: #0c4a6e; margin:0;'>📌 Aspek Terkait: {data['aspek']}</h4></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # X-RAY KATA
            st.markdown("#### 🩻 X-Ray Kata (Deteksi Pemicu Sentimen)")
            kata_negatif = ['error', 'lambat', 'susah', 'gagal', 'kecewa', 'buruk', 'rugi', 'jelek', 'gabisa', 'ngebug', 'lemot', 'lelet']
            kata_positif = ['cepat', 'bagus', 'mudah', 'lancar', 'puas', 'keren', 'membantu', 'terbaik', 'ramah', 'mantap']
            
            teks_xray = data['teks_asli'].lower()
            for kata in kata_negatif:
                teks_xray = teks_xray.replace(kata, f"<span style='background-color: #fca5a5; color: #991b1b; padding: 2px 6px; border-radius: 4px; font-weight: bold;'>{kata}</span>")
            for kata in kata_positif:
                teks_xray = teks_xray.replace(kata, f"<span style='background-color: #86efac; color: #166534; padding: 2px 6px; border-radius: 4px; font-weight: bold;'>{kata}</span>")
            
            st.markdown(f"<div style='padding: 15px; border: 2px dashed gray; border-radius: 8px; font-size: 16px; background-color: rgba(128,128,128,0.05);'>\"{teks_xray}\"</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # --- BAGIAN C: ACTION PLANNER ---
            st.markdown("#### 🛡️ Sistem Pendukung Keputusan (DSS)")
            
            # NORMALISASI TEKS: Paksa jadi huruf kecil dan buang spasi lebih agar Python tidak bingung
            sentimen_cek = str(data['sentimen']).strip().lower()
            
            col_act1, col_act2 = st.columns(2)
            with col_act1:
                st.info("🚨 **Rekomendasi Eskalasi**")
                if sentimen_cek == "negatif":
                    if data['aspek'] == "Kinerja" or data['aspek'] == "Sistem":
                        st.write("⚠️ Terdeteksi keluhan teknis. Eskalasikan ke **Tim IT / Network Engineer**.")
                    else:
                        st.write(f"⚠️ Terdeteksi keluhan {data['aspek']}. Eskalasikan ke **Manager Operasional**.")
                elif sentimen_cek == "positif":
                    st.write("🌟 Sentimen baik. Arsipkan untuk bahan promosi **Tim Marketing**.")
                else:
                    st.write("👀 Terus pantau perkembangan tren ulasan ini.")

            with col_act2:
                st.success("💬 **Draft Balasan CS**")
                if sentimen_cek == "negatif":
                    draf = f"Mohon maaf atas ketidaknyamanan terkait {data['aspek']}. Keluhan ini telah dicatat dan dieskalasikan ke tim terkait agar segera diperbaiki."
                elif sentimen_cek == "positif":
                    draf = f"Terima kasih atas ulasan positif Bapak/Ibu mengenai {data['aspek']} kami! Kami akan terus mempertahankan kualitas layanan kami."
                else:
                    draf = "Terima kasih atas masukannya. Kami selalu menerima segala bentuk feedback untuk evaluasi ke depannya."
                st.text_area("Salin teks di bawah ini:", value=draf, height=80, label_visibility="collapsed")
            
            
                
                