import re
import streamlit as st
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ==========================================
# INISIALISASI SASTRAWI (DENGAN CACHE RESOURCE)
# ==========================================
# st.cache_resource memastikan Sastrawi hanya diload 1x saja ke RAM Server
@st.cache_resource
def init_sastrawi():
    factory_stemmer = StemmerFactory()
    stem = factory_stemmer.create_stemmer()
    
    factory_stopword = StopWordRemoverFactory()
    stop = factory_stopword.create_stop_word_remover()
    return stem, stop

stemmer, stopword_remover = init_sastrawi()

# ==========================================
# 🚀 HACK 1: KAMUS MEMORI (LRU CACHE)
# ==========================================
# Mesin akan mengingat 10.000 kata unik. Jika kata sudah pernah di-stemming,
# mesin tidak akan memanggil Sastrawi lagi, melainkan mengambil dari ingatan (0.001 detik).
@st.cache_data(max_entries=10000)
def stem_satu_kata(kata):
    return stemmer.stem(kata)

# ==========================================
# 📘 KAMUS ALAY SUPER LENGKAP (VERSI FINAL SKRIPSI)
# ==========================================
kamus_alay = {
    # --- ISTILAH BANK & APLIKASI ---
    "mbanking": "mobile banking",
    "mbca": "mobile banking bca",
    "apk": "aplikasi",
    "app": "aplikasi",
    "apps": "aplikasi",
    "appnya": "aplikasi",
    "uinya": "ui",
    "min": "admin",
    "minn": "admin",
    "cs": "customer service",
    "rek": "rekening",
    "tf": "transfer",
    "trf": "transfer",
    "notif": "notifikasi",
    "keblokir": "terblokir",
    
    # --- KATA NEGASI (SANGAT PENTING UNTUK SENTIMEN NEGATIF) ---
    "ga": "tidak",
    "gak": "tidak",
    "gk": "tidak",
    "ngga": "tidak",
    "nggak": "tidak",
    "engga": "tidak",
    "kaga": "tidak",
    "tdk": "tidak",
    "gabisa": "tidak bisa",
    "gaada": "tidak ada",
    "gatau": "tidak tahu",
    "gapernah": "tidak pernah",

    # --- KATA KETERANGAN WAKTU & SIFAT ---
    "skrg": "sekarang",
    "skrng": "sekarang",
    "kemaren": "kemarin",
    "malem": "malam",
    "cepet": "cepat",
    "satset": "cepat",
    "sat": "cepat",
    "set": "cepat",
    "lemot": "lambat",
    "ribet": "rumit",
    "bener": "benar",
    "ijo": "hijau",

    # --- KATA KERJA TIDAK BAKU ---
    "pake": "pakai",
    "make": "pakai",
    "pke": "pakai",
    "dipake": "dipakai",
    "nyoba": "coba",
    "bikin": "buat",
    "liat": "lihat",
    "nanya": "tanya",
    "kelar": "selesai",
    "pengen": "ingin",
    "males": "malas",
    "digunain": "digunakan",
    "aktifin": "aktifkan",
    "benerin": "perbaiki",
    "daftarin": "daftarkan",
    "matiin": "matikan",
    "muter": "putar",
    "enakan": "lebih enak",

    # --- SINGKATAN & BAHASA CHAT ---
    "yg": "yang",
    "kalo": "kalau",
    "klo": "kalau",
    "kl": "kalau",
    "aja": "saja",
    "doang": "saja",
    "cuman": "cuma",
    "cmn": "cuma",
    "cm": "cuma",
    "udah": "sudah",
    "udh": "sudah",
    "dah": "sudah",
    "uda": "sudah",
    "sdh": "sudah",
    "blm": "belum",
    "bgt": "banget",
    "gimana": "bagaimana",
    "gmn": "bagaimana",
    "gw": "saya",
    "gue": "saya",
    "gua": "saya",
    "sy": "saya",
    "ak": "aku",
    "ku": "aku",
    "lu": "kamu",
    "tp": "tapi",
    "tpi": "tapi",
    "mulu": "terus",
    "trs": "terus",
    "trus": "terus",
    "sampe": "sampai",
    "emang": "memang",
    "jg": "juga",
    "gitu": "begitu",
    "gt": "begitu",
    "gini": "begini",
    "lg": "lagi",
    "jd": "jadi",
    "jdi": "jadi",
    "tetep": "tetap",
    "krn": "karena",
    "karna": "karena",
    "dgn": "dengan",
    "dr": "dari",
    "tau": "tahu",
    "kaya": "seperti",
    "kek": "seperti",
    "kayak": "seperti",
    "kyk": "seperti",
    "biar": "supaya",
    "ni": "ini",
    "knp": "kenapa",
    "bs": "bisa",
    "yah": "ya",
    "yaa": "ya",
    "yaaa": "ya",
    "yak": "ya",
    "utk": "untuk",
    "dapet": "dapat",
    "sm": "sama",
    "lbh": "lebih",
    "ko": "kok",
    "pdhl": "padahal",
    "msh": "masih",
    "masi": "masih",
    "hrs": "harus",
    "eror": "error",
    
    # --- VARIASI HURUF & LAINNYA ---
    "hallo": "halo",
    "ka": "kak",
    "org": "orang",
    "tu": "itu",
    "hape": "hp",
    "nomer": "nomor",
    "telpon": "telepon",
    "telp": "telepon",
    "rb": "ribu",
    "jt": "juta",
    "wkwk": "tertawa",
    "wkwkwk": "tertawa",
    "wkwkw": "tertawa",
    "anjir": "astaga",
    "njir": "astaga"
}

# ==========================================
# FUNGSI UTAMA PEMBERSIHAN TEKS
# ==========================================
def bersihkan_teks(teks):
    # 1. Jadikan huruf kecil semua (Lowercase)
    teks = str(teks).lower()
    
    # 2. Hapus noise (Mention @, URL, Hashtag #, Angka, dan Tanda Baca)
    teks = re.sub(r'http\S+', '', teks)
    teks = re.sub(r'@[a-zA-Z0-9_]+', '', teks)
    teks = re.sub(r'#\w+', '', teks)
    teks = re.sub(r'[^a-zA-Z\s]', ' ', teks) # Hanya sisakan huruf alphabet dan spasi
    
    # 3. 💥 NORMALISASI KAMUS ALAY
    kata_kata = teks.split()
    kata_normal = [kamus_alay.get(k, k) for k in kata_kata]
    teks = ' '.join(kata_normal)
    
    # 4. Hapus Stopword
    teks = stopword_remover.remove(teks)
    
    # 5. 🚀 STEMMING KECEPATAN TINGGI (Menggunakan fungsi Cache)
    kata_kata_bersih = teks.split()
    # Panggil fungsi yang sudah punya ingatan (cache)
    hasil_stem = [stem_satu_kata(k) for k in kata_kata_bersih]
    teks_final = ' '.join(hasil_stem)
    
    return teks_final