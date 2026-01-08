# 📊 Digital Marketing Analytics Dashboard

Dashboard ini digunakan untuk menganalisis kinerja kampanye *digital marketing* berdasarkan metrik utama seperti *Click Through Rate (CTR)*, *Cost Per Click (CPC)*, *Conversion Rate*, dan *Return on Investment (ROI)*. Aplikasi dibangun menggunakan *Python* dan *Streamlit* serta mendukung unggah dataset CSV.

---

## 1. Deskripsi Proyek

Proyek ini bertujuan untuk:
- Mengolah data kampanye digital marketing secara otomatis
- Menghitung metrik kinerja utama (CTR, CPC, CR, ROI)
- Menyajikan hasil analisis dalam bentuk tabel dan visualisasi grafik
- Membantu evaluasi dan pengambilan keputusan berbasis data

Dashboard bersifat *reusable* dan dapat digunakan dengan dataset lain selama struktur kolom sesuai.

---

## 2. Teknologi yang Digunakan

- *Python* 3.9+
- *Streamlit*
- *Pandas*

---

## 3. Struktur Folder Proyek

```
Digital_Marketing_Dashboard/
│
├── app.py
├── requirements.txt
├── digital_marketing_ads_2000.csv
├── README.md
```

---

## 4. Instalasi dan Persiapan Lingkungan

### 4.1 Clone Repository
```
git clone https://github.com/username/Digital_Marketing_Dashboard.git
cd Digital_Marketing_Dashboard
```

### 4.2 (Opsional) Virtual Environment
```
python -m venv venv
source venv/bin/activate
```

### 4.3 Install Dependency
```
pip install -r requirements.txt
```

---

## 5. Menjalankan Aplikasi

```
streamlit run app.py
```

Akses di browser:
```
http://localhost:8501
```

---

## 6. Format Dataset CSV

Kolom wajib:
- date
- campaign_name
- impressions
- clicks
- cost
- conversions
- revenue

---

## 7. Cara Upload Dataset

1. Jalankan aplikasi
2. Klik Browse files
3. Pilih file CSV
4. Dashboard akan menampilkan hasil otomatis

---

## 8. Hasil Program dan Visualisasi

### 8.1 KPI Agregat
<img width="1266" height="150" alt="image" src="https://github.com/user-attachments/assets/2e56cc1f-a58b-458a-a800-ecd23e853ce1" />


### 8.2 Tabel Performa Campaign
<img width="1266" height="255" alt="image" src="https://github.com/user-attachments/assets/1461b4bf-0254-4622-9998-704389dc0daa" />


### 8.3 Visualisasi CTR
<img width="630" height="387" alt="image" src="https://github.com/user-attachments/assets/92aae825-a1ef-4ba6-993f-5dce02988756" />


### 8.4 Visualisasi ROI
<img width="674" height="364" alt="image" src="https://github.com/user-attachments/assets/3c78c377-79db-429b-9bc1-c234c3f2bb2b" />


---

## 9. Dokumentasi Tambahan

- Link dashboard Streamlit : https://dashboarddigmar.streamlit.app/
- Source code GitHub : https://github.com/Muhjat7/streamlit_text_classification


---

## 10. Catatan Akhir

Dashboard ini dapat dikembangkan dengan fitur filter, segmentasi campaign, dan integrasi data real-time.
