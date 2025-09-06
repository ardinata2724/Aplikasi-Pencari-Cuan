import streamlit as st
import pandas as pd
import os
import time
import random
import numpy as np
from collections import defaultdict, Counter
from itertools import product
from datetime import datetime, timedelta
import json
import uuid
import traceback
import tensorflow as tf
import gc

# ==============================================================================
# KONFIGURASI & FUNGSI PASSWORD
# ==============================================================================

ADMIN_PASSWORD = "Andi1991"
SESSION_TIMEOUT_MINUTES = 15
PASSWORDS_FILE = "passwords.json"
DEVICE_LOG_FILE = "device_log.json"

def get_valid_passwords():
    if not os.path.exists(PASSWORDS_FILE): return []
    try:
        with open(PASSWORDS_FILE, 'r') as f:
            content = f.read()
            if not content.strip(): return []
            return json.loads(content)
    except json.JSONDecodeError:
        st.error(f"Peringatan: File '{PASSWORDS_FILE}' tidak dapat dibaca (kemungkinan format salah atau rusak).")
        return []

def get_device_log():
    if not os.path.exists(DEVICE_LOG_FILE): return {}
    try:
        with open(DEVICE_LOG_FILE, 'r') as f: return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError): return {}

def save_device_log(log_data):
    with open(DEVICE_LOG_FILE, 'w') as f: json.dump(log_data, f, indent=4)

def force_logout():
    device_log = get_device_log()
    password_to_logout = None
    for pwd, sid in device_log.items():
        if sid == st.session_state.get('user_session_id'):
            password_to_logout = pwd
            break
    if password_to_logout and password_to_logout in device_log:
        del device_log[password_to_logout]
        save_device_log(device_log)
    for key in ['logged_in', 'is_admin', 'user_session_id', 'last_activity_time']:
        if key in st.session_state: del st.session_state[key]

def check_password_per_device():
    for key, default in [('logged_in', False), ('is_admin', False), ('user_session_id', None), ('last_activity_time', None)]:
        if key not in st.session_state: st.session_state[key] = default
    if st.session_state.logged_in and st.session_state.last_activity_time:
        if datetime.now() - st.session_state.last_activity_time > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
            force_logout()
            st.warning(f"Anda telah otomatis logout karena tidak aktif selama {SESSION_TIMEOUT_MINUTES} menit.")
            time.sleep(2); st.rerun()
    if st.session_state.logged_in:
        device_log = get_device_log()
        if any(sid == st.session_state.user_session_id for sid in device_log.values()):
            return True
        else:
            force_logout(); st.warning("Sesi Anda tidak valid. Silakan login kembali."); st.rerun()
    st.title("🔐 Login Aplikasi")
    with st.form(key="login_form"):
        password = st.text_input("Masukkan Password Anda", type="password")
        login_button = st.form_submit_button("Login")
    if login_button:
        valid_passwords = get_valid_passwords()
        device_log = get_device_log()
        if password in valid_passwords:
            session_id = device_log.get(password, str(uuid.uuid4()))
            device_log[password] = session_id
            save_device_log(device_log)
            st.session_state.user_session_id = session_id
            st.session_state.logged_in = True
            st.session_state.is_admin = (password == ADMIN_PASSWORD)
            st.rerun()
        else:
            st.error("😕 Password salah atau tidak terdaftar.")
            st.session_state.logged_in = False
    return False

# ==============================================================================
# BAGIAN 1: FUNGSI-FUNGSI INTI
# ==============================================================================

class PositionalEncoding(tf.keras.layers.Layer):
    def call(self, x):
        seq_len, d_model = tf.shape(x)[1], tf.shape(x)[2]
        pos = tf.cast(tf.range(seq_len)[:, tf.newaxis], dtype=tf.float32)
        i = tf.cast(tf.range(d_model)[tf.newaxis, :], dtype=tf.float32)
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = pos * angle_rates
        sines, cosines = tf.math.sin(angle_rads[:, 0::2]), tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        return x + tf.cast(tf.expand_dims(pos_encoding, 0), tf.float32)

DIGIT_LABELS, BBFS_LABELS, JUMLAH_LABELS, SHIO_LABELS, JALUR_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"], ["bbfs_ribuan-ratusan", "bbfs_ratusan-puluhan", "bbfs_puluhan-satuan"], ["jumlah_depan", "jumlah_tengah", "jumlah_belakang"], ["shio_depan", "shio_tengah", "shio_belakang"], ["jalur_ribuan-ratusan", "jalur_ratusan-puluhan", "jalur_puluhan-satuan"]
JALUR_ANGKA_MAP = {1: "01*13*25*37*49*61*73*85*97*04*16*28*40*52*64*76*88*00*07*19*31*43*55*67*79*91*10*22*34*46*58*70*82*94", 2: "02*14*26*38*50*62*74*86*98*05*17*29*41*53*65*77*89*08*20*32*44*56*68*80*92*11*23*35*47*59*71*83*95", 3: "03*15*27*39*51*63*75*87*99*06*18*30*42*54*66*78*90*09*21*33*45*57*69*81*93*12*24*36*48*60*72*84*96"}

@st.cache_resource
def load_cached_model(model_path):
    from tensorflow.keras.models import load_model
    if os.path.exists(model_path):
        try: return load_model(model_path, custom_objects={"PositionalEncoding": PositionalEncoding})
        except Exception as e: st.error(f"Gagal memuat model di {model_path}: {e}")
    return None

def top6_markov(df, top_n=6):
    if df.empty or len(df) < 10: return [], None
    data = df["angka"].astype(str).tolist(); matrix = [defaultdict(lambda: defaultdict(int)) for _ in range(3)]
    for number in data:
        digits = f"{int(number):04d}"; [matrix[i][digits[i]][digits[i+1]] for i in range(3)]
    freq_ribuan = Counter(int(x[0]) for x in data); hasil = [[k for k, _ in freq_ribuan.most_common(top_n)]]
    for i in range(3):
        kandidat = [int(k) for prev in matrix[i] for k in matrix[i][prev].keys()]; top = [k for k, _ in Counter(kandidat).most_common()]; hasil.append(top)
    return [list(dict.fromkeys(h))[:top_n] for h in hasil], None

def calculate_angka_main_stats(df, top_n=5):
    if df.empty or len(df) < 10: return {"jumlah_2d": "Data tidak cukup", "colok_bebas": "Data tidak cukup"}
    angka_str = df["angka"].astype(str).str.zfill(4); puluhan = angka_str.str[2].astype(int); satuan = angka_str.str[3].astype(int); jumlah = (puluhan + satuan) % 10
    jumlah_2d = ", ".join(map(str, jumlah.value_counts().nlargest(top_n).index)); colok_bebas = ", ".join(c for c, _ in Counter("".join(angka_str)).most_common(top_n))
    return {"jumlah_2d": jumlah_2d, "colok_bebas": colok_bebas}

def calculate_markov_ai(df, top_n=6, mode='belakang'):
    if df.empty or len(df) < 10: return "Data tidak cukup untuk analisis."
    mode_to_idx = {'depan': 3, 'tengah': 1, 'belakang': 0}; start_idx = mode_to_idx[mode]; angka_str_list = df["angka"].astype(str).str.zfill(4).tolist(); transitions = defaultdict(list)
    for num_str in angka_str_list: transitions[num_str[start_idx]].extend(d for i, d in enumerate(num_str) if i != start_idx)
    prediction_map = {}
    for start_digit, following_digits in transitions.items():
        final_digits = list(dict.fromkeys(d for d, _ in Counter(following_digits).most_common()))
        if len(final_digits) < top_n:
            all_digits = list(map(str, range(10))); random.shuffle(all_digits)
            final_digits.extend(d for d in all_digits if d not in final_digits and len(final_digits) < top_n)
        prediction_map[start_digit] = "".join(final_digits[:top_n])
    return "\n".join(f"{num_str} = {prediction_map.get(num_str[start_idx], '')} ai" for num_str in angka_str_list[-30:])

def tf_preprocess_data(df, window_size=7):
    from tensorflow.keras.utils import to_categorical
    if len(df) < window_size + 1: return np.array([]), {}
    angka = df["angka"].values; labels_to_process = DIGIT_LABELS + BBFS_LABELS + JUMLAH_LABELS + SHIO_LABELS; sequences, targets = [], {label: [] for label in labels_to_process}
    for i in range(len(angka) - window_size):
        window = [str(x).zfill(4) for x in angka[i:i+window_size+1]]
        if any(not x.isdigit() for x in window): continue
        sequences.append([int(d) for num in window[:-1] for d in num]); target_digits = [int(d) for d in window[-1]]
        for j, label in enumerate(DIGIT_LABELS): targets[label].append(to_categorical(target_digits[j], num_classes=10))
        for label, val in {"jumlah_depan": (target_digits[0] + target_digits[1]) % 10, "jumlah_tengah": (target_digits[1] + target_digits[2]) % 10, "jumlah_belakang": (target_digits[2] + target_digits[3]) % 10}.items(): targets[label].append(to_categorical(val, num_classes=10))
        for label, pair in {"bbfs_ribuan-ratusan": [target_digits[0], target_digits[1]], "bbfs_ratusan-puluhan": [target_digits[1], target_digits[2]], "bbfs_puluhan-satuan": [target_digits[2], target_digits[3]]}.items():
            multi_hot = np.zeros(10, dtype=np.float32); multi_hot[np.unique(pair)] = 1.0; targets[label].append(multi_hot)
        for label, num in {"shio_depan": target_digits[0]*10 + target_digits[1], "shio_tengah": target_digits[1]*10 + target_digits[2], "shio_belakang": target_digits[2]*10 + target_digits[3]}.items():
            shio_idx = (num - 1) % 12 if num > 0 else 11; targets[label].append(to_categorical(shio_idx, num_classes=12))
    return np.array(sequences), {label: np.array(v) for label, v in targets.items() if v}

def tf_preprocess_data_for_jalur(df, window_size, target_position):
    from tensorflow.keras.utils import to_categorical
    if len(df) < window_size + 1: return np.array([]), np.array([])
    jalur_map = {1: [1, 4, 7, 10], 2: [2, 5, 8, 11], 3: [3, 6, 9, 12]}; shio_to_jalur = {shio: jalur for jalur, shios in jalur_map.items() for shio in shios}
    pos_map = {'ribuan-ratusan': (0, 1), 'ratusan-puluhan': (1, 2), 'puluhan-satuan': (2, 3)}; idx1, idx2 = pos_map[target_position]
    angka = df["angka"].values; sequences, targets = [], []
    for i in range(len(angka) - window_size):
        window = [str(x).zfill(4) for x in angka[i:i+window_size+1]]
        if any(not x.isdigit() for x in window): continue
        sequences.append([int(d) for num in window[:-1] for d in num]); target_digits = [int(d) for d in window[-1]]
        num = target_digits[idx1] * 10 + target_digits[idx2]; shio = (num - 1) % 12 + 1 if num > 0 else 12
        targets.append(to_categorical(shio_to_jalur[shio] - 1, num_classes=3))
    return np.array(sequences), np.array(targets)

# --- PERUBAHAN BARU DI SINI ---
def build_tf_model(input_len, model_type, problem_type, num_classes):
    from tensorflow.keras.models import Model; from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
    inputs = Input(shape=(input_len,)); x = Embedding(10, 64)(inputs); x = PositionalEncoding()(x)
    if model_type == "transformer":
        attn = MultiHeadAttention(num_heads=4, key_dim=64)(x, x); x = LayerNormalization()(x + attn)
    else: 
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        # Dropout(0.3) dihapus
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    # Dropout(0.2) dihapus
    outputs, loss = (Dense(num_classes, activation='sigmoid')(x), "binary_crossentropy") if problem_type == "multilabel" else (Dense(num_classes, activation='softmax')(x), "categorical_crossentropy")
    return Model(inputs, outputs), loss
# --- AKHIR PERUBAHAN BARU ---

def top_n_model(df, lokasi, window_dict, model_type, top_n):
    results = []; loc_id = lokasi.lower().strip().replace(" ", "_")
    for label in DIGIT_LABELS:
        tf.keras.backend.clear_session(); gc.collect()
        ws = window_dict.get(label, 7); X, _ = tf_preprocess_data(df, ws)
        if X.shape[0] == 0: return None, None
        model_path = f"saved_models/{loc_id}_{label}_{model_type}.h5"; model = load_cached_model(model_path)
        if model is None: st.error(f"Model {label} tidak ditemukan."); return None, None
        pred = model.predict(X, verbose=0); results.append(list(np.mean(pred, axis=0).argsort()[-top_n:][::-1]))
    return results, None

def _train_single_model_for_ws(X_train, y_train, X_val, y_val, model_params):
    tf.keras.backend.clear_session(); gc.collect()
    from tensorflow.keras.callbacks import EarlyStopping; from tensorflow.keras.metrics import TopKCategoricalAccuracy
    input_len, model_type, problem_type, num_classes, k = model_params['input_len'], model_params['model_type'], model_params['problem_type'], model_params['num_classes'], model_params['k']
    model, loss = build_tf_model(input_len, model_type, problem_type, num_classes)
    metrics = ['accuracy']
    if problem_type != 'multilabel': metrics.append(TopKCategoricalAccuracy(k=k))
    model.compile(optimizer="adam", loss=loss, metrics=metrics)
    model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val), callbacks=[EarlyStopping(monitor='val_loss', patience=3)], verbose=0)
    return model.evaluate(X_val, y_val, verbose=0), model.predict(X_val, verbose=0)

def find_best_window_size(df, label, model_type, min_ws, max_ws, top_n, top_n_shio):
    from sklearn.model_selection import train_test_split
    best_ws, best_score, table_data = None, -1, []
    is_jalur_scan = label in JALUR_LABELS
    
    if is_jalur_scan:
        pt, k, nc = "jalur_multiclass", 2, 3
        cols = ["Window Size", "Prediksi", "Angka Jalur"]
    elif label in BBFS_LABELS:
        pt, k, nc = "multilabel", top_n, 10
        cols = ["Window Size", f"Top-{k}", "Sisa Angka"]
    elif label in SHIO_LABELS:
        pt, k, nc = "shio", top_n_shio, 12
        cols = ["Window Size", f"Top-{k}", "Sisa Angka"]
    else:
        pt, k, nc = "multiclass", top_n, 10
        cols = ["Window Size", f"Top-{k}", "Sisa Angka"]

    bar = st.progress(0, text=f"Memulai Scan {label.upper()}... [0%]"); total_ws = (max_ws - min_ws) + 1
    for i, ws in enumerate(range(min_ws, max_ws + 1)):
        bar.progress((i + 1) / total_ws, text=f"Mencoba WS={ws}... [{int(((i + 1) / total_ws) * 100)}%]")
        try:
            if is_jalur_scan:
                X, y = tf_preprocess_data_for_jalur(df, ws, label.split('_')[1])
                if not y.any() or y.shape[0] < 10 or X.shape[0] < 10: continue
            else:
                X, y_dict = tf_preprocess_data(df, ws)
                if label not in y_dict or y_dict[label].shape[0] < 10 or X.shape[0] < 10: continue
                y = y_dict[label]
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            model_params = {'input_len': X.shape[1], 'model_type': model_type, 'problem_type': 'multiclass' if is_jalur_scan else pt, 'num_classes': nc, 'k': k}
            evals, preds = _train_single_model_for_ws(X_train, y_train, X_val, y_val, model_params)
            
            if is_jalur_scan:
                top_indices = np.argsort(preds[-1])[::-1][:2]
                pred_str = f"{top_indices[0] + 1}-{top_indices[1] + 1}"
                angka_jalur_str = f"Jalur {top_indices[0] + 1} => {JALUR_ANGKA_MAP[top_indices[0] + 1]}\n\nJalur {top_indices[1] + 1} => {JALUR_ANGKA_MAP[top_indices[1] + 1]}"
                score = (evals[1] * 0.3) + (evals[2] * 0.7)
                table_data.append((ws, pred_str, angka_jalur_str))
            else:
                avg_conf = np.mean(np.sort(preds, axis=1)[:, -k:]) * 100
                top_indices = np.argsort(preds[-1])[::-1][:k]
                
                sisa_angka_str = ""
                if pt == "shio":
                    pred_str = ", ".join(map(str, top_indices + 1))
                    all_items = set(range(1, 13))
                    predicted_items = set(top_indices + 1)
                    missing_items = sorted(list(all_items - predicted_items))
                    sisa_angka_str = ", ".join(map(str, missing_items))
                else:
                    pred_str = ", ".join(map(str, top_indices))
                    all_items = set(range(10))
                    predicted_items = set(top_indices)
                    missing_items = sorted(list(all_items - predicted_items))
                    sisa_angka_str = ", ".join(map(str, missing_items))
                
                score = (evals[1] * 0.7) + (avg_conf / 100 * 0.3) if pt == 'multilabel' else (evals[1] * 0.2) + (evals[2] * 0.5) + (avg_conf / 100 * 0.3)
                table_data.append((ws, pred_str, sisa_angka_str))

            if score > best_score: best_score, best_ws = score, ws
        except Exception as e:
            st.warning(f"Gagal di WS={ws}: {e}"); st.code(traceback.format_exc()); continue
    bar.empty()
    return best_ws, pd.DataFrame(table_data, columns=cols) if table_data else pd.DataFrame()


def train_and_save_model(df, lokasi, window_dict, model_type):
    from sklearn.model_selection import train_test_split; from tensorflow.keras.callbacks import EarlyStopping
    st.info(f"Memulai pelatihan untuk {lokasi}..."); lokasi_id = lokasi.lower().strip().replace(" ", "_")
    if not os.path.exists("saved_models"): os.makedirs("saved_models")
    for label in DIGIT_LABELS:
        tf.keras.backend.clear_session(); gc.collect()
        ws = window_dict.get(label, 7); bar = st.progress(0, text=f"Memproses {label.upper()} (WS={ws})..."); X, y_dict = tf_preprocess_data(df, ws)
        if label not in y_dict or y_dict[label].shape[0] < 10: st.warning(f"Data tidak cukup untuk melatih '{label.upper()}'."); bar.empty(); continue
        X_train, X_val, y_train, y_val = train_test_split(X, y_dict[label], test_size=0.2, random_state=42); bar.progress(50, text=f"Melatih {label.upper()}...")
        model, loss = build_tf_model(X.shape[1], model_type, 'multiclass', 10); model.compile(optimizer='adam', loss=loss, metrics=['accuracy']); model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), callbacks=[EarlyStopping(monitor='val_loss', patience=5)], verbose=0)
        model_path = f"saved_models/{lokasi_id}_{label}_{model_type}.h5"; bar.progress(75, text=f"Menyimpan {label.upper()}..."); model.save(model_path); bar.progress(100, text=f"Model {label.upper()} berhasil disimpan!"); time.sleep(1); bar.empty()

# ==============================================================================
# APLIKASI STREAMLIT UTAMA
# ==============================================================================
st.set_page_config(page_title="Prediksi 4D", layout="wide")
if check_password_per_device():
    st.session_state.last_activity_time = datetime.now()
    for key, default in [('angka_list', []), ('scan_outputs', {}), ('scan_queue', []), ('current_scan_job', None)]:
        if key not in st.session_state: st.session_state[key] = default
    st.title("Prediksi 4D"); st.caption("editing by: Andi Prediction")
    try: from lokasi_list import lokasi_list
    except ImportError: lokasi_list = ["BULLSEYE", "HONGKONGPOOLS", "HONGKONG LOTTO", "SYDNEYPOOLS", "SYDNEY LOTTO", "SINGAPURA"]
    with st.sidebar:
        st.header("⚙️ Pengaturan"); selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list); putaran = st.number_input("🔁 Jumlah Putaran Terakhir", 10, 1000, 100)
        st.markdown("---"); st.markdown("### 🎯 Opsi Prediksi"); jumlah_digit = st.slider("🔢 Jumlah Digit Prediksi", 1, 9, 9); jumlah_digit_shio = st.slider("🐉 Jumlah Digit Prediksi Khusus Shio", 1, 12, 12)
        metode = st.selectbox("🧠 Metode", ["Markov", "LSTM AI"]); use_transformer = st.checkbox("🤖 Gunakan Transformer", value=True); model_type = "transformer" if use_transformer else "lstm"
        st.markdown("---"); st.markdown("### 🪟 Window Size per Digit"); window_per_digit = {label: st.number_input(f"{label.upper()}", 1, 100, 7, key=f"win_{label}") for label in DIGIT_LABELS}
        st.markdown("---")
        if st.button("🚪 Logout"): force_logout(); st.success("Anda berhasil logout."); time.sleep(1); st.rerun()

    def get_file_name_from_lokasi(lokasi):
        cleaned_lokasi = lokasi.lower().replace(" ", "")
        if "hongkonglotto" in cleaned_lokasi: return "keluaran hongkong lotto.txt"
        if "hongkongpools" in cleaned_lokasi: return "keluaran hongkongpools.txt"
        if "sydneylotto" in cleaned_lokasi: return "keluaran sydney lotto.txt"
        if "sydneypools" in cleaned_lokasi: return "keluaran sydneypools.txt"
        return f"keluaran {lokasi.lower()}.txt"

    if st.button("Ambil Data dari Keluaran Angka", use_container_width=True):
        folder_data = "data_keluaran"; file_path = os.path.join(folder_data, get_file_name_from_lokasi(selected_lokasi))
        try:
            with open(file_path, 'r') as f: lines = f.readlines()
            angka_from_file = [line.strip()[:4] for line in lines[-putaran:] if line.strip() and line.strip()[:4].isdigit()]
            if angka_from_file: st.session_state.angka_list = angka_from_file; st.success(f"{len(angka_from_file)} data berhasil diambil dari '{file_path}'.")
        except FileNotFoundError: st.error(f"File tidak ditemukan: '{file_path}'. Pastikan file ada di dalam folder '{folder_data}'.")

    with st.expander("✏️ Edit Data Angka Manual", expanded=True):
        riwayat_text = st.text_area("1 angka per baris:", "\n".join(st.session_state.angka_list), height=300, key="manual_data_input")
        if riwayat_text != "\n".join(st.session_state.angka_list):
            processed_angka = []
            for line in riwayat_text.splitlines():
                if not line.strip():
                    continue
                
                parts = line.strip().split()
                last_part = parts[-1]
                
                if len(last_part) == 4 and last_part.isdigit():
                    processed_angka.append(last_part)
            
            st.session_state.angka_list = processed_angka
            st.rerun()

    df = pd.DataFrame({"angka": st.session_state.get("angka_list", [])})
    tab_list = ["🪟 Scan Window Size", "⚙️ Manajemen Model", "🎯 Angka Main", "🔮 Prediksi & Hasil"]
    if st.session_state.is_admin: tab_list.append("👑 Admin")
    tabs = st.tabs(tab_list)
    
    with tabs[3]: # Prediksi & Hasil
        if st.button("🚀 Jalankan Prediksi", use_container_width=True, type="primary"):
            if not df.empty and len(df) >= max(window_per_digit.values()) + 1:
                result, _ = None, None
                if metode == "Markov": result, _ = top6_markov(df, jumlah_digit)
                elif metode == "LSTM AI": result, _ = top_n_model(df, selected_lokasi, window_per_digit, model_type, jumlah_digit)
                if result:
                    st.subheader(f"🎯 Hasil Prediksi Top {jumlah_digit}"); [st.markdown(f"**{label.upper()}:** {', '.join(map(str, res))}") for label, res in zip(DIGIT_LABELS, result)]
                    st.divider(); all_combinations = list(product(*result)); st.subheader(f"🔢 Semua Kombinasi 4D ({len(all_combinations)} Line)")
                    st.text_area("Kombinasi Penuh", " * ".join("".join(map(str, c)) for c in all_combinations), height=300)
            else: st.warning("❌ Data tidak cukup untuk prediksi.")

    with tabs[1]: # Manajemen Model
        st.subheader("Manajemen Model AI"); lokasi_id = selected_lokasi.lower().strip().replace(" ", "_"); cols = st.columns(4)
        for i, label in enumerate(DIGIT_LABELS):
            with cols[i]:
                model_path = f"saved_models/{lokasi_id}_{label}_{model_type}.h5"; st.markdown(f"##### {label.upper()}")
                if os.path.exists(model_path):
                    st.success("✅ Tersedia"); 
                    if st.button("Hapus", key=f"hapus_{label}", use_container_width=True): os.remove(model_path); st.rerun()
                else: st.warning("⚠️ Belum ada")
        if st.button("📚 Latih & Simpan Semua Model AI", use_container_width=True, type="primary"):
            if len(df) >= max(window_per_digit.values()) + 10: train_and_save_model(df, selected_lokasi, window_per_digit, model_type); st.success("✅ Semua model berhasil dilatih!"); st.rerun()
            else: st.error("Data tidak cukup untuk melatih.")

    with tabs[0]: # Scan Window Size
        st.subheader("Pencarian Window Size (WS) Optimal per Kategori"); scan_cols = st.columns(2)
        min_ws = scan_cols[0].number_input("Min WS", 3, 99, 5) 
        max_ws = scan_cols[1].number_input("Max WS", min_ws + 1, 100, 31)
        if st.button("❌ Hapus Hasil Scan"): st.session_state.scan_outputs = {}; st.rerun()
        st.divider()
        def create_scan_button(label, container):
            is_pending = label in st.session_state.scan_queue or st.session_state.current_scan_job == label
            if container.button(f"🔎 Scan {label.replace('_', ' ').upper()}", key=f"scan_{label}", use_container_width=True, disabled=is_pending):
                st.session_state.scan_queue.append(label); st.toast(f"✅ Scan untuk '{label.upper()}' ditambahkan ke antrian."); st.rerun()
        category_tabs = st.tabs(["Digit", "Jumlah", "BBFS", "Shio", "Jalur Main"])
        scan_labels = [DIGIT_LABELS, JUMLAH_LABELS, BBFS_LABELS, SHIO_LABELS, JALUR_LABELS]
        for i, tab in enumerate(category_tabs):
            with tab:
                cols = st.columns(len(scan_labels[i]))
                for label, col in zip(scan_labels[i], cols):
                    create_scan_button(label, col)
        st.divider()
        if st.session_state.scan_outputs:
            st.markdown("---"); st.subheader("✅ Hasil Scan Selesai")
            for label in [l for cat in scan_labels for l in cat]:
                if label in st.session_state.scan_outputs:
                    data = st.session_state.scan_outputs[label]
                    with st.expander(f"Hasil untuk {label.replace('_', ' ').upper()}", expanded=True):
                        df_result = data.get("table")
                        if df_result is not None and not df_result.empty:
                            st.dataframe(df_result)
                            if data["ws"] is not None: st.info(f"💡 **WS terbaik yang ditemukan: {data['ws']}**")
                        else: st.warning("Tidak ada hasil yang valid untuk rentang WS ini.")
        if st.session_state.scan_queue:
            scan_items = [f"**{job.replace('_', ' ').upper()}**" for job in st.session_state.scan_queue]
            queue_text = ' ➡️ '.join(scan_items)
            st.info(f"Antrian Berikutnya: {queue_text}")
        if not st.session_state.current_scan_job and st.session_state.scan_queue:
            st.session_state.current_scan_job = st.session_state.scan_queue.pop(0); st.rerun()
        if st.session_state.current_scan_job:
            label = st.session_state.current_scan_job
            if len(df) < max_ws + 10:
                st.error(f"Data tidak cukup untuk scan {label.upper()}. Tugas dibatalkan.")
                st.session_state.current_scan_job = None; time.sleep(2); st.rerun()
            else:
                st.warning(f"⏳ Sedang menjalankan scan untuk **{label.replace('_', ' ').upper()}**...")
                best_ws, result_table = find_best_window_size(df, label, model_type, min_ws, max_ws, jumlah_digit, jumlah_digit_shio)
                st.session_state.scan_outputs[label] = {"ws": best_ws, "table": result_table}; st.session_state.current_scan_job = None; st.rerun()

    with tabs[2]: # Angka Main
        st.subheader("Analisis Angka Main dari Data Historis")
        if not df.empty and len(df) >= 10:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("##### Analisis AI Berdasarkan Posisi")
                for mode in ['depan', 'tengah', 'belakang']:
                    posisi = {'depan': 'EKOR', 'tengah': 'AS', 'belakang': 'KOP'}.get(mode)
                    title = f"Analisis AI {mode.capitalize()} (berdasarkan digit {posisi})"
                    with st.expander(title, expanded=(mode=='depan')):
                        st.text_area(f"Hasil Analisis ({mode.capitalize()})", calculate_markov_ai(df, jumlah_digit, mode), height=300, label_visibility="collapsed", key=f"ai_{mode}")
            with col2:
                st.markdown("##### Statistik Lainnya"); stats = calculate_angka_main_stats(df, jumlah_digit)
                st.markdown(f"**Jumlah 2D (Belakang):**"); st.code(stats['jumlah_2d'])
                st.markdown(f"**Colok Bebas:**"); st.code(stats['colok_bebas'])
        else: st.warning("Data historis tidak cukup (minimal 10 baris).")

    if st.session_state.is_admin:
        with tabs[4]: # Admin
            st.subheader("👑 Panel Manajemen Sesi")
            st.write("Di sini Anda bisa melihat semua password yang sedang aktif digunakan dan melakukan logout paksa jika diperlukan.")
            device_log = get_device_log()
            active_users = {p: s for p, s in device_log.items() if p != ADMIN_PASSWORD}
            if not active_users: st.success("✅ Tidak ada sesi pengguna (non-admin) yang sedang aktif.")
            else:
                st.markdown("---")
                for password, session_id in active_users.items():
                    col1, col2 = st.columns([3, 1])
                    with col1: st.text(f"Password: '{password}' sedang digunakan.")
                    with col2:
                        if st.button("Logout Paksa", key=f"logout_{password}"):
                            del device_log[password]; save_device_log(device_log); st.success(f"Sesi untuk password '{password}' berhasil dihapus!"); time.sleep(1); st.rerun()
                st.markdown("---")
