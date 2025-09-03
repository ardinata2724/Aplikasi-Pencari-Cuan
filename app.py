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

# ==============================================================================
# KONFIGURASI & FUNGSI PASSWORD
# ==============================================================================

# --- PENTING: Tentukan password admin Anda di sini ---
ADMIN_PASSWORD = "Andi1991" 
# --- Tentukan durasi timeout dalam menit ---
SESSION_TIMEOUT_MINUTES = 15

PASSWORDS_FILE = "passwords.json"
DEVICE_LOG_FILE = "device_log.json"

def get_valid_passwords():
    """
    Membaca daftar password dari file JSON.
    Fungsi ini sudah diperbaiki untuk menangani file kosong atau format JSON yang salah.
    """
    if not os.path.exists(PASSWORDS_FILE):
        return []
    try:
        with open(PASSWORDS_FILE, 'r') as f:
            content = f.read()
            # Jika file kosong, kembalikan list kosong agar tidak error
            if not content.strip():
                return []
            return json.loads(content)
    except json.JSONDecodeError:
        # Jika format JSON salah, beri peringatan dan kembalikan list kosong
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
    """Fungsi untuk melakukan logout paksa dan membersihkan sesi."""
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
        if key in st.session_state:
            del st.session_state[key]

def check_password_per_device():
    """Memeriksa password, sesi, dan timeout."""
    for key, default in [('logged_in', False), ('is_admin', False), ('user_session_id', None), ('last_activity_time', None)]:
        if key not in st.session_state: st.session_state[key] = default

    if st.session_state.logged_in and st.session_state.last_activity_time:
        if datetime.now() - st.session_state.last_activity_time > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
            force_logout()
            st.warning(f"Anda telah otomatis logout karena tidak aktif selama {SESSION_TIMEOUT_MINUTES} menit.")
            time.sleep(2); st.rerun()

    if st.session_state.logged_in:
        device_log = get_device_log()
        password_found = any(sid == st.session_state.user_session_id for sid in device_log.values())
        if password_found:
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
            if password in device_log:
                session_id = device_log[password]
                st.session_state.user_session_id = session_id
                st.session_state.logged_in = True
                st.session_state.is_admin = (password == ADMIN_PASSWORD)
                st.rerun()
            else:
                session_id = str(uuid.uuid4())
                st.session_state.user_session_id = session_id
                st.session_state.logged_in = True
                st.session_state.is_admin = (password == ADMIN_PASSWORD)
                device_log[password] = session_id
                save_device_log(device_log)
                st.rerun()
        else:
            st.error("😕 Password salah atau tidak terdaftar.")
            st.session_state.logged_in = False
            
    return False

# ==============================================================================
# BAGIAN 1: FUNGSI-FUNGSI INTI (Tidak ada perubahan di bagian ini)
# ==============================================================================
DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]
BBFS_LABELS = ["bbfs_ribuan-ratusan", "bbfs_ratusan-puluhan", "bbfs_puluhan-satuan"]
JUMLAH_LABELS = ["jumlah_depan", "jumlah_tengah", "jumlah_belakang"]
SHIO_LABELS = ["shio_depan", "shio_tengah", "shio_belakang"]
JALUR_LABELS = ["jalur_ribuan-ratusan", "jalur_ratusan-puluhan", "jalur_puluhan-satuan"]
JALUR_ANGKA_MAP = {1: "01*13*25*37*49*61*73*85*97*04*16*28*40*52*64*76*88*00*07*19*31*43*55*67*79*91*10*22*34*46*58*70*82*94", 2: "02*14*26*38*50*62*74*86*98*05*17*29*41*53*65*77*89*08*20*32*44*56*68*80*92*11*23*35*47*59*71*83*95", 3: "03*15*27*39*51*63*75*87*99*06*18*30*42*54*66*78*90*09*21*33*45*57*69*81*93*12*24*36*48*60*72*84*96"}
@st.cache_resource
def _get_positional_encoding_layer():
    import tensorflow as tf
    class PositionalEncoding(tf.keras.layers.Layer):
        def call(self, x):
            seq_len, d_model = tf.shape(x)[1], tf.shape(x)[2]
            pos = tf.cast(tf.range(seq_len)[:, tf.newaxis], dtype=tf.float32); i = tf.cast(tf.range(d_model)[tf.newaxis, :], dtype=tf.float32)
            angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32)); angle_rads = pos * angle_rates
            sines, cosines = tf.math.sin(angle_rads[:, 0::2]), tf.math.cos(angle_rads[:, 1::2]); pos_encoding = tf.concat([sines, cosines], axis=-1)
            return x + tf.cast(tf.expand_dims(pos_encoding, 0), tf.float32)
    return PositionalEncoding
@st.cache_resource
def load_cached_model(model_path):
    from tensorflow.keras.models import load_model
    PositionalEncoding = _get_positional_encoding_layer()
    if os.path.exists(model_path):
        try: return load_model(model_path, custom_objects={"PositionalEncoding": PositionalEncoding})
        except Exception as e: st.error(f"Gagal memuat model di {model_path}: {e}")
    return None
def top6_markov(df, top_n=6):
    if df.empty or len(df) < 10: return [], None
    data = df["angka"].astype(str).tolist(); matrix = [defaultdict(lambda: defaultdict(int)) for _ in range(3)]
    for number in data:
        digits = f"{int(number):04d}"
        for i in range(3): matrix[i][digits[i]][digits[i+1]] += 1
    freq_ribuan = Counter([int(x[0]) for x in data]); hasil = [[k for k, _ in freq_ribuan.most_common(top_n)]]
    for i in range(3):
        kandidat = [int(k) for prev in matrix[i] for k in matrix[i][prev].keys()]; top = [k for k, _ in Counter(kandidat).most_common()]; hasil.append(top)
    unique_hasil = [list(dict.fromkeys(h))[:top_n] for h in hasil]; return unique_hasil, None
def calculate_angka_main_stats(df, top_n=5):
    if df.empty or len(df) < 10: return {"jumlah_2d": "Data tidak cukup", "colok_bebas": "Data tidak cukup"}
    angka_str = df["angka"].astype(str).str.zfill(4); puluhan = angka_str.str[2].astype(int); satuan = angka_str.str[3].astype(int); jumlah = (puluhan + satuan) % 10
    jumlah_2d = ", ".join(map(str, jumlah.value_counts().nlargest(top_n).index)); all_digits = "".join(angka_str.tolist()); colok_bebas = ", ".join([item[0] for item in Counter(all_digits).most_common(top_n)])
    return {"jumlah_2d": jumlah_2d, "colok_bebas": colok_bebas}
def calculate_markov_ai(df, top_n=6, mode='belakang'):
    if df.empty or len(df) < 10: return "Data tidak cukup untuk analisis."
    mode_to_idx = {'depan': 3, 'tengah': 1, 'belakang': 0}; start_idx = mode_to_idx[mode]; angka_str_list = df["angka"].astype(str).str.zfill(4).tolist(); transitions = defaultdict(list)
    for num_str in angka_str_list:
        start_digit = num_str[start_idx]; following_digits = [d for i, d in enumerate(num_str) if i != start_idx]; transitions[start_digit].extend(following_digits)
    prediction_map = {}
    for start_digit, following_digits in transitions.items():
        top_digits_counts = Counter(following_digits).most_common(); final_digits = list(dict.fromkeys([d for d, c in top_digits_counts]))
        if len(final_digits) < top_n:
            all_possible_digits = list(map(str, range(10))); random.shuffle(all_possible_digits)
            for digit in all_possible_digits:
                if len(final_digits) >= top_n: break
                if digit not in set(final_digits): final_digits.append(digit)
        prediction_map[start_digit] = "".join(final_digits[:top_n])
    output_lines = [f"{num_str} = {prediction_map.get(num_str[start_idx], '')} ai" for num_str in angka_str_list[-30:]]; return "\n".join(output_lines)
def tf_preprocess_data(df, window_size=7):
    from tensorflow.keras.utils import to_categorical
    if len(df) < window_size + 1: return np.array([]), {}
    angka = df["angka"].values; labels_to_process = DIGIT_LABELS + BBFS_LABELS + JUMLAH_LABELS + SHIO_LABELS; sequences, targets = [], {label: [] for label in labels_to_process}
    for i in range(len(angka) - window_size):
        window = [str(x).zfill(4) for x in angka[i:i+window_size+1]];
        if any(not x.isdigit() for x in window): continue
        sequences.append([int(d) for num in window[:-1] for d in num]); target_digits = [int(d) for d in window[-1]]
        for j, label in enumerate(DIGIT_LABELS): targets[label].append(to_categorical(target_digits[j], num_classes=10))
        jumlah_map = {"jumlah_depan": (target_digits[0] + target_digits[1]) % 10, "jumlah_tengah": (target_digits[1] + target_digits[2]) % 10, "jumlah_belakang": (target_digits[2] + target_digits[3]) % 10}
        for label, value in jumlah_map.items(): targets[label].append(to_categorical(value, num_classes=10))
        bbfs_map = {"bbfs_ribuan-ratusan": [target_digits[0], target_digits[1]], "bbfs_ratusan-puluhan": [target_digits[1], target_digits[2]], "bbfs_puluhan-satuan": [target_digits[2], target_digits[3]]}
        for label, digit_pair in bbfs_map.items():
            multi_hot_target = np.zeros(10, dtype=np.float32)
            for digit in np.unique(digit_pair): multi_hot_target[digit] = 1.0
            targets[label].append(multi_hot_target)
        shio_num_map = {"shio_depan": target_digits[0] * 10 + target_digits[1], "shio_tengah": target_digits[1] * 10 + target_digits[2], "shio_belakang": target_digits[2] * 10 + target_digits[3]}
        for label, two_digit_num in shio_num_map.items():
            shio_index = (two_digit_num - 1) % 12 if two_digit_num > 0 else 11; targets[label].append(to_categorical(shio_index, num_classes=12))
    final_targets = {label: np.array(v) for label, v in targets.items() if v}; return np.array(sequences), final_targets
def tf_preprocess_data_for_jalur(df, window_size, target_position):
    from tensorflow.keras.utils import to_categorical
    if len(df) < window_size + 1: return np.array([]), np.array([])
    jalur_map = {1: [1, 4, 7, 10], 2: [2, 5, 8, 11], 3: [3, 6, 9, 12]}; shio_to_jalur = {shio: jalur for jalur, shios in jalur_map.items() for shio in shios}; position_map = {'ribuan-ratusan': (0, 1), 'ratusan-puluhan': (1, 2), 'puluhan-satuan': (2, 3)}; idx1, idx2 = position_map[target_position]; angka = df["angka"].values; sequences, targets = [], []
    for i in range(len(angka) - window_size):
        window = [str(x).zfill(4) for x in angka[i:i+window_size+1]];
        if any(not x.isdigit() for x in window): continue
        sequences.append([int(d) for num in window[:-1] for d in num]); target_digits = [int(d) for d in window[-1]]; two_digit_num = target_digits[idx1] * 10 + target_digits[idx2]; shio_value = (two_digit_num - 1) % 12 + 1 if two_digit_num > 0 else 12; targets.append(to_categorical(shio_to_jalur[shio_value] - 1, num_classes=3))
    return np.array(sequences), np.array(targets)
def build_tf_model(input_len, model_type, problem_type, num_classes):
    from tensorflow.keras.models import Model; from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout, Dense, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
    PositionalEncoding = _get_positional_encoding_layer(); inputs = Input(shape=(input_len,)); x = Embedding(10, 64)(inputs); x = PositionalEncoding()(x)
    if model_type == "transformer":
        attn = MultiHeadAttention(num_heads=4, key_dim=64)(x, x); x = LayerNormalization()(x + attn)
    else: x = Bidirectional(LSTM(128, return_sequences=True))(x); x = Dropout(0.3)(x)
    x = GlobalAveragePooling1D()(x); x = Dense(128, activation='relu')(x); x = Dropout(0.2)(x)
    outputs, loss = (Dense(num_classes, activation='sigmoid')(x), "binary_crossentropy") if problem_type == "multilabel" else (Dense(num_classes, activation='softmax')(x), "categorical_crossentropy")
    model = Model(inputs, outputs); return model, loss
def top_n_model(df, lokasi, window_dict, model_type, top_n):
    results = []; loc_id = lokasi.lower().strip().replace(" ", "_")
    for label in DIGIT_LABELS:
        ws = window_dict.get(label, 7); X, _ = tf_preprocess_data(df, ws)
        if X.shape[0] == 0: return None, None
        model_path = f"saved_models/{loc_id}_{label}_{model_type}.h5"; model = load_cached_model(model_path)
        if model is None: st.error(f"Model {label} tidak ditemukan."); return None, None
        pred = model.predict(X, verbose=0); results.append(list(np.mean(pred, axis=0).argsort()[-top_n:][::-1]))
    return results, None
def find_best_window_size(df, label, model_type, min_ws, max_ws, top_n, top_n_shio):
    from sklearn.model_selection import train_test_split; from tensorflow.keras.callbacks import EarlyStopping; from tensorflow.keras.metrics import TopKCategoricalAccuracy
    best_ws, best_score, table_data = None, -1, []; is_jalur_scan = label in JALUR_LABELS
    if is_jalur_scan: pt, k, nc, cols = "jalur_multiclass", 2, 3, ["Window Size", "Prediksi", "Angka Jalur"]
    elif label in BBFS_LABELS: pt, k, nc, cols = "multilabel", top_n, 10, ["Window Size", f"Top-{k}"]
    elif label in SHIO_LABELS: pt, k, nc, cols = "shio", top_n_shio, 12, ["Window Size", f"Top-{k}"]
    else: pt, k, nc, cols = "multiclass", top_n, 10, ["Window Size", f"Top-{k}"]
    bar = st.progress(0, text=f"Memulai Scan {label.upper()}... [0%]"); total_ws = (max_ws - min_ws) + 1
    for i, ws in enumerate(range(min_ws, max_ws + 1)):
        progress_value = (i + 1) / total_ws; percentage = int(progress_value * 100); bar.progress(progress_value, text=f"Mencoba WS={ws}... [{percentage}%]")
        try:
            if is_jalur_scan: X, y = tf_preprocess_data_for_jalur(df, ws, label.split('_')[1])
            else: X, y_dict = tf_preprocess_data(df, ws); y = y_dict.get(label)
            if X.shape[0] < 10: continue
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42); model, loss = build_tf_model(X.shape[1], model_type, 'multiclass' if is_jalur_scan else pt, nc); metrics = ['accuracy']
            if pt != 'multilabel': metrics.append(TopKCategoricalAccuracy(k=k))
            model.compile(optimizer="adam", loss=loss, metrics=metrics); model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val), callbacks=[EarlyStopping(monitor='val_loss', patience=3)], verbose=0)
            evals = model.evaluate(X_val, y_val, verbose=0); preds = model.predict(X_val, verbose=0)
            if is_jalur_scan:
                top_indices = np.argsort(preds[-1])[::-1][:2]; pred_str = f"{top_indices[0] + 1}-{top_indices[1] + 1}"; angka_jalur_str = f"Jalur {top_indices[0] + 1} => {JALUR_ANGKA_MAP[top_indices[0] + 1]}\n\nJalur {top_indices[1] + 1} => {JALUR_ANGKA_MAP[top_indices[1] + 1]}"; score = (evals[1] * 0.3) + (evals[2] * 0.7); table_data.append((ws, pred_str, angka_jalur_str))
            else:
                avg_conf = np.mean(np.sort(preds, axis=1)[:, -k:])*100; top_indices = np.argsort(preds[-1])[::-1][:k]; pred_str = ", ".join(map(str, top_indices + 1)) if pt == "shio" else ", ".join(map(str, top_indices)); score = (evals[1] * 0.7) + (avg_conf/100*0.3) if pt=='multilabel' else (evals[1]*0.2)+(evals[2]*0.5)+(avg_conf/100*0.3); table_data.append((ws, pred_str))
            if score > best_score: best_score, best_ws = score, ws
        except Exception as e: st.warning(f"Gagal di WS={ws}: {e}"); continue
    bar.empty(); return best_ws, pd.DataFrame(table_data, columns=cols) if table_data else pd.DataFrame()
def train_and_save_model(df, lokasi, window_dict, model_type):
    from sklearn.model_selection import train_test_split; from tensorflow.keras.callbacks import EarlyStopping
    st.info(f"Memulai pelatihan untuk {lokasi}..."); lokasi_id = lokasi.lower().strip().replace(" ", "_")
    if not os.path.exists("saved_models"): os.makedirs("saved_models")
    for label in DIGIT_LABELS:
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

    if 'angka_list' not in st.session_state: st.session_state.angka_list = []
    if 'scan_outputs
