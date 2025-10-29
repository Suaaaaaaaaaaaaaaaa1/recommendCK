import streamlit as st
import pandas as pd
import pickle
import os
import ast
import gdown
import random 
import re  # <-- THÊM THƯ VIỆN NÀY
from surprise import SVD

# --- 1. Định nghĩa Tên tệp và File IDs ---
MODEL_FILE_PATH = 'svd_model.pkl' 
METADATA_FILE_PATH = 'recipes_metadata.csv' 

# !!! ID CỦA BẠN TỪ LẦN TRƯỚC !!!
MODEL_FILE_ID = '1mSWLAjm2Ho6Aox61PrIQJUgNyJObKSbu' 
METADATA_FILE_ID = '1jCm7OruZnwkkd5GRU42dycNcQdGOKNRv'

# --- 2. Hàm Tải tệp chung ---
def download_file_from_gdrive(file_id, dest_path):
    if not os.path.exists(dest_path):
        with st.spinner(f"Đang tải tài nguyên: {dest_path} (lần đầu tiên)..."):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, dest_path, quiet=False)
    return dest_path

# --- 3. Hàm Tải Model và Dữ liệu ---
@st.cache_resource
def load_model(file_id, dest_path):
    try:
        model_path = download_file_from_gdrive(file_id, dest_path)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"LỖI khi tải mô hình: {e}")
        return None

@st.cache_data
def load_metadata(file_id, dest_path):
    try:
        metadata_path = download_file_from_gdrive(file_id, dest_path)
        df = pd.read_csv(metadata_path)
        return df
    except Exception as e:
        st.error(f"LỖI khi tải metadata: {e}")
        return pd.DataFrame()

# --- 4. HÀM LẤY HÌNH ẢNH (ĐÃ SỬA LỖI HOÀN TOÀN) ---
def get_first_image_url(images_str):
    placeholder_image = "https://cdn.freelogovectors.net/wp-content/uploads/2022/10/foodcom-logo-freelogovectors.net_-400x144.png"
    
    # Kiểm tra nếu dữ liệu rỗng hoặc không phải chuỗi
    if not isinstance(images_str, str) or pd.isna(images_str):
        return placeholder_image

    # Dùng regex để tìm URL đầu tiên (bên trong dấu " ")
    # (Hỗ trợ cả định dạng c("url1",...) và ['url1',...])
    match = re.search(r'"(https://[^"]+)"', images_str)
    
    if match:
        # Nếu tìm thấy (ví dụ: c("url1", ...))
        return match.group(1) # match.group(1) là url1
    else:
        # Nếu không tìm thấy (ví dụ: URL trần không có dấu ")
        if images_str.startswith('http'):
            return images_str
            
    # Nếu mọi thứ thất bại (ví dụ: [] hoặc "")
    return placeholder_image

# --- 5. HÀM TÍNH TOÁN (Đã bỏ sample) ---
def get_all_predictions(user_id):
    all_ids = all_recipe_ids_tuple
    predictions = []
    for recipe_id in all_ids:
        pred = model.predict(uid=user_id, iid=recipe_id)
        predictions.append((recipe_id, pred.est))
        
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions

# --- 6. Xây dựng giao diện Streamlit ---
st.set_page_config(layout="wide")
st.title("Hệ thống Gợi ý Món ăn 🍲 🍳 🍰")

model = load_model(MODEL_FILE_ID, MODEL_FILE_PATH)
metadata_df = load_metadata(METADATA_FILE_ID, METADATA_FILE_PATH)

if model and not metadata_df.empty:
    st.header("Tìm món ăn cho bạn")
    
    # Sửa lỗi typo (gõ nhầm)
    all_recipe_ids_tuple = tuple(metadata_df['RecipeId'].unique())
    metadata_df = metadata_df.set_index('RecipeId')
    
    # --- WIDGETS ---
    user_id_input = st.number_input(
        "Nhập User ID của bạn:", 
        min_value=1, 
        value=1535,
        step=1,
        help="Hãy nhập một User ID (ví dụ: 1535, 2046, 5201...)"
    )
    
    num_recs = st.slider("Số lượng gợi ý:", min_value=5, max_value=20, value=10)

    # --- THÊM LẠI NÚT BẤM ---
    if st.button("Tìm kiếm gợi ý"):
        with st.spinner("Đang tính toán gợi ý (trên toàn bộ dữ liệu)..."):
            all_preds = get_all_predictions(user_id_input) 
            st.session_state['all_predictions'] = all_preds
    
    # --- HIỂN THỊ KẾT QUẢ TỪ SESSION_STATE ---
    if 'all_predictions' in st.session_state:
        all_preds = st.session_state['all_predictions']
        top_n_preds = all_preds[:num_recs]
        
        top_n_ids = [recipe_id for recipe_id, score in top_n_preds]
        
        valid_top_n_ids = [idx for idx in top_n_ids if idx in metadata_df.index]
        if valid_top_n_ids:
            recs_df = metadata_df.loc[valid_top_n_ids].copy()
            
            st.subheader(f"Kết quả gợi ý:")
            
            cols = st.columns(2)
            col_idx = 0
            
            for index, row in recs_df.iterrows():
                with cols[col_idx]:
                    # DÙNG HÀM MỚI (ĐÃ SỬA LỖI)
                    image_url = get_first_image_url(row['Images'])
                    
                    st.image(image_url, caption=f"Recipe ID: {row.name}", use_container_width=True)
                    st.subheader(row['Name'])
                    if 'Description' in row and pd.notna(row['Description']):
                         st.markdown(f"**Mô tả:** {row['Description'][:150]}...")
                    st.divider()
                
                col_idx = (col_idx + 1) % 2
        else:
            st.warning("Không tìm thấy gợi ý nào.")
else:
    st.error("Không thể tải mô hình hoặc dữ liệu từ Google Drive. Vui lòng kiểm tra lại File IDs.")
