import streamlit as st
import pandas as pd
import pickle
import os
import ast
import gdown
import random 
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
    # Dùng cache_resource để tải model 1 LẦN DUY NHẤT
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
    # Dùng cache_data để tải metadata 1 LẦN DUY NHẤT
    try:
        metadata_path = download_file_from_gdrive(file_id, dest_path)
        df = pd.read_csv(metadata_path)
        return df
    except Exception as e:
        st.error(f"LỖI khi tải metadata: {e}")
        return pd.DataFrame()

# --- 4. Hàm lấy hình ảnh (Không đổi) ---
def get_first_image_url(images_str):
    placeholder_image = "https://cdn.freelogovectors.net/wp-content/uploads/2022/10/foodcom-logo-freelogovectors.net_-400x144.png"
    if not isinstance(images_str, str) or pd.isna(images_str):
        return placeholder_image
    try:
        evaluated_data = ast.literal_eval(images_str)
        if isinstance(evaluated_data, list):
            if len(evaluated_data) > 0:
                return evaluated_data[0]
            else:
                return placeholder_image
        if isinstance(evaluated_data, str):
            if evaluated_data.startswith('http'):
                return evaluated_data
            else:
                return placeholder_image
    except (ValueError, SyntaxError):
        if images_str.startswith('http'):
            return images_str
    return placeholder_image

# --- 5. HÀM TÍNH TOÁN (ĐÃ BỎ SAMPLE) ---
# Hàm này không cần cache nữa vì nó được gọi bằng nút bấm
def get_all_predictions(user_id):
    """
    Tính toán dự đoán trên TOÀN BỘ danh sách món ăn.
    """
    
    # 1. Lấy TOÀN BỘ ID (dùng biến toàn cục)
    all_ids = all_recipe_ids_tuple

    # 2. Dự đoán trên TOÀN BỘ (dùng biến toàn cục)
    predictions = []
    for recipe_id in all_ids:
        pred = model.predict(uid=user_id, iid=recipe_id)
        predictions.append((recipe_id, pred.est))
        
    # 3. Sắp xếp danh sách
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions

# --- 6. Xây dựng giao diện Streamlit ---
st.set_page_config(layout="wide")
st.title("Hệ thống Gợi ý Món ăn 🍲 🍳 🍰")

# --- NẠP CÁC BIẾN "TOÀN CỤC" ---
model = load_model(MODEL_FILE_ID, MODEL_FILE_PATH)
metadata_df = load_metadata(METADATA_FILE_ID, METADATA_FILE_PATH)

if model and not metadata_df.empty:
    st.header("Tìm món ăn cho bạn")
    
    # --- SỬA LỖI TYPO (gõ nhầm) TẠI ĐÂY ---
    # Tên cột phải là 'Recipe_ID' (có dấu gạch dưới)
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
            # Chạy hàm tính toán MỚI (không sample)
            all_preds = get_all_predictions(user_id_input) 
            # LƯU kết quả vào session_state
            st.session_state['all_predictions'] = all_preds
    
    # --- HIỂN THỊ KẾT QUẢ TỪ SESSION_STATE ---
    # Luôn kiểm tra xem 'all_predictions' đã tồn tại chưa
    if 'all_predictions' in st.session_state:
        # Lấy Top N từ kết quả đã lưu
        all_preds = st.session_state['all_predictions']
        top_n_preds = all_preds[:num_recs]
        
        top_n_ids = [recipe_id for recipe_id, score in top_n_preds]
        
        # Tra cứu metadata
        valid_top_n_ids = [idx for idx in top_n_ids if idx in metadata_df.index]
        if valid_top_n_ids:
            recs_df = metadata_df.loc[valid_top_n_ids].copy()
            
            st.subheader(f"Kết quả gợi ý:")
            
            cols = st.columns(2)
            col_idx = 0
            
            for index, row in recs_df.iterrows():
                with cols[col_idx]:
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
