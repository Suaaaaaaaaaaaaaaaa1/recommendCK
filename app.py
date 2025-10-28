import streamlit as st
import pandas as pd
import pickle
import os
import ast
import gdown
import random # <-- THÊM THƯ VIỆN NÀY
from surprise import SVD

# --- 1. Định nghĩa Tên tệp và File IDs ---
MODEL_FILE_PATH = 'svd_model.pkl' 
METADATA_FILE_PATH = 'recipes_metadata.csv' 

# !!! THAY THẾ CÁC ID CỦA BẠN VÀO ĐÂY !!!
MODEL_FILE_ID = '16v3zUzOhPqnF6n3-80lYq7UcYRmej7RJ' 
METADATA_FILE_ID = '1x_Zb0mO_rOjhep71QveJcVleBLO2HCEs'

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

# --- 4. Hàm lấy hình ảnh (Không đổi) ---
def get_first_image_url(images_str):
    placeholder_image = "Không có ảnh" 
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

# --- 5. HÀM TÍNH TOÁN (ĐÃ SỬA LỖI) ---
@st.cache_data
def get_sampled_predictions(user_id, _model, all_recipe_ids, sample_size=20000):
    """
    Tính toán dự đoán trên một MẪU NGẪU NHIÊN để tránh crash RAM.
    """
    
    # 1. Lấy mẫu ngẫu nhiên
    if len(all_recipe_ids) > sample_size:
        # Chuyển đổi sang list để random.sample có thể hoạt động
        sampled_ids = random.sample(list(all_recipe_ids), sample_size)
    else:
        sampled_ids = all_recipe_ids

    # 2. Chỉ dự đoán trên MẪU đã lấy
    predictions = []
    for recipe_id in sampled_ids:
        pred = _model.predict(uid=user_id, iid=recipe_id)
        predictions.append((recipe_id, pred.est))
        
    # 3. Sắp xếp danh sách (nhỏ hơn nhiều)
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions

# --- 6. Xây dựng giao diện Streamlit ---
st.set_page_config(layout="wide")
st.title("Hệ thống Gợi ý Món ăn 🍲 🍳 🍰")

model = load_model(MODEL_FILE_ID, MODEL_FILE_PATH)
metadata_df = load_metadata(METADATA_FILE_ID, METADATA_FILE_PATH)

if model and not metadata_df.empty:
    st.header("Tìm món ăn cho bạn")
    
    # Lấy danh sách ID món ăn (chỉ chạy 1 lần)
    all_recipe_ids_list = metadata_df['RecipeId'].unique()
    
    # Đặt index sau (để tra cứu nhanh)
    metadata_df = metadata_df.set_index('RecipeId')
    
    user_id_input = st.number_input(
        "Nhập User ID của bạn:", 
        min_value=1, 
        value=1535,
        step=1,
        help="Hãy nhập một User ID (ví dụ: 1535, 2046, 5201...)"
    )
    
    num_recs = st.slider("Số lượng gợi ý:", min_value=5, max_value=20, value=10)

    # --- SỬA LỖI TẠI ĐÂY ---
    with st.spinner("Đang tính toán gợi ý..."):
        
        # 1. Gọi hàm LẤY MẪU (đã cache, siêu nhanh)
        all_preds = get_sampled_predictions(user_id_input, model, all_recipe_ids_list)
        
        # 2. Lấy Top N
        top_n_preds = all_preds[:num_recs]
        
        # 3. Lấy Recipe IDs
        top_n_ids = [recipe_id for recipe_id, score in top_n_preds]
        
        # 4. Tra cứu metadata
        recs_df = metadata_df.loc[top_n_ids].copy()
        
        st.subheader(f"Gợi ý cho User {user_id_input}:")
        
        cols = st.columns(2)
        col_idx = 0
        
        for index, row in recs_df.iterrows():
            with cols[col_idx]:
                image_url = get_first_image_url(row['Images'])
                st.image(image_url, caption=f"Recipe ID: {row.name}", use_column_width=True)
                st.subheader(row['Name'])
                if 'Description' in row and pd.notna(row['Description']):
                     st.markdown(f"**Mô tả:** {row['Description'][:150]}...")
                st.divider()
            
            col_idx = (col_idx + 1) % 2
else:
    st.error("Không thể tải mô hình hoặc dữ liệu từ Google Drive. Vui lòng kiểm tra lại File IDs.")
