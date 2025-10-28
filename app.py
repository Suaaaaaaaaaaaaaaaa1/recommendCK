
import streamlit as st
import pandas as pd
import pickle
import os
import ast
import gdown  # Thư viện để tải file từ Google Drive
from surprise import SVD

# --- 1. Định nghĩa Tên tệp và File IDs ---
MODEL_FILE_PATH = 'svd_model.pkl' # Tên tệp sẽ lưu trên server Streamlit
METADATA_FILE_PATH = 'recipes_metadata.csv' # Tên tệp sẽ lưu trên server Streamlit

# !!! THAY THẾ CÁC ID CỦA BẠN VÀO ĐÂY !!!
MODEL_FILE_ID = '16v3zUzOhPqnF6n3-80lYq7UcYRmej7RJ' 
METADATA_FILE_ID = '1x_Zb0mO_rOjhep71QveJcVleBLO2HCEs'

# --- 2. Hàm Tải tệp chung ---
def download_file_from_gdrive(file_id, dest_path):
    """
    Tải tệp từ Google Drive nếu nó chưa tồn tại.
    """
    if not os.path.exists(dest_path):
        with st.spinner(f"Đang tải tài nguyên: {dest_path} (lần đầu tiên)..."):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, dest_path, quiet=False)
    return dest_path

# --- 3. Hàm Tải Model và Dữ liệu ---
@st.cache_resource
def load_model(file_id, dest_path):
    """
    Tải tệp model từ GDrive, sau đó nạp vào bộ nhớ.
    """
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
    """
    Tải tệp metadata (CSV) từ GDrive, sau đó nạp vào DataFrame.
    """
    try:
        metadata_path = download_file_from_gdrive(file_id, dest_path)
        df = pd.read_csv(metadata_path)
        df = df.set_index('Recipe_ID')
        return df
    except Exception as e:
        st.error(f"LỖI khi tải metadata: {e}")
        return pd.DataFrame()

# --- 4. Hàm lấy hình ảnh (Đã sửa lỗi) ---
def get_first_image_url(images_str):
    placeholder_image = "https://i.imgur.com/gY9R3t1.png" 
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

# --- 5. Hàm Gợi ý chính (Không đổi) ---
def get_recommendations(user_id, model, metadata_df, num_recs=10):
    # Lấy các ID món ăn từ metadata đã tải về
    all_recipe_ids = metadata_df.index.unique()
    
    predictions = []
    # Chỉ dự đoán trên các món ăn có trong metadata
    for recipe_id in all_recipe_ids: 
        pred = model.predict(uid=user_id, iid=recipe_id)
        predictions.append((recipe_id, pred.est))
        
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n_ids = [recipe_id for recipe_id, score in predictions[:num_recs]]
    
    # Tra cứu thông tin
    recommended_recipes = metadata_df.loc[top_n_ids].copy()
    recommended_recipes['image_url'] = recommended_recipes['Images'].apply(get_first_image_url)
    return recommended_recipes

# --- 6. Xây dựng giao diện Streamlit ---
st.set_page_config(layout="wide")
st.title("Hệ thống Gợi ý Món ăn 🍲 🍳 🍰")

# Tải cả hai tệp từ Google Drive
model = load_model(MODEL_FILE_ID, MODEL_FILE_PATH)
metadata = load_metadata(METADATA_FILE_ID, METADATA_FILE_PATH)

if model and not metadata.empty:
    st.header("Tìm món ăn cho bạn")
    
    user_id_input = st.number_input(
        "Nhập User ID của bạn:", 
        min_value=1, 
        value=1535,
        step=1,
        help="Hãy nhập một User ID (ví dụ: 1535, 2046, 5201...)"
    )
    
    num_recs = st.slider("Số lượng gợi ý:", min_value=5, max_value=20, value=10)

    if st.button("Tìm kiếm gợi ý"):
        with st.spinner("Đang tính toán..."):
            recs_df = get_recommendations(user_id_input, model, metadata, num_recs)
            
            st.subheader(f"Gợi ý cho User {user_id_input}:")
            
            cols = st.columns(2)
            col_idx = 0
            
            for index, row in recs_df.iterrows():
                with cols[col_idx]:
                    st.image(row['image_url'], caption=f"Recipe ID: {row.name}", use_column_width=True)
                    st.subheader(row['Name'])
                    # Kiểm tra xem cột 'Description' có tồn tại không
                    if 'Description' in row and pd.notna(row['Description']):
                         st.markdown(f"**Mô tả:** {row['Description'][:150]}...")
                    st.divider()
                
                col_idx = (col_idx + 1) % 2
else:
    st.error("Không thể tải mô hình hoặc dữ liệu từ Google Drive. Vui lòng kiểm tra lại File IDs.")
