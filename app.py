import streamlit as st
import pandas as pd
import pickle
import os
import ast
import gdown
import random 
import re
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

# --- 4. Hàm lấy hình ảnh (Không đổi) ---
def get_first_image_url(images_str):
    placeholder_image = "https://cdn.freelogovectors.net/wp-content/uploads/2022/10/foodcom-logo-freelogovectors.net_-400x144.png"
    if not isinstance(images_str, str) or pd.isna(images_str):
        return placeholder_image
    match = re.search(r'"(https://[^"]+)"', images_str)
    if match:
        return match.group(1)
    else:
        if images_str.startswith('http'):
            return images_str
    return placeholder_image

# --- 5. HÀM TÍNH TOÁN (Cho Tab 2) ---
def get_all_predictions(user_id):
    all_ids = all_recipe_ids_tuple
    predictions = []
    for recipe_id in all_ids:
        pred = model.predict(uid=user_id, iid=recipe_id)
        predictions.append((recipe_id, pred.est))
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions

# --- 6. HÀM XÂY DỰNG TAB 1 (Duyệt món ăn) ---
def build_browse_tab(metadata_df):
    
    # <<< SỬA LỖI TẠI ĐÂY: Quay lại dùng cú pháp 'with'
    # 6.1. Xử lý hiển thị chi tiết
    if 'detail_recipe_id' in st.session_state and st.session_state['detail_recipe_id'] is not None:
        recipe_id = st.session_state['detail_recipe_id']
        recipe_data = metadata_df[metadata_df['RecipeId'] == recipe_id].iloc[0]
        
        # Dùng 'with' (giờ sẽ hoạt động vì bạn đã nâng cấp Streamlit)
        with st.dialog(f"Chi tiết món ăn: {recipe_data['Name']}"):
            st.image(get_first_image_url(recipe_data['Images']), use_container_width=True)
            st.subheader(recipe_data['Name'])
            st.dataframe(recipe_data) 
            
            if st.button("Đóng", key="close_dialog"):
                st.session_state['detail_recipe_id'] = None
                st.rerun() 
    # <<< KẾT THÚC SỬA LỖI

    # 6.2. Bộ lọc
    with st.expander("Tìm kiếm và Lọc", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            search_name = st.text_input("Tìm theo Tên món ăn")
            search_id = st.number_input("Tìm theo ID món ăn", value=None, step=1, placeholder="Nhập ID...")
        
        with col2:
            categories = ["Tất cả"] + sorted(list(metadata_df['RecipeCategory'].dropna().unique()))
            search_category = st.selectbox("Lọc theo Danh mục", options=categories)
            search_ingredients = st.text_input("Lọc theo Nguyên liệu (phân cách bằng dấu phẩy)")

    # 6.3. Áp dụng bộ lọc
    filtered_df = metadata_df.copy()

    if search_name:
        filtered_df = filtered_df[filtered_df['Name'].str.contains(search_name, case=False, na=False)]
    
    if search_id is not None:
        filtered_df = filtered_df[filtered_df['RecipeId'] == search_id]
        
    if search_category != "Tất cả":
        filtered_df = filtered_df[filtered_df['RecipeCategory'] == search_category]

    if search_ingredients:
        ingredients_list = [ing.strip() for ing in search_ingredients.split(',') if ing.strip()]
        for ing in ingredients_list:
            filtered_df = filtered_df[filtered_df['RecipeIngredientParts'].str.contains(ing, case=False, na=False)]

    # 6.4. Phân trang
    total_items = len(filtered_df)
    st.write(f"Tìm thấy **{total_items}** món ăn phù hợp.")
    
    if total_items > 0:
        ITEMS_PER_PAGE = 10
        total_pages = max(1, (total_items + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
        
        page_col1, page_col2 = st.columns([1, 1])
        with page_col1:
            page_number = st.number_input("Trang", min_value=1, max_value=total_pages, value=1, step=1)
        with page_col2:
            st.write(f"Tổng số trang: {total_pages}")

        start_idx = (page_number - 1) * ITEMS_PER_PAGE
        end_idx = min(start_idx + ITEMS_PER_PAGE, total_items)
        
        items_to_display = filtered_df.iloc[start_idx:end_idx]

        # 6.5. Vòng lặp hiển thị
        cols = st.columns(2)
        col_idx = 0
        
        for index, row in items_to_display.iterrows():
            with cols[col_idx]:
                image_url = get_first_image_url(row['Images'])
                st.image(image_url, caption=f"Recipe ID: {row['RecipeId']}", use_container_width=True)
                st.subheader(row['Name'])
                
                if st.button("Xem chi tiết", key=f"detail_{row['RecipeId']}"):
                    st.session_state['detail_recipe_id'] = row['RecipeId']
                    st.rerun() 
                
                st.divider()
            
            col_idx = (col_idx
