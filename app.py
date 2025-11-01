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

# --- 4. HÀM LẤY HÌNH ẢNH (Không đổi) ---
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

# --- 5. HÀM LÀM SẠCH TEXT (Không đổi) ---
def format_c_string(text_str, as_list=False):
    placeholder = "Không có thông tin"
    if not isinstance(text_str, str) or pd.isna(text_str):
        return placeholder if not as_list else []
    matches = re.findall(r'"([^"]+)"', text_str)
    if matches:
        if as_list:
            return matches 
        else:
            return ", ".join(matches) 
    else:
        return text_str if not as_list else [text_str]

# --- 6. HÀM TÍNH TOÁN (Cho Tab 2) ---
def get_all_predictions(user_id):
    all_ids = all_recipe_ids_tuple
    predictions = []
    for recipe_id in all_ids:
        pred = model.predict(uid=user_id, iid=recipe_id)
        predictions.append((recipe_id, pred.est))
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions

# --- 7. HÀM XÂY DỰNG TAB 1 (Duyệt món ăn) ---
def build_browse_tab(metadata_df):
    
    # 7.1. Chế độ XEM CHI TIẾT
    if 'detail_recipe_id' in st.session_state and st.session_state.detail_recipe_id is not None:
        recipe_id = st.session_state.detail_recipe_id
        recipe_data = metadata_df[metadata_df['RecipeId'] == recipe_id].iloc[0]
        
        # Bố cục trang chi tiết
        if st.button("⬅️ Quay lại danh sách"):
            st.session_state.detail_recipe_id = None
            st.rerun()
        
        img_col, info_col = st.columns([1, 2])
        with img_col:
            st.image(get_first_image_url(recipe_data.get('Images')), use_container_width=True)
        with info_col:
            st.subheader(recipe_data.get('Name', 'N/A'))
            st.markdown(f"**ID:** {recipe_data.get('RecipeId', 'N/A')}")
            st.markdown(f"**Tác giả:** {recipe_data.get('AuthorName', 'N/A')}")
            st.markdown(f"**Danh mục:** {recipe_data.get('RecipeCategory', 'N/A')}")
            st.markdown(f"**Ngày đăng:** {recipe_data.get('DatePublished', 'N/A')}")
            st.markdown("---")
            st.markdown(f"**Thời gian chuẩn bị:** {recipe_data.get('PrepTime', 'N/A')}")
            st.markdown(f"**Thời gian nấu:** {recipe_data.get('CookTime', 'N/A')}")
            st.markdown(f"**Tổng thời gian:** {recipe_data.get('TotalTime', 'N/A')}")
            st.markdown("---")
            st.markdown(f"**Đánh giá:** {recipe_data.get('AggregatedRating', 'N/A')} / 5.0 ({recipe_data.get('ReviewCount', 0)} lượt)")
            st.markdown("---")
            st.markdown(f"**Calories:** {recipe_data.get('Calories', 'N/A')}")
            st.markdown(f"**Chất béo (Fat):** {recipe_data.get('FatContent', 'N/A')}")
            st.markdown(f"**Chất béo bão hòa:** {recipe_data.get('SaturatedFatContent', 'N/A')}")
            st.markdown(f"**Cholesterol:** {recipe_data.get('CholesterolContent', 'N/A')}")
            st.markdown(f"**Sodium:** {recipe_data.get('SodiumContent', 'N/A')}")
            st.markdown(f"**Carbohydrate:** {recipe_data.get('CarbohydrateContent', 'N/A')}")
            st.markdown(f"**Chất xơ (Fiber):** {recipe_data.get('FiberContent', 'N/A')}")
            st.markdown(f"**Đường (Sugar):** {recipe_data.get('SugarContent', 'N/A')}")
            st.markdown(f"**Chất đạm (Protein):** {recipe_data.get('ProteinContent', 'N/A')}")

        st.markdown("---")
        st.subheader("Mô tả")
        st.write(recipe_data.get('Description', 'N/A'))
        st.subheader("Nguyên liệu")
        ingredients_formatted = format_c_string(recipe_data.get('RecipeIngredientParts'), as_list=False)
        st.write(ingredients_formatted)
        st.subheader("Hướng dẫn")
        instructions_list = format_c_string(recipe_data.get('RecipeInstructions'), as_list=True)
        if isinstance(instructions_list, list):
            for i, step in enumerate(instructions_list):
                st.markdown(f"{i+1}. {step}")
        else:
            st.write(instructions_list)
        return 
    
    # 7.2. Chế độ DANH SÁCH (Mặc định)
    
    # <<< THÊM MỚI TẠI ĐÂY: Hàm reset trang
    def reset_page_number():
        if st.session_state.page_number > 1:
            st.session_state.page_number = 1

    with st.expander("Tìm kiếm và Lọc", expanded=True):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            # <<< SỬA LỖI TẠI ĐÂY: Quay lại st.text_input và liên kết với session_state
            st.text_input("Tìm theo Tên món ăn", key="search_name", on_change=reset_page_number)
            st.text_input("Lọc theo Nguyên liệu (phân cách bằng dấu phẩy)", key="search_ingredients", on_change=reset_page_number)
        
        with col2:
            st.number_input("Tìm theo ID món ăn", value=None, step=1, placeholder="Nhập ID...", key="search_id", on_change=reset_page_number)
            categories = ["Tất cả"] + sorted(list(metadata_df['RecipeCategory'].dropna().unique()))
            st.selectbox("Lọc theo Danh mục", options=categories, key="search_category", on_change=reset_page_number)
        
        with col3:
             # <<< THÊM MỚI TẠI ĐÂY: Nút Xóa bộ lọc
            st.write("Xóa bộ lọc:") # Thêm label
            if st.button("Xóa toàn bộ", use_container_width=True):
                st.session_state.search_name = ""
                st.session_state.search_id = None
                st.session_state.search_category = "Tất cả"
                st.session_state.search_ingredients = ""
                st.session_state.page_number = 1
                st.rerun()

    # Áp dụng bộ lọc (Đọc từ st.session_state)
    filtered_df = metadata_df.copy()

    if st.session_state.search_name:
        filtered_df = filtered_df[filtered_df['Name'].str.contains(st.session_state.search_name, case=False, na=False)]
    if st.session_state.search_id is not None:
        filtered_df = filtered_df[filtered_df['RecipeId'] == st.session_state.search_id]
    if st.session_state.search_category != "Tất cả":
        filtered_df = filtered_df[filtered_df['RecipeCategory'] == st.session_state.search_category]
    if st.session_state.search_ingredients:
        ingredients_list = [ing.strip() for ing in st.session_state.search_ingredients.split(',') if ing.strip()]
        for ing in ingredients_list:
            filtered_df = filtered_df[filtered_df['RecipeIngredientParts'].str.contains(ing, case=False, na=False)]

    # Phân trang
    total_items = len(filtered_df)
    st.write(f"Tìm thấy **{total_items}** món ăn phù hợp.")
    
    if total_items > 0:
        ITEMS_PER_PAGE = 9 
        total_pages = max(1, (total_items + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
        
        page_col1, page_col2 = st.columns([1, 1])
        with page_col1:
            # <<< SỬA LỖI TẠI ĐÂY: Liên kết với session_state
            st.number_input("Trang", min_value=1, max_value=total_pages, step=1, key="page_number")
        with page_col2:
            st.write(f"Tổng số trang: {total_pages}")

        start_idx = (st.session_state.page_number - 1) * ITEMS_PER_PAGE
        end_idx = min(start_idx + ITEMS_PER_PAGE, total_items)
        items_to_display = filtered_df.iloc[start_idx:end_idx]

        # Vòng lặp hiển thị (3 cột)
        cols = st.columns(3)
        col_idx = 0
        
        for index, row in items_to_display.iterrows():
            with cols[col_idx]:
                image_url = get_first_image_url(row['Images'])
                st.image(image_url, caption=f"Recipe ID: {row['RecipeId']}", use_container_width=True)
                st.subheader(row['Name'])
                
                if st.button("Xem chi tiết", key=f"detail_{row['RecipeId']}"):
                    st.session_state.detail_recipe_id = row['RecipeId']
                    st.rerun() 
                
                st.divider()
            
            col_idx = (col_idx + 1) % 3

# --- 8. HÀM XÂY DỰNG TAB 2 (Gợi ý) ---
def build_predict_tab(metadata_df_indexed):
    
    st.header("Tìm món ăn cho bạn")
    
    user_id_input = st.number_input(
        "Nhập User ID của bạn:", 
        min_value=1, 
        value=1535,
        step=1,
        help="Hãy nhập một User ID (ví dụ: 1535, 2046, 5201...)"
    )
    num_recs = st.slider("Số lượng gợi ý:", min_value=3, max_value=21, value=9, step=3)

    if st.button("Tìm kiếm gợi ý"):
        with st.spinner("Đang tính toán gợi ý (trên toàn bộ dữ liệu)..."):
            all_preds = get_all_predictions(user_id_input) 
            st.session_state['all_predictions'] = all_preds
    
    if 'all_predictions' in st.session_state and st.session_state.all_predictions is not None:
        all_preds = st.session_state.all_predictions
        top_n_preds = all_preds[:num_recs]
        top_n_ids = [recipe_id for recipe_id, score in top_n_preds]
        
        valid_top_n_ids = [idx for idx in top_n_ids if idx in metadata_df_indexed.index]
        if valid_top_n_ids:
            recs_df = metadata_df_indexed.loc[valid_top_n_ids].copy()
            
            st.subheader(f"Kết quả gợi ý:")
            
            # 3 cột
            cols = st.columns(3)
            col_idx = 0
            
            for index, row in recs_df.iterrows():
                with cols[col_idx]:
                    image_url = get_first_image_url(row['Images'])
                    st.image(image_url, caption=f"Recipe ID: {row.name}", use_container_width=True)
                    st.subheader(row['Name'])
                    if 'Description' in row and pd.notna(row['Description']):
                         st.markdown(f"**Mô tả:** {row['Description'][:150]}...")
                    st.divider()
                
                col_idx = (col_idx + 1) % 3
        else:
            st.warning("Không tìm thấy gợi ý nào.")

# --- 9. CHẠY ỨNG DỤNG CHÍNH ---
st.set_page_config(layout="wide")
st.title("Hệ thống Gợi ý Món ăn 🍲 🍳 🍰")

model = load_model(MODEL_FILE_ID, MODEL_FILE_PATH)
metadata_df = load_metadata(METADATA_FILE_ID, METADATA_FILE_PATH)

if model and not metadata_df.empty:
    
    # <<< THÊM MỚI TẠI ĐÂY: Khởi tạo tất cả session state
    if 'detail_recipe_id' not in st.session_state:
        st.session_state.detail_recipe_id = None
    if 'all_predictions' not in st.session_state:
        st.session_state.all_predictions = None
    if 'search_name' not in st.session_state:
        st.session_state.search_name = ""
    if 'search_id' not in st.session_state:
        st.session_state.search_id = None
    if 'search_category' not in st.session_state:
        st.session_state.search_category = "Tất cả"
    if 'search_ingredients' not in st.session_state:
        st.session_state.search_ingredients = ""
    if 'page_number' not in st.session_state:
        st.session_state.page_number = 1
    # <<< KẾT THÚC THÊM MỚI
    
    all_recipe_ids_tuple = tuple(metadata_df['RecipeId'].unique())
    metadata_df_indexed = metadata_df.set_index('RecipeId')
    
    tab1, tab2 = st.tabs(["Duyệt Món Ăn", "Gợi Ý Cho Bạn"])

    with tab1:
        build_browse_tab(metadata_df)
    with tab2:
        build_predict_tab(metadata_df_indexed)
else:
    st.error("Không thể tải mô hình hoặc dữ liệu từ Google Drive. Vui lòng kiểm tra lại File IDs.")
