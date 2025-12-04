import streamlit as st
import requests
import pandas as pd
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Lviv City Pulse",
    page_icon="🏙️",
    layout="wide"
)

# --- Constants ---
MODEL_API_URL = "http://model-api:8000/predict"
DISTRICTS = [
    "Галицький", 
    "Залізничний", 
    "Личаківський", 
    "Сихівський", 
    "Франківський", 
    "Шевченківський"
]
TOP_CATEGORIES = [
    "Несправний ліфт", 
    "Відкритий люк", 
    "Витік води", 
    "Відсутнє вуличне освітлення", 
    "Ями на дорозі"
]
OTHER_CATEGORY = "Інше (ввести вручну)"


# --- Main App ---
def predict_page():
    st.title("Lviv City Pulse: Прогноз виконання звернень")
    st.markdown("Введіть деталі вашого звернення, щоб отримати прогнозний час його виконання.")

    with st.form("prediction_form"):
        st.subheader("Деталі звернення")
        
        # Input fields
        district = st.selectbox("Оберіть район:", DISTRICTS)
        
        category_choice = st.selectbox("Оберіть категорію:", TOP_CATEGORIES + [OTHER_CATEGORY])
        
        custom_category = ""
        if category_choice == OTHER_CATEGORY:
            custom_category = st.text_input("Введіть вашу категорію:")

        submitted = st.form_submit_button("Отримати прогноз")

    if submitted:
        # Determine the final category
        final_category = custom_category if category_choice == OTHER_CATEGORY else category_choice
        
        if not final_category:
            st.warning("Будь ласка, введіть або оберіть категорію.")
            return

        with st.spinner("Отримуємо прогноз від моделей..."):
            try:
                payload = {
                    "district": district,
                    "category": final_category
                }
                response = requests.post(MODEL_API_URL, data=json.dumps(payload))
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

                predictions = response.json().get("predictions", {})
                
                st.subheader("Результати прогнозу (днів до виконання)")
                cols = st.columns(len(predictions))
                
                max_days = 0
                model_with_max_days = ""

                # Display metrics
                for idx, (model_name, days) in enumerate(predictions.items()):
                    with cols[idx]:
                        st.metric(label=model_name, value=f"{days:.1f} днів")
                    if days > max_days:
                        max_days = days
                        model_with_max_days = model_name

                # Highlight the most pessimistic prediction
                st.info(f"**Безпечна оцінка:** Найбільш песимістичний прогноз ({model_with_max_days}) становить **{max_days:.1f} днів**.", icon="🛡️")

            except requests.exceptions.RequestException as e:
                st.error(f"Не вдалося підключитися до сервісу моделей. Перевірте, чи він запущений. Помилка: {e}")
            except Exception as e:
                st.error(f"Сталася неочікувана помилка: {e}")

def about_page():
    st.title("Про проект")
    st.markdown("""
    **Lviv City Pulse** - це портфельний проект, розроблений для демонстрації навичок в MLOps та архітектурі програмного забезпечення.
    
    ### Архітектура
    Система побудована на основі мікросервісної архітектури з використанням Docker та складається з трьох основних компонентів:
    1.  **База даних (PostgreSQL):** Зберігає історичні дані про звернення громадян.
    2.  **Сервіс моделей (Python + FastAPI):** Надає API для тренування моделей та отримання прогнозів. Використовує Linear Regression, Random Forest, та XGBoost.
    3.  **Інтерфейс (Python + Streamlit):** Цей веб-додаток, який ви зараз використовуєте для взаємодії з системою.

    ### Мета
    Прогнозування часу, необхідного для вирішення звернень громадян до служби 1580 у Львові, на основі відкритих даних.
    """)

# --- Sidebar Navigation ---
st.sidebar.title("Навігація")
page = st.sidebar.radio("Оберіть сторінку", ["Прогноз", "Про проект"])

if page == "Прогноз":
    predict_page()
else:
    about_page()
