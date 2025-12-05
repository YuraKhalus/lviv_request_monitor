import streamlit as st
import requests
import pandas as pd
import json
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(page_title="Lviv City Pulse", page_icon="🏙️", layout="wide")

# --- API Constants ---
API_BASE_URL = "http://model-api:8000"
PREDICT_URL = f"{API_BASE_URL}/predict"
ACTUAL_URL = f"{API_BASE_URL}/actual"
PERFORMANCE_URL = f"{API_BASE_URL}/performance"

# --- UI Constants ---
DISTRICTS = ["Галицький район", "Залізничний район", "Личаківський район", "Сихівський район", "Франківський район", "Шевченківський район"]
TOP_CATEGORIES = [
    "Аварійна ситуація з системою електропостачання у житловому будинку", 
    "Порушення правил паркування",
    "Питання оплати та надання послуг",
    "Несправний (зупинений) ліфт житлового будинку",
    "Відсутня подача холодної води у житловому будинку",
    "Скарга на комунальні підприємства",
    "Порушення графіку руху громадського транспорту",
    "Ями, вибоїни в асфальтовому покритті проїжджої частини",
    "Відсутнє гаряче водопостачання (недавно) житлового будинку",
    "Інші порушення правил перевезення громадським транспортом",
    "Застрягання кабіни ліфта",
    "Аварійна ситуація з системою електропостачання у квартирі",
    "Інші проблеми з порядком на дорогах та громадських територіях",
    "Водій проігнорував зупинку громадського транспорту",
    "Відсутнє опалення по стояку житлового будинку",
    "Відсутнє опалення по житловому будинку",
    "Відсутнє зовнішнє освітлення",
    "Інші проблеми по обслуговуванню будинку",
    "Прорив водопровідних мереж (витік на вулиці)",
    "Не прибрана прибудинкова територія від сміття чи листя"
]
OTHER_CATEGORY = "Інше (ввести вручну)"

def create_gauge_chart(value):
    """Creates a Plotly gauge chart for urgency."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Оцінка терміновості (днів)"},
        gauge={
            'axis': {'range': [None, 15], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "black"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 3], 'color': 'lightgreen'},
                {'range': [3, 7], 'color': 'yellow'},
                {'range': [7, 15], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }))
    fig.update_layout(height=300)
    return fig

# --- Page 1: Prediction Page ---
def render_prediction_page():
    st.title("Lviv City Pulse: Прогноз виконання звернень")
    st.markdown("Введіть деталі, щоб отримати прогнозний час виконання вашого звернення.")

    with st.form("prediction_form"):
        district = st.selectbox("Оберіть район:", DISTRICTS)
        category_choice = st.selectbox("Оберіть категорію:", TOP_CATEGORIES + [OTHER_CATEGORY])
        custom_category = st.text_input("Введіть вашу категорію:", key="custom_cat") if category_choice == OTHER_CATEGORY else ""
        submitted = st.form_submit_button("Отримати прогноз")

    if submitted:
        final_category = custom_category if category_choice == OTHER_CATEGORY else category_choice
        if not final_category:
            st.warning("Будь ласка, введіть або оберіть категорію."); return

        payload = {"district": district, "category": final_category}
        with st.spinner("Отримуємо прогноз..."):
            try:
                # --- Get Prediction and Actual Case Data ---
                predict_resp = requests.post(PREDICT_URL, data=json.dumps(payload))
                predict_resp.raise_for_status()
                predictions = predict_resp.json().get("predictions", {})

                actual_resp = requests.post(ACTUAL_URL, data=json.dumps(payload))
                actual_resp.raise_for_status()
                actual_days = actual_resp.json().get("actual_days")

                # --- Display Main Metrics (preserved) ---
                st.subheader("🤖 Результати прогнозу (днів до виконання)")
                cols = st.columns(len(predictions))
                max_days, model_with_max_days = 0, ""
                for idx, (model, days) in enumerate(predictions.items()):
                    with cols[idx]:
                        st.metric(label=model, value=f"{days:.1f} днів")
                    if days > max_days: max_days, model_with_max_days = days, model
                
                # --- Display Text Outputs (preserved) ---
                st.success(f"**Безпечна оцінка:** Найбільш песимістичний прогноз ({model_with_max_days}) становить **{max_days:.1f} днів**.", icon="🛡️")
                if actual_days is not None:
                    st.info(f"**Для довідки:** Випадковий реальний випадок з такими ж параметрами було вирішено за **{int(actual_days)} днів**.", icon="📚")
                else: 
                    st.warning("Не знайдено реальних історичних випадків для порівняння.", icon="⚠️")

                st.markdown("---") 

                # --- Display Gauge Chart ---
                st.subheader("Індикатор терміновості")
                st.caption("Показує, наскільки швидко це питання зазвичай вирішується, порівняно з міськими стандартами.")
                gauge_value = predictions.get("XGBoost", 0)
                st.plotly_chart(create_gauge_chart(gauge_value), use_container_width=True)

            except requests.exceptions.RequestException as e:
                st.error(f"Не вдалося підключитися до сервісу моделей. Перевірте, чи він запущений. Помилка: {e}")
            except Exception as e:
                st.error(f"Сталася неочікувана помилка: {e}")

# --- Page 2: Model Analytics Page ---
def render_analytics_page():
    st.title("Аналітика продуктивності моделей")
    st.markdown("Візуалізація порівняння реальних значень та прогнозів моделей на тестовому наборі даних.")
    try:
        with st.spinner("Завантаження даних..."):
            response = requests.get(PERFORMANCE_URL)
            response.raise_for_status()
            df = pd.DataFrame(response.json())
            st.subheader("Порівняння 'Реальність vs. Прогноз'")
            st.line_chart(df)
            st.subheader("Таблиця з даними")
            st.dataframe(df)
    except requests.exceptions.RequestException:
        st.error("Не вдалося завантажити дані. Переконайтеся, що моделі були навчені.")
    except Exception as e:
        st.error(f"Сталася помилка: {e}")

# --- Main App Navigation ---
st.sidebar.title("Навігація")
page_options = ["Прогнозування", "Аналітика Моделей"]
selected_page = st.sidebar.radio("Оберіть сторінку:", page_options)

if selected_page == "Прогнозування":
    render_prediction_page()
elif selected_page == "Аналітика Моделей":
    render_analytics_page()
