import streamlit as st
import requests
import pandas as pd
import aiohttp
import asyncio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
import shap
import lightgbm
from lightgbm import LGBMClassifier
from streamlit_shap import st_shap
import base64
import joblib
from io import BytesIO  # Import BytesIO for loading the model content
import traceback
import nest_asyncio
nest_asyncio.apply()
import pickle
import nest_asyncio
import ast

# Замените URL на ваш FastAPI сервер
fastapi_url = "https://score-ff2bfc305853.herokuapp.com"

# Функция для загрузки данных с учетом offset и limit
@st.cache_data(show_spinner=False)
def load_data_batch(offset: int, limit: int, show_spinner=False):
    try:
        load_data_url = f"{fastapi_url}/load_data"
        params = {'offset': offset, 'limit': limit}
        response = requests.get(load_data_url, params=params)
        response.raise_for_status()  # Проверяем, что запрос прошел успешно
        df_batch_response = response.json()
        return pd.DataFrame(df_batch_response)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")



@st.cache_resource
def load_model():
    model_url = "https://score-ff2bfc305853.herokuapp.com/load_model"

    try:
        # Получаем содержимое модели по URL
        response = requests.get(model_url)
        response.raise_for_status()
        model_bytes = response.content

        # Загружаем модель из бинарных данных с использованием joblib
        model = joblib.load(BytesIO(model_bytes))

        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise ValueError(f"Error loading model: {str(e)}") from e

@st.cache_data(show_spinner=False)
def compute_shap_values(_explainer_shap, df):
    return _explainer_shap.shap_values(df)
    
def reset_app():
    # Сброс всех значений в session_state, которые вы используете
    keys_to_reset = ['option1', 'option2', 'option3', 'selected_client_id']
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    # Очистка содержимого экрана может быть достигнута через перезапуск приложения
    st.experimental_rerun()

def main():
    col1, col2 = st.columns([2, 10])
    

    offset = 0
    limit = 8000  
    model = load_model()
    explainer_shap = shap.TreeExplainer(model)
    if "df_test" not in st.session_state:
        st.session_state.df_test = load_data_batch(offset, limit)

    if "is_initial_load" not in st.session_state:
        col2.markdown('<p style="font-size:36px; font-weight:bold;">Bienvenue dans l\'application<br>Prêt à dépenser!</p>', unsafe_allow_html=True)
        col1.image('https://raw.githubusercontent.com/MarinaSupiot/projet_7/main/Prets_a_depenser.png', width=110)
        st.session_state.is_initial_load = True

    while offset < 8000:
        offset += limit  
        next_batch = load_data_batch(offset, limit)

        if next_batch is not None and next_batch.shape[0] == 0:
            break

        st.session_state.df_test = pd.concat([st.session_state.df_test, next_batch], ignore_index=True)


    if 'selected_client' not in st.session_state:
        st.session_state.selected_client = ''
        st.session_state.option1 = False
        st.session_state.option2 = False
        st.session_state.option3 = False
    client_list = list(st.session_state.df_test['SK_ID_CURR'].unique())                

    option = st.sidebar.selectbox(
        "Entrer le numero de client",
        client_list,
        index=0,
        placeholder="Select le numero de client",
        format_func=lambda x: '' if x == '' else x,  # Форматирование для отображения пустой строки по умолчанию
    )

    if option and option != st.session_state.selected_client:
        st.session_state.selected_client = option        
        st.session_state.option1 = False
        st.session_state.option2 = False
        st.session_state.option3 = False

    option1_key = f"option1_{st.session_state.selected_client}"
    option2_key = f"option2_{st.session_state.selected_client}"
    option3_key = f"option3_{st.session_state.selected_client}"

  


    if st.sidebar.button("Statut de la demande"):
        filtered_df = st.session_state.df_test[st.session_state.df_test['SK_ID_CURR'] == option]

        if not filtered_df.empty:
            input_data = filtered_df.drop(columns=['SK_ID_CURR']).values

            if len(input_data.shape) == 2 and input_data.shape[0] > 0 and input_data.shape[1] > 0:

                if model is not None:
                    prediction_probabilities = model.predict_proba(input_data)
                    prediction = (prediction_probabilities[:, 1] > 0.556).astype(int)
                    


                    if prediction[0] == 1:
                        st.markdown('<p style="font-size:40px; text-align:center; color:#00008B; font-weight: bold; ">Désolé, votre demande a été rejetée.</p>', unsafe_allow_html=True)
                    else:
                        st.markdown('<p style="font-size:40px; text-align:center; color:#00008B; font-weight: bold; ">Félicitation ! Votre demande est acceptée !</p>', unsafe_allow_html=True)
                        st.balloons()

                    probability_value = prediction_probabilities[0][1]
                else:
                    st.error("Failed to load the model.")

                red_color = '#FF0000'
                green_color = '#00FF00'
                green_range = (0, 0.556)
                red_range = (0.556, 1)

                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=probability_value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'shape': "angular",
                        'axis': {'range': [None, 1]},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': red_range, 'color': red_color},
                            {'range': green_range, 'color': green_color},
                        ],
                    }
                ))

                fig.update_layout(
                    autosize=False,
                    margin=dict(l=20, r=20, b=20, t=40),
                )

                st.plotly_chart(fig)
            else:
                st.write("Erreur : Les données d'entrée sont vides ou n'ont pas deux dimensions.")

    st.markdown('<div id="section_option2"></div>', unsafe_allow_html=True)
    
    st.session_state.option1 = st.sidebar.checkbox("Facteurs d'influence globaux", value=st.session_state.option1, key=option1_key)
    st.session_state.option2 = st.sidebar.checkbox("Facteurs d'influence individuels", value=st.session_state.option2, key=option2_key)
    st.session_state.option3 = st.sidebar.checkbox("Comparaison avec les autres clients", value=st.session_state.option3, key=option3_key)

    spinner_text = st.empty()
    spinner = st.spinner()
# Проверка состояния чекбокса option1
    if st.session_state.option1:
            st.title("Facteurs d'influence globaux")
            df_test_indexed = st.session_state.df_test.set_index('SK_ID_CURR')

            # Проверка, что shap_values не является None
            shap_values = compute_shap_values(explainer_shap, df_test_indexed)
            if shap_values is not None:
                num_columns = st.slider("Sélectionnez le nombre de colonnes", min_value=1, max_value=80, value=10)

                # Запустите Spinner перед выполнением долгой операции
                with spinner:
                    # Обновление текста Spinner
                    #spinner_text.text("Calcul en cours...")


                    st_shap(shap.summary_plot(shap_values[1], df_test_indexed, show=False, max_display=num_columns))
                spinner_text.text("")

            else:
                st.write("Erreur : Les valeurs SHAP ne sont pas disponibles.")   

    if st.session_state.option2:

                st.title("Facteurs d'influence individuels")
                df_test_indexed = st.session_state.df_test.set_index('SK_ID_CURR')
                idx_option2 = df_test_indexed.index.get_loc(option)

                # Проверка, что shap_values не является None
                shap_values_option2 = compute_shap_values(explainer_shap, df_test_indexed)
                if shap_values_option2 is not None:
                    # Add a unique key to the slider widget
                    num_columns_2 = st.slider("Sélectionnez le nombre de colonnes", min_value=1, max_value=80, value=10, key="slider_option2")
                    # Запустите Spinner перед выполнением долгой операции
                    with st.spinner("Calcul en cours..."):
                        # Обновление текста Spinner
                        #spinner_text.text("Calcul en cours...")

                        st.text("")
                        left_margin, _, right_margin = st.columns([1, 5, 20])
                        left_margin.text("")
                        st_shap(shap.plots._waterfall.waterfall_legacy(explainer_shap.expected_value[1], shap_values_option2[1][idx_option2, :], df_test_indexed.iloc[idx_option2, :],  max_display=num_columns_2))
                        right_margin.text("")


                else:
                    st.write("Erreur : Les valeurs SHAP ne sont pas disponibles.")   

# Check Option 3 is Selected
    if st.session_state.option3:
            st.markdown('<a href="#section_option2">Aller aux facteurs d\'influence individuels</a>', unsafe_allow_html=True)
            st.title("Analyse comparée avec les autres clients")

            # Значения Shapley для конкретного клиента
            client_shap_values = shap_values_option2[0][idx_option2]

            # Выберите, например, 10 лучших функций
            top_features_indices = np.abs(shap_values_option2[0][idx_option2]).argsort()[-10:][::-1]
            top_features_indices = top_features_indices + 1
            top_features_names = st.session_state.df_test.columns[top_features_indices]

            # Данные для выбранных функций для всех клиентов
            data_for_top_features = st.session_state.df_test[top_features_names]

            import seaborn as sns


            # Выбор типа графика
            plot_type = st.radio("Choisir le type de graphique :", ("Histogramme", "Boxplot"))

            # Вычислите количество строк и столбцов в фигуре
            num_rows = 5
            num_columns = 2  # Всегда 2 графика в каждой строке

            # Создайте фигуру и ось
            fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 20))

            # Переменная для отслеживания текущей позиции в фигуре
            current_row, current_column = 0, 0

            for i, feature_name in enumerate(top_features_names):
                if plot_type == "Histogramme":
                    axs[current_row, current_column].hist(data_for_top_features[feature_name], bins=30, color='skyblue', label='All Clients', density=True)
                    axs[current_row, current_column].axvline(x=data_for_top_features.loc[idx_option2, feature_name], color='red', linestyle='dashed', linewidth=2, label='Selected Client')
                    axs[current_row, current_column].set_title(f'Distribution of {feature_name}\nfor All Clients vs Selected Client')
                    axs[current_row, current_column].set_xlabel(feature_name)
                    axs[current_row, current_column].set_ylabel('Density')
                    axs[current_row, current_column].legend()

                elif plot_type == "Boxplot":
                    # Постройте боксплот для каждой функции
                    sns.boxplot(x=data_for_top_features[feature_name], color='lightblue', ax=axs[current_row, current_column])

                    # Укажите место выбранного клиента на каждом боксплоте
                    axs[current_row, current_column].scatter(x=[0], y=[data_for_top_features.loc[idx_option2, feature_name]], color='red', marker='o', label='Selected Client')

                    # Настройте оси и метки
                    axs[current_row, current_column].set_title(f'Box Plot of {feature_name}\nfor All Clients vs Selected Client')
                    axs[current_row, current_column].set_xlabel(feature_name)
                    axs[current_row, current_column].set_ylabel('Value')
                    axs[current_row, current_column].legend()

                    # Установите одинаковый масштаб осей для всех боксплотов
                    axs[current_row, current_column].set_ylim(axs[0, 0].get_ylim())

                # Перейдите к следующему столбцу
                current_column += 1
                if current_column == num_columns:
                    current_row += 1
                    current_column = 0

            # Убедитесь, что все подписи осей видны
            plt.tight_layout()

            # Отобразите график
            st.pyplot(fig, clear_figure=True)                      
    
        

        



# ... (rest of the code)

if __name__ == "__main__":
    main()
