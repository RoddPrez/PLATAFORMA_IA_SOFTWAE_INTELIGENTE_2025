# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import streamlit as st
# from sklearn.datasets import load_iris
# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.preprocessing import image
# import pandas as pd
# import random
# import string
# import seaborn as sns
# import matplotlib.dates as mdates
# import plotly.express as px
# 
# # Config
# st.set_page_config(page_title="Plataforma AI Integrada", layout="wide")
# st.title("🧠 Plataforma Integrada de Modelos de IA")
# 
# menu = st.sidebar.selectbox("Selecciona un módulo", [
#     "🏁 Algoritmo Genético",
#     "🔎 Clasificación con Naïve Bayes",
#     "🧠 Clasificación con Red Neuronal",
#     "📷 Clasificación de Imágenes (Transfer Learning)",
#     "💬 Procesamiento de Lenguaje Natural"
# ])
# 
# # ---------------------------- GENÉTICOS ----------------------------------
# if menu == "🏁 Algoritmo Genético":
#     st.header("📈 Optimización de Portafolio con Algoritmo Genético")
# 
#     st.sidebar.header("⚙️ Parámetros de Simulación")
#     empresas = [
#         "Apple", "Microsoft", "Amazon", "Google", "Meta",
#         "Tesla", "NVIDIA", "Coca-Cola", "Pfizer", "McDonald's"
#     ]
#     selected_states = {}
# 
#     st.sidebar.markdown("### 📊 Selecciona los activos para invertir")
#     cols = st.sidebar.columns(2)
#     for i, empresa in enumerate(empresas):
#         col = cols[i % 2]
#         selected_states[empresa] = col.toggle(empresa, value=i < 4)
# 
#     selected_empresas = [k for k, v in selected_states.items() if v]
# 
#     if not selected_empresas:
#         st.warning("⚠️ Selecciona al menos un activo para continuar.")
#         st.stop()
# 
#     days = 250
#     np.random.seed(42)
#     base_prices = [170, 300, 140, 125, 290, 240, 500, 60, 45, 280]
#     start_date = pd.to_datetime("2023-01-01")
#     dates = pd.bdate_range(start=start_date, periods=days)
#     data = pd.DataFrame(index=dates)
# 
#     for i, empresa in enumerate(empresas):
#         drift = np.random.choice([0.0004, 0.0007, -0.0002])
#         noise = np.random.normal(0, 0.01, days)
#         returns = drift + noise
#         prices = base_prices[i] * np.cumprod(1 + returns)
#         data[empresa] = prices
# 
#     data = data[selected_empresas]
#     returns = data.pct_change().dropna()
#     mean_returns = returns.mean()
#     cov_matrix = returns.cov()
# 
#     st.subheader("📈 Precios históricos simulados (en USD)")
#     st.line_chart(data)
# 
#     st.write("### 🔁 Retornos esperados y matriz de covarianza")
# 
#     retorno_df = pd.DataFrame({"Retorno esperado": mean_returns})
#     st.dataframe(retorno_df)
# 
#     st.dataframe(cov_matrix)
# 
#     with st.expander("ℹ️ ¿Qué significan estos números? Haz clic para más información."):
#         st.markdown("""
#         - **Retorno esperado**: representa la ganancia promedio diaria estimada para cada activo seleccionado.
#           Por ejemplo, si un activo tiene un retorno esperado de `0.0012`, significa que en promedio aumenta un **0.12% diario**.
# 
#         - **Matriz de covarianza**: muestra cómo se mueven los activos **en conjunto**:
#             - Un valor **positivo** indica que los precios de dos activos tienden a subir o bajar juntos.
#             - Un valor **negativo** indica que cuando uno sube, el otro tiende a bajar.
#             - El valor en la **diagonal principal** (ej. `Apple` con `Apple`) representa la **varianza** del activo, es decir, qué tan volátil es individualmente.
# 
#         Esta información es crucial para calcular el riesgo y retorno total del portafolio, que es lo que optimiza el algoritmo genético.
#         """)
# 
#     pop_size = st.sidebar.slider("👥 Tamaño de población", 10, 300, 50)
#     generations = st.sidebar.slider("🔁 Generaciones", 10, 500, 100)
#     mutation_rate = st.sidebar.slider("🧬 Tasa de mutación", 0.0, 1.0, 0.1)
#     risk_aversion = st.sidebar.slider("⚖️ Aversión al riesgo (lambda)", 0.0, 10.0, 2.0)
# 
#     def portfolio_return(weights):
#         return np.dot(weights, mean_returns)
# 
#     def portfolio_risk(weights):
#         return np.dot(weights.T, np.dot(cov_matrix, weights))
# 
#     def fitness(weights):
#         ret = portfolio_return(weights)
#         risk = portfolio_risk(weights)
#         return ret - risk_aversion * risk
# 
#     def normalize(weights):
#         weights = np.abs(weights)
#         return weights / np.sum(weights)
# 
#     if st.button("🚀 Ejecutar Optimización"):
#         st.subheader("🔍 Ejecutando Algoritmo Genético...")
#         population = [normalize(np.random.rand(len(selected_empresas))) for _ in range(pop_size)]
#         best_scores = []
#         best_weights = None
#         best_fit = -np.inf
# 
#         for gen in range(generations):
#             scores = np.array([fitness(ind) for ind in population])
#             top_indices = np.argsort(scores)[-pop_size//2:]
#             selected = [population[i] for i in top_indices]
#             children = []
# 
#             while len(children) < pop_size - len(selected):
#                 idx1, idx2 = np.random.choice(len(selected), 2, replace=False)
#                 p1, p2 = selected[idx1], selected[idx2]
#                 child = normalize((p1 + p2) / 2)
# 
#                 if np.random.rand() < mutation_rate:
#                     mutation = np.random.normal(0, 0.05, len(selected_empresas))
#                     child = normalize(child + mutation)
# 
#                 children.append(child)
# 
#             population = selected + children
#             current_best = max(population, key=fitness)
#             current_fit = fitness(current_best)
# 
#             if current_fit > best_fit:
#                 best_fit = current_fit
#                 best_weights = current_best
# 
#             best_scores.append(best_fit)
# 
#         st.success(f"✅ Mejor Fitness: {best_fit:.5f}")
#         st.write("### 💾 Portafolio Óptimo:")
#         result_df = pd.DataFrame({
#             'Empresa': selected_empresas,
#             'Peso óptimo': best_weights
#         })
#         st.dataframe(result_df.set_index("Empresa"))
# 
#         st.subheader("📈 Evolución del Fitness")
#         fig1, ax1 = plt.subplots(figsize=(10, 4))
#         ax1.plot(best_scores, color='skyblue')
#         ax1.set_xlabel("Generaciones")
#         ax1.set_ylabel("Fitness")
#         ax1.set_title("Mejor Fitness por Generación")
#         ax1.grid(True, color='gray', linestyle='--', linewidth=0.5)
#         ax1.set_facecolor('black')
#         fig1.patch.set_facecolor('black')
#         ax1.tick_params(colors='white')
#         ax1.xaxis.label.set_color('white')
#         ax1.yaxis.label.set_color('white')
#         ax1.title.set_color('white')
#         st.pyplot(fig1)
# 
#         st.markdown("### 🧹 Distribución del Portafolio Óptimo")
#         pie_df = pd.DataFrame({
#             "Empresa": selected_empresas,
#             "Peso": best_weights
#         })
# 
#         pie_df = pie_df.dropna()
#         pie_df = pie_df[pie_df["Peso"] > 0]
#         pie_df = pie_df[pie_df["Empresa"].notna()]
#         pie_df = pie_df[pie_df["Empresa"] != "undefined"]
#         pie_df = pie_df.reset_index(drop=True)
# 
#         if pie_df.empty:
#             st.warning("⚠️ No hay datos válidos para mostrar el gráfico.")
#         else:
#             fig_pie = px.pie(
#                 pie_df,
#                 names="Empresa",
#                 values="Peso",
#                 hole=0.4,
#                 labels={"Empresa": "Empresa", "Peso": "Proporción"}
#             )
# 
#             fig_pie.update_traces(
#                 textinfo='label+percent',
#                 textfont_size=16,
#                 textposition='outside',
#                 marker=dict(
#                     line=dict(color='white', width=2)
#                 ),
#                 pull=[0.05]*len(pie_df)
#             )
# 
#             fig_pie.update_layout(
#                 showlegend=True,
#                 plot_bgcolor='rgba(0,0,0,0)',
#                 paper_bgcolor='rgba(0,0,0,0)',
#                 font=dict(color='white', size=14),
#                 title_font=dict(size=22, color='white'),
#                 hoverlabel=dict(
#                     bgcolor="black",
#                     font_size=14,
#                     font_color='white',
#                     font_family="Arial"
#                 )
#             )
# 
#             st.plotly_chart(fig_pie, use_container_width=True)
# 
#         st.markdown("### 📊 Detalle del Portafolio")
#         detalle_df = pie_df.copy()
#         detalle_df["Peso (%)"] = (detalle_df["Peso"] * 100).round(2).astype(str) + " %"
#         st.dataframe(detalle_df[["Empresa", "Peso (%)"]].set_index("Empresa"))
# 
#         st.markdown("""
#         ---
#         ✅ **Gen:** vector de pesos que suman 1
#         ✅ **Fitness:** retorno esperado - λ · varianza
#         ✅ **Selección:** top 50%
#         ✅ **Cruce:** promedio entre padres
#         ✅ **Mutación:** perturbación gaussiana
#         """)
# 
# 
# # ----------------------------- NAIVE BAYES --------------------------------
# # ----------------------------- NAIVE BAYES --------------------------------
# elif menu == "🔎 Clasificación con Naïve Bayes":
#     st.header("🔍 Clasificador de Abandono de Clientes - Naïve Bayes")
# 
#     st.sidebar.markdown("""
#     Este modelo predice si un cliente **ABANDONARÁ** o **PERMANECERÁ** en la empresa basada en su comportamiento y características demográficas.
# 
#     📊 Modelo entrenado con el dataset **BankChurners**
#     🧠 Método: Naïve Bayes Gaussiano
#     🗂️ Variables: Edad, género, ingresos, uso de la tarjeta, inactividad, entre otros.
# 
#     ---
#     """)
# 
#     import joblib
#     import pandas as pd
# 
#     modelo = joblib.load("naive_bayes_model.pkl")
#     encoders = joblib.load("label_encoders.pkl")
# 
#     with st.form("formulario_cliente"):
#         st.subheader("📝 Ingresar datos del cliente")
# 
#         edad = st.number_input("Edad del cliente", min_value=18, max_value=100, value=35)
#         st.caption("🔹 Edad del titular de la tarjeta de crédito.")
# 
#         genero = st.selectbox("Género", encoders["Gender"].classes_)
#         st.caption("🔹 Género registrado del cliente (según el banco).")
# 
#         dependientes = st.number_input("Número de dependientes", 0, 10, 1)
#         st.caption("🔹 Personas que dependen económicamente del cliente.")
# 
#         educacion = st.selectbox("Nivel educativo", encoders["Education_Level"].classes_)
#         st.caption("🔹 Nivel de estudios alcanzado por el cliente.")
# 
#         estado_civil = st.selectbox("Estado civil", encoders["Marital_Status"].classes_)
#         st.caption("🔹 Estado civil registrado.")
# 
#         ingresos = st.selectbox("Categoría de ingresos", encoders["Income_Category"].classes_)
#         st.caption("🔹 Rango de ingresos anuales estimados.")
# 
#         tarjeta = st.selectbox("Tipo de tarjeta", encoders["Card_Category"].classes_)
#         st.caption("🔹 Categoría de tarjeta de crédito asignada.")
# 
#         meses = st.slider("Meses como cliente", 6, 100, 36)
#         st.caption("🔹 Tiempo (en meses) que lleva siendo cliente.")
# 
#         relaciones = st.slider("Relaciones con el banco", 1, 10, 4)
#         st.caption("🔹 Cantidad de productos o servicios contratados con el banco.")
# 
#         inactividad = st.slider("Meses inactivo (últimos 12)", 0, 12, 1)
#         st.caption("🔹 Meses sin actividad registrada en el último año.")
# 
#         contactos = st.slider("Contactos con el banco (últimos 12)", 0, 10, 2)
#         st.caption("🔹 Número de veces que el cliente fue contactado por el banco.")
# 
#         limite_credito = st.number_input("Límite de crédito", min_value=0, value=5000)
#         st.caption("🔹 Monto máximo autorizado en la tarjeta.")
# 
#         saldo_revolvente = st.number_input("Saldo revolvente total", min_value=0, value=1000)
#         st.caption("🔹 Monto pendiente no pagado del último ciclo.")
# 
#         promedio_compra = st.number_input("Promedio disponible para comprar", min_value=0, value=4000)
#         st.caption("🔹 Crédito restante disponible para compras.")
# 
#         cambio_monto = st.number_input("Cambio total del monto (T4 a T1)", value=0.8)
#         st.caption("🔹 Variación porcentual del monto transaccionado entre trimestres.")
# 
#         total_transacciones = st.number_input("Monto total de transacciones", value=3000)
#         st.caption("🔹 Suma de todas las transacciones realizadas.")
# 
#         conteo_transacciones = st.slider("Cantidad total de transacciones", 0, 150, 60)
#         st.caption("🔹 Número de operaciones realizadas.")
# 
#         cambio_conteo = st.number_input("Cambio en cantidad de transacciones (T4 a T1)", value=0.7)
#         st.caption("🔹 Variación en el número de transacciones entre trimestres.")
# 
#         utilizacion_prom = st.slider("Índice promedio de uso del crédito", 0.0, 1.0, 0.2)
#         st.caption("🔹 Porcentaje promedio del crédito utilizado.")
# 
#         submit = st.form_submit_button("🔍 Predecir resultado")
# 
#     if submit:
#         nuevo_cliente = {
#             'Customer_Age': edad,
#             'Gender': genero,
#             'Dependent_count': dependientes,
#             'Education_Level': educacion,
#             'Marital_Status': estado_civil,
#             'Income_Category': ingresos,
#             'Card_Category': tarjeta,
#             'Months_on_book': meses,
#             'Total_Relationship_Count': relaciones,
#             'Months_Inactive_12_mon': inactividad,
#             'Contacts_Count_12_mon': contactos,
#             'Credit_Limit': limite_credito,
#             'Total_Revolving_Bal': saldo_revolvente,
#             'Avg_Open_To_Buy': promedio_compra,
#             'Total_Amt_Chng_Q4_Q1': cambio_monto,
#             'Total_Trans_Amt': total_transacciones,
#             'Total_Trans_Ct': conteo_transacciones,
#             'Total_Ct_Chng_Q4_Q1': cambio_conteo,
#             'Avg_Utilization_Ratio': utilizacion_prom
#         }
# 
#         input_df = pd.DataFrame([nuevo_cliente])
# 
#         for col in input_df.select_dtypes(include='object').columns:
#             input_df[col] = encoders[col].transform(input_df[col])
# 
#         pred = modelo.predict(input_df)[0]
#         prob = modelo.predict_proba(input_df)[0]
# 
#         st.subheader("🔎 Resultado de la predicción")
# 
#         if pred == 1:
#             st.error(f"🚨 El modelo predice que este cliente probablemente **ABANDONARÁ** la empresa.")
#             st.markdown(f"**📉 Probabilidad de abandono:** `{prob[1]:.2%}`\n\n**📈 Probabilidad de permanencia:** `{prob[0]:.2%}`")
#             st.info("💡 Recomendación: Se puede contactar al cliente con una campaña de retención o beneficios.")
#         else:
#             st.success(f"✅ Este cliente probablemente **PERMANECERÁ** con la empresa.")
#             st.markdown(f"**📈 Probabilidad de permanencia:** `{prob[0]:.2%}`\n\n**📉 Probabilidad de abandono:** `{prob[1]:.2%}`")
#             st.info("👍 No se requieren acciones inmediatas.")
# 
# # ------------------------------ RED NEURONAL -------------------------------
# elif menu == "🧠 Clasificación con Red Neuronal":
#     st.header("🧠 Proyección de comportamiento del cliente (Red Neuronal)")
# 
#     st.sidebar.markdown("""
#     Este módulo permite realizar una predicción sobre si un cliente **aceptará un depósito a plazo**, usando un modelo neuronal entrenado con datos reales.
# 
#     ✨ Puedes modificar los valores como si se tratara de un **cliente actual o potencial**, y observar cómo cambian las probabilidades de conversión.
#     """)
# 
#     import joblib
#     from tensorflow.keras.models import load_model
# 
#     model = load_model("bank_mlp_model.h5")
#     scaler = joblib.load("scaler_bank.pkl")
#     columnas_entrenamiento = joblib.load(open("columnas_entrenamiento.pkl", "rb"))
# 
#     st.subheader("📋 Características del cliente (Simulación)")
# 
#     age = st.number_input("Edad del cliente", min_value=18, max_value=100, value=35)
#     st.caption("Edad proyectada del cliente en años. Puede influir en el nivel de riesgo o madurez financiera.")
# 
#     job = st.selectbox("Ocupación", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
#                                      'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed'])
#     st.caption("Sector laboral del cliente simulado. Algunos sectores pueden tener mayor tasa de respuesta.")
# 
#     marital = st.selectbox("Estado civil", ['married', 'single', 'divorced'])
#     st.caption("Estado civil proyectado. Este dato puede estar correlacionado con prioridades financieras.")
# 
#     education = st.selectbox("Nivel educativo", ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
#                                                  'professional.course', 'university.degree'])
#     st.caption("Nivel educativo estimado. Puede relacionarse con la propensión a invertir.")
# 
#     default = st.selectbox("¿Historial de incumplimiento crediticio?", ['no', 'yes'])
#     st.caption("Historial simulado de crédito fallido. Afecta la confianza y elegibilidad del cliente.")
# 
#     balance = st.number_input("Balance promedio anual (€)", value=1000)
#     st.caption("Promedio de saldo anual disponible en la cuenta del cliente. Refleja capacidad económica.")
# 
#     housing = st.selectbox("¿Tiene préstamo hipotecario activo?", ['yes', 'no'])
#     st.caption("Indica si el cliente simulado posee actualmente una hipoteca.")
# 
#     loan = st.selectbox("¿Tiene préstamo personal activo?", ['yes', 'no'])
#     st.caption("Condición de préstamo personal. Puede indicar carga financiera adicional.")
# 
#     contact = st.selectbox("Medio de contacto preferido", ['cellular', 'telephone'])
#     st.caption("Canal más efectivo para contactar al cliente. Algunas campañas rinden mejor en celular.")
# 
#     day = st.number_input("Día del último contacto proyectado", min_value=1, max_value=31, value=15)
#     st.caption("Día del mes en que se realizaría el próximo contacto estimado.")
# 
#     month = st.selectbox("Mes del contacto proyectado", ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
#                                                          'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
#     st.caption("Mes en que se tiene pensado contactar al cliente. La época del año influye en la conversión.")
# 
#     duration = st.number_input("Duración estimada del contacto (segundos)", min_value=0, value=180)
#     st.caption("Tiempo estimado de conversación. Debe ser igual o mayor a 0.")
# 
#     campaign = st.number_input("Cantidad de contactos previstos en esta campaña", min_value=1, value=2)
#     st.caption("Número total de interacciones planeadas con este cliente en la campaña actual. Mínimo 1.")
# 
#     pdays = st.number_input("Días desde el último contacto anterior", value=-1)
#     st.caption("Días transcurridos desde la última campaña. Puede ser -1 (nunca contactado) o ≥ 1.")
#     if pdays != -1 and pdays < 1:
#         st.warning("⚠️ 'Días desde el último contacto anterior' debe ser -1 o un valor mayor o igual a 1.")
# 
#     previous = st.number_input("Cantidad de contactos previos en otras campañas", min_value=0, value=0)
#     st.caption("Número de veces que este cliente fue contactado en campañas pasadas. Mínimo 0.")
# 
#     poutcome = st.selectbox("Resultado de la campaña anterior", ['unknown', 'other', 'failure', 'success'])
#     st.caption("Desenlace anterior del contacto con este cliente. Puede influir en la respuesta futura.")
# 
#     if st.button("🔮 Predecir comportamiento proyectado"):
#         # Validaciones adicionales
#         if pdays != -1 and pdays < 1:
#             st.error("❌ 'Días desde el último contacto anterior' debe ser -1 o ≥ 1.")
#         else:
#             nuevo_cliente = {
#                 "age": age,
#                 "job": job,
#                 "marital": marital,
#                 "education": education,
#                 "default": default,
#                 "balance": balance,
#                 "housing": housing,
#                 "loan": loan,
#                 "contact": contact,
#                 "day": day,
#                 "month": month,
#                 "duration": duration,
#                 "campaign": campaign,
#                 "pdays": pdays,
#                 "previous": previous,
#                 "poutcome": poutcome
#             }
# 
#             df = pd.DataFrame([nuevo_cliente])
#             df = pd.get_dummies(df)
# 
#             for col in columnas_entrenamiento:
#                 if col not in df.columns:
#                     df[col] = 0
#             df = df[columnas_entrenamiento]
# 
#             X_scaled = scaler.transform(df)
#             prob = model.predict(X_scaled)[0][0]
#             clasificacion = "✅ **Sí, aceptará el depósito.**" if prob > 0.5 else "❌ **No aceptará el depósito.**"
# 
#             st.subheader("📊 Resultado de la predicción")
#             st.write(f"**Probabilidad de aceptación estimada:** {prob:.2%}")
#             st.markdown(clasificacion)
# 
# # ---------------------------- TRANSFER LEARNING ---------------------------
# # ---------------------------- RECONOCIMIENTO DE EMOCIONES ---------------------------
# elif menu == "📷 Clasificación de Imágenes (Transfer Learning)":
#     st.header("😊 Clasificación de Emociones Faciales con Transfer Learning")
# 
#     st.sidebar.markdown("""
#         Este módulo utiliza un modelo previamente entrenado mediante **Transfer Learning** con la arquitectura **EfficientNetB0**, adaptado para la tarea de reconocimiento de emociones faciales a partir de imágenes.
# 
#     🔍 **Transfer Learning aplicado**:
#     - Se utilizó la arquitectura **EfficientNetB0** preentrenada en ImageNet.
#     - Se reemplazaron las capas superiores por una capa densa de salida con **7 clases** (una por emoción).
#     - Solo se entrenaron las capas superiores, manteniendo congeladas las capas base para aprovechar el conocimiento visual general aprendido en grandes volúmenes de imágenes.
# 
#     📚 **Dataset utilizado**: [FER2013 (Facial Expression Recognition 2013) – Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
#     - Conjunto de datos con **35,887 imágenes en escala de grises** de **48x48 píxeles**, clasificadas en 7 emociones:
#       - 😠 **Enojo** (`Angry`)
#       - 🤢 **Asco** (`Disgust`)
#       - 😨 **Miedo** (`Fear`)
#       - 😊 **Felicidad** (`Happy`)
#       - 😢 **Tristeza** (`Sad`)
#       - 😲 **Sorpresa** (`Surprise`)
#       - 😐 **Neutral** (`Neutral`)
# 
#     💡 El modelo resultante es capaz de identificar la **emoción predominante** en rostros humanos con alta precisión, y visualizar la distribución de todas las emociones presentes en la imagen.
# 
#     📦 Este modelo ya está integrado en la plataforma, listo para ser utilizado con imágenes en formato JPG o PNG.
#     """)
# 
#     from tensorflow.keras.models import load_model
#     from tensorflow.keras.preprocessing import image
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from PIL import Image
# 
#     # Cargar modelo FER2013
#     model = load_model("fer2013_emotion_model_1.h5")
#     emotion_labels_en = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
#     emotion_labels_es = ['Enojo', 'Asco', 'Miedo', 'Felicidad', 'Tristeza', 'Sorpresa', 'Neutral']
# 
#     st.subheader("📸 Sube una imagen de un rostro humano")
#     uploaded_image = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png"])
# 
#     if uploaded_image:
#         img = Image.open(uploaded_image).convert("RGB")
#         st.image(img, caption="Imagen cargada", use_column_width=True)
# 
#         # Preprocesar imagen (48x48)
#         img_resized = img.resize((48, 48))
#         img_array = image.img_to_array(img_resized)
#         img_array = img_array / 255.0
#         img_array = np.expand_dims(img_array, axis=0)
# 
#         # Predicción
#         pred = model.predict(img_array)
#         idx = np.argmax(pred)
#         emotion = emotion_labels_es[idx]
#         confidence = np.max(pred) * 100
# 
#         st.success(f"😊 Emoción detectada: **{emotion}** con {confidence:.2f}% de confianza")
# 
#         st.subheader("📊 Distribución de emociones")
#         st.bar_chart({es: float(score) for es, score in zip(emotion_labels_es, pred[0])})
# 
# # ------------------------------ NLP --------------------------------------
# elif menu == "💬 Procesamiento de Lenguaje Natural":
#     st.header("💬 Análisis de Sentimientos en Español con Transfer Learning")
# 
#     st.markdown("""
#     Este módulo analiza el sentimiento de un texto usando un modelo basado en **BETO (BERT en español)** ya entrenado.
# 
#     - 🟥 Negativo
#     - 🟨 Neutro
#     - 🟩 Positivo
#     """)
# 
#     import tensorflow as tf
#     import numpy as np
#     from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
# 
#     @st.cache_resource
#     def load_local_model():
#         model_path = "modelo_sentimientos"  # carpeta donde están los archivos del modelo
#         model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#         return model, tokenizer
# 
#     model, tokenizer = load_local_model()
#     labels = ["NEGATIVO", "NEUTRO", "POSITIVO"]
# 
#     texto = st.text_area("✍️ Escribe un comentario o reseña en español")
# 
#     if texto:
#         inputs = tokenizer(texto, return_tensors="tf", truncation=True, padding="max_length", max_length=128)
# 
#         with st.spinner("🔎 Analizando sentimiento..."):
#             outputs = model(**inputs)
#             probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
# 
#         pred_label = labels[np.argmax(probs)]
#         confidence = np.max(probs) * 100
# 
#         st.success(f"🔍 Sentimiento detectado: **{pred_label}** con {confidence:.2f}% de confianza")
# 
#         st.subheader("📊 Distribución de sentimientos")
#         st.bar_chart({label: float(prob) for label, prob in zip(labels, probs)})
#
