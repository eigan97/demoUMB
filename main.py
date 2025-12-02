import streamlit as st
import tensorflow as tf
import numpy as np

# Diccionario de traducción de nombres de clases al español
class_names_es = {
    'Apple___Apple_scab': 'Manzana - Sarna del manzano',
    'Apple___Black_rot': 'Manzana - Podredumbre negra',
    'Apple___Cedar_apple_rust': 'Manzana - Roya del cedro',
    'Apple___healthy': 'Manzana - Saludable',
    'Blueberry___healthy': 'Arándano - Saludable',
    'Cherry_(including_sour)___Powdery_mildew': 'Cereza - Oídio',
    'Cherry_(including_sour)___healthy': 'Cereza - Saludable',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Maíz - Mancha foliar por Cercospora',
    'Corn_(maize)___Common_rust_': 'Maíz - Roya común',
    'Corn_(maize)___Northern_Leaf_Blight': 'Maíz - Tizón foliar del norte',
    'Corn_(maize)___healthy': 'Maíz - Saludable',
    'Grape___Black_rot': 'Uva - Podredumbre negra',
    'Grape___Esca_(Black_Measles)': 'Uva - Esca (Sarampión negro)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Uva - Tizón foliar (Isariopsis)',
    'Grape___healthy': 'Uva - Saludable',
    'Orange___Haunglongbing_(Citrus_greening)': 'Naranja - Huanglongbing (Enverdecimiento de los cítricos)',
    'Peach___Bacterial_spot': 'Durazno - Mancha bacteriana',
    'Peach___healthy': 'Durazno - Saludable',
    'Pepper,_bell___Bacterial_spot': 'Pimiento - Mancha bacteriana',
    'Pepper,_bell___healthy': 'Pimiento - Saludable',
    'Potato___Early_blight': 'Papa - Tizón temprano',
    'Potato___Late_blight': 'Papa - Tizón tardío',
    'Potato___healthy': 'Papa - Saludable',
    'Raspberry___healthy': 'Frambuesa - Saludable',
    'Soybean___healthy': 'Soya - Saludable',
    'Squash___Powdery_mildew': 'Calabaza - Oídio',
    'Strawberry___Leaf_scorch': 'Fresa - Quemadura foliar',
    'Strawberry___healthy': 'Fresa - Saludable',
    'Tomato___Bacterial_spot': 'Tomate - Mancha bacteriana',
    'Tomato___Early_blight': 'Tomate - Tizón temprano',
    'Tomato___Late_blight': 'Tomate - Tizón tardío',
    'Tomato___Leaf_Mold': 'Tomate - Moho foliar',
    'Tomato___Septoria_leaf_spot': 'Tomate - Mancha foliar por Septoria',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Tomate - Ácaros (Araña roja de dos manchas)',
    'Tomato___Target_Spot': 'Tomate - Mancha objetivo',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomate - Virus del rizado amarillo de la hoja',
    'Tomato___Tomato_mosaic_virus': 'Tomate - Virus del mosaico',
    'Tomato___healthy': 'Tomate - Saludable'
}

# Predicción del modelo TensorFlow
def model_prediction(test_image):
    model  = tf.keras.models.load_model('./model/trained_model.h5')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) # Convertir imagen única a un batch
    prediction = model.predict(input_arr, verbose=0)
    result_index = np.argmax(prediction)
    confidence = float(prediction[0][result_index])
    return result_index, confidence

# Barra lateral
st.sidebar.title("Panel de Control")
app_mode = st.sidebar.selectbox("Seleccionar Página",["Inicio","Acerca de","Reconocimiento de Enfermedades"])

# Página de Inicio
if(app_mode=="Inicio"):
    st.header("SISTEMA DE RECONOCIMIENTO DE ENFERMEDADES EN PLANTAS")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    ¡Bienvenido al Sistema de Reconocimiento de Enfermedades en Plantas! 🌿🔍
    
    Nuestra misión es ayudar a identificar enfermedades en plantas de manera eficiente. Sube una imagen de una planta y nuestro sistema la analizará para detectar cualquier signo de enfermedad. ¡Juntos, protejamos nuestros cultivos y aseguremos una cosecha más saludable!

    ### ¿Cómo Funciona?
    1. **Subir Imagen:** Ve a la página de **Reconocimiento de Enfermedades** y sube una imagen de una planta con sospecha de enfermedades.
    2. **Análisis:** Nuestro sistema procesará la imagen utilizando algoritmos avanzados para identificar posibles enfermedades.
    3. **Resultados:** Visualiza los resultados y recomendaciones para tomar medidas.

    ### ¿Por Qué Elegirnos?
    - **Precisión:** Nuestro sistema utiliza técnicas de aprendizaje automático de última generación para una detección precisa de enfermedades.
    - **Fácil de Usar:** Interfaz simple e intuitiva para una experiencia de usuario fluida.
    - **Rápido y Eficiente:** Recibe resultados en segundos, permitiendo una toma de decisiones rápida.

    ### Comenzar
    Haz clic en la página de **Reconocimiento de Enfermedades** en la barra lateral para subir una imagen y experimentar el poder de nuestro Sistema de Reconocimiento de Enfermedades en Plantas!

    ### Acerca de Nosotros
    Conoce más sobre el proyecto, nuestro equipo y nuestros objetivos en la página **Acerca de**.
""")

# Página Acerca de
elif(app_mode=="Acerca de"):
    st.header("Acerca de")
    st.markdown("""
    #### Acerca del Conjunto de Datos
    Este conjunto de datos fue recreado utilizando aumento de datos offline a partir del conjunto de datos original. El conjunto de datos original se puede encontrar en este repositorio de github. Este conjunto de datos consiste en aproximadamente 87K imágenes RGB de hojas de cultivos sanas y enfermas que están categorizadas en 38 clases diferentes. El conjunto de datos total está dividido en una proporción 80/20 de entrenamiento y validación preservando la estructura de directorios. Un nuevo directorio que contiene 33 imágenes de prueba fue creado posteriormente para propósitos de predicción.
    #### Contenido
    1. Entrenamiento (70295 imágenes)
    2. Validación (17572 imágenes)
    3. Prueba (33 imágenes)
""")
    
# Página de Reconocimiento de Enfermedades
elif(app_mode=="Reconocimiento de Enfermedades"):
    st.header("Reconocimiento de Enfermedades")
    test_image = st.file_uploader("Elige una Imagen:")
    if(st.button("Mostrar Imagen")):
        st.image(test_image,use_column_width=True)
    # Botón de Predicción
    if(st.button("Predecir")):
        with st.spinner("Por favor espera.."):
            st.write("Nuestra Predicción")
            result_index, confidence = model_prediction(test_image)
            # Definir Clases (mantener en inglés para el modelo)
            class_name = ['Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy']
            # Obtener el nombre en inglés y traducirlo al español
            class_name_en = class_name[result_index]
            class_name_spanish = class_names_es.get(class_name_en, class_name_en)
            st.success("El modelo predice que es: **{}**".format(class_name_spanish))
            st.info("Confianza de la predicción: **{:.2%}**".format(confidence))
            
            # Alerta si la confianza es baja
            if confidence < 0.5:
                st.warning("⚠️ La confianza es baja. Por favor, verifica que la imagen sea clara y muestre una hoja de planta.")
