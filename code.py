# Paso 1: Clonar dataset desde GitHub
#!git clone https://github.com/usuario/repositorio-ejemplo.git  # Reemplaza con el link real

import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import kagglehub
from kagglehub import KaggleDatasetAdapter
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.preprocessing import StandardScaler

# Paso 1: Fuente de datos desde GitHub o Kaggle
USE_KAGGLE = True  # Cambiar a True si deseas usar la fuente desde Kaggle

if USE_KAGGLE:
    # Fuente alternativa desde Kaggle
    #!pip install -q kagglehub[pandas-datasets]


    #dataset_path ="/kaggle/input/webots-car-navigation"

    dataset_path = kagglehub.dataset_download(
        "paradajoseluis/navigation-webots-2"
    )

    file_path = os.path.join(dataset_path, "steering_angles.csv")  # Construct the path to the CSV within the downloaded dataset
    df = pd.read_csv(file_path)

    IMAGE_FOLDER = os.path.join(dataset_path, "DATASET2", "DATASET2") # Adjust based on the actual downloaded structure if needed

    print(f"Dataset downloaded to: {dataset_path}")
    print(f"CSV file path: {file_path}")
    print(f"Image folder path: {IMAGE_FOLDER}")
    print(os.listdir(IMAGE_FOLDER))  #Verificamos acceso a las imagenes

else:
    # Clonar dataset desde GitHub
    #!git clone https://github.com/edwinhdez/navegacion_autonoma_proyecto_final.git  # Reemplaza con el link real

    # Ruta al archivo CSV y carpeta de imágenes
    CSV_PATH = "navegacion_autonoma_proyecto_final/dataset/steering_angles_intersection.csv"
    IMAGE_FOLDER = "/content/navegacion_autonoma_proyecto_final/dataset/DATASET"
    # Cargar el archivo CSV

    df = pd.read_csv(CSV_PATH)
    print(os.listdir(IMAGE_FOLDER))  #Verificamos acceso a las imagenes


# Renombrar columnas para consistencia
df = df.rename(columns={
    'image_name': 'center',
    'steering_angle': 'steering',
    'INTERSECTION': 'intersection'
})

df['intersection'] = df['intersection'].astype(int)


# Parámetros de imagen
IMG_HEIGHT, IMG_WIDTH = 66, 200

def preprocess_image(img_name):
    """
    Preprocesa una imagen para el modelo de navegación autónoma.

    Parámetros:
        img_name (str): Nombre del archivo de la imagen a procesar.

    Proceso:
        - Construye la ruta completa de la imagen usando IMAGE_FOLDER.
        - Lee la imagen desde disco.
        - Lanza un error si la imagen no se puede leer.
        - Convierte la imagen de BGR a RGB.
        - Redimensiona la imagen a (IMG_WIDTH, IMG_HEIGHT).
        - Normaliza los valores de los pixeles a rango [0, 1].

    Retorna:
        np.ndarray: Imagen preprocesada lista para ser usada por el modelo.
    """
    path = os.path.join(IMAGE_FOLDER, img_name)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    return img

def augment_image(image, angle):
    """
    Realiza aumentos de datos en la imagen y ajusta el ángulo de giro correspondiente.

    Parámetros:
        image (np.ndarray): Imagen de entrada en formato RGB y valores en [0, 1].
        angle (float): Ángulo de giro original.

    Proceso:
        - Con probabilidad 0.5, voltea horizontalmente la imagen y cambia el signo del ángulo.
        - Modifica el brillo de la imagen en el espacio HSV de forma aleatoria.
        - Convierte la imagen de nuevo a RGB y normaliza a [0, 1].

    Retorna:
        tuple: (imagen aumentada, ángulo ajustado)
    """
    import random
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        angle = -angle
    hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * (0.3 + np.random.uniform())
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) / 255.0
    return image, angle

def load_data(df, augment=True):
    """
    Carga y preprocesa todas las imágenes y etiquetas desde el DataFrame.

    Parámetros:
        df (pd.DataFrame): DataFrame con las columnas 'center', 'steering' e 'intersection'.

    Proceso:
        - Itera sobre cada fila del DataFrame.
        - Preprocesa la imagen correspondiente usando preprocess_image.
        - Almacena la imagen, el ángulo de giro y la etiqueta de intersección en listas.

    Retorna:
        tuple: (np.ndarray de imágenes, np.ndarray de ángulos, np.ndarray de intersecciones)
    """
    images, angles, ints = [], [], []
    for _, row in df.iterrows():
        img = preprocess_image(row['center'])
        angle = row['steering']
        if augment:
            img, angle = augment_image(img, angle)
        images.append(img)
        angles.append(angle)
        ints.append(row['intersection'])
    return np.array(images), np.array(angles), np.array(ints)

# Define los bordes de los bins para el ángulo de dirección
bin_edges = np.array([-0.10, -0.05, 0.00, 0.05, 0.10])

# Crea la columna 'steering_bin' en el DataFrame para estratificar (usa la columna correcta)
df['steering_bin'] = pd.cut(df['steering'], bins=bin_edges, labels=False, include_lowest=True)

# Elimina filas con NaN en 'steering_bin' para evitar errores en la estratificación
df = df.dropna(subset=['steering_bin'])

# Paso 1: Dividir el DataFrame original en Train (70%) y Temp (30%) usando estratificación
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df['steering_bin'],
    random_state=42
)

# Paso 2: Dividir Temp en Validación (15%) y Test (15%) también con estratificación
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,  # 50% de 30% = 15%
    stratify=temp_df['steering_bin'],
    random_state=42
)

# (Opcional) Imprime el tamaño de cada conjunto
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Paso 3: Cargar imágenes y etiquetas desde los subconjuntos
X_train, y_angle_train, y_inter_train = load_data(train_df)
X_val, y_angle_val, y_inter_val = load_data(val_df)

X_test, y_angle_test, y_inter_test = load_data(test_df)

# Parámetro para activar/desactivar el uso de StandardScaler
USE_SCALER = True  # Cambia a False si NO quieres usar StandardScaler

if USE_SCALER:
    scaler = StandardScaler()
    y_angle_train_scaled = scaler.fit_transform(y_angle_train.reshape(-1, 1))
    y_angle_val_scaled = scaler.transform(y_angle_val.reshape(-1, 1))
    y_angle_test_scaled = scaler.transform(y_angle_test.reshape(-1, 1))

    media = scaler.mean_[0]
    desviacion = scaler.scale_[0]

    print("Media usada por el scaler:", media)
    print("Desviación estándar usada por el scaler:", desviacion)
else:
    scaler = None
    y_angle_train_scaled = y_angle_train
    y_angle_val_scaled = y_angle_val
    y_angle_test_scaled = y_angle_test


# Modelo multitarea NVIDIA
import tensorflow as tf
from tensorflow.keras import layers, models, Input

def build_multitask_model():
    """
    Construye y compila un modelo multitarea basado en la arquitectura NVIDIA para conducción autónoma.
    El modelo tiene dos salidas:
        - intersection: predicción binaria (intersección o no)
        - angle: regresión del ángulo de giro
    Utiliza varias capas convolucionales y densas para extraer características de las imágenes.
    """
    inputs = Input(shape=(66, 200, 3))  # Entrada: imagen RGB de 66x200 píxeles

    # Primera capa convolucional: 24 filtros 5x5, stride 2, activación ReLU
    x = layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu')(inputs)
    # Segunda capa convolucional: 36 filtros 5x5, stride 2, activación ReLU
    x = layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu')(x)
    # Tercera capa convolucional: 48 filtros 5x5, stride 2, activación ReLU
    x = layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu')(x)
    # Cuarta capa convolucional: 64 filtros 3x3, activación ReLU
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    # Quinta capa convolucional: 64 filtros 3x3, activación ReLU
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    # Aplanar la salida para conectarla a las capas densas
    x = layers.Flatten()(x)
    # Capa densa de 100 neuronas, activación ReLU
    x = layers.Dense(100, activation='relu')(x)
    # Capa densa de 50 neuronas, activación ReLU
    x = layers.Dense(50, activation='relu')(x)
    # Capa densa de 10 neuronas, activación ReLU
    x = layers.Dense(10, activation='relu')(x)

    # Salida 1: predicción de intersección (binaria, activación sigmoide)
    inter_out = layers.Dense(1, activation='sigmoid', name='intersection')(x)
    # Salida 2: predicción de ángulo (regresión, activación lineal)
    angle_out = layers.Dense(1, activation='linear', name='angle')(x)

    # Definir el modelo con dos salidas
    model = models.Model(inputs=inputs, outputs=[inter_out, angle_out])

    # Compilar el modelo con optimizador Adam, pérdidas y métricas para cada salida
    model.compile(
        optimizer='adam',  # Optimizador Adam
        loss={
            'intersection': 'binary_crossentropy',  # Pérdida binaria para intersección
            'angle': 'mse'                          # Error cuadrático medio para ángulo
        },
        metrics={
            'intersection': 'accuracy',             # Métrica de exactitud para intersección
            'angle': 'mae'                          # Error absoluto medio para ángulo
        }
    )

    return model  # Devuelve el modelo compilado

# Construye el modelo multitarea y muestra un resumen de su arquitectura
model = build_multitask_model()
model.summary()


# Crear callback para checkpoints para guardar automáticamente el mejor modelo durante el entrenamiento
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint_cb = ModelCheckpoint(
    "checkpoint_best.h5",           # Nombre del archivo a guardar
    save_best_only=True,            # Solo guarda si mejora la métrica de validación
    monitor="val_loss",             # Monitorea la pérdida total de validación
    mode="min",                     # Busca el mínimo de la pérdida
    save_weights_only=False,        # Guarda el modelo completo
    verbose=1
)

# Entrenamiento del modelo multitarea
history = model.fit(
    X_train,  # Imágenes de entrenamiento
    {
        'intersection': y_inter_train,  # Etiquetas de intersección para entrenamiento
        'angle': y_angle_train_scaled          # Etiquetas de ángulo para entrenamiento
    },
    validation_data=(X_val, {           # Datos de validación
        'intersection': y_inter_val,    # Etiquetas de intersección para validación
        'angle': y_angle_val_scaled            # Etiquetas de ángulo para validación
    }),
    epochs=10,                          # Número de épocas de entrenamiento
    batch_size=32,                       # Tamaño del batch
    callbacks=[checkpoint_cb]           # Lista de callbacks
)

def graficar_accuracy_entrenamiento(history):
    """
    Grafica el accuracy de la tarea de intersección durante el entrenamiento y validación.

    Parámetros:
        history: objeto History devuelto por model.fit()
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['intersection_accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_intersection_accuracy'], label='Validación')
    plt.title('Accuracy de Intersección durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

# Ejemplo de uso:
graficar_accuracy_entrenamiento(history)

# Evaluación del modelo en el conjunto de prueba
print("Evaluación en el conjunto de prueba:")
model.evaluate(
    X_test,  # Imágenes de prueba
    {
        'intersection': y_inter_test,  # Etiquetas de intersección para prueba
        'angle': y_angle_test_scaled          # Etiquetas de ángulo para prueba
    }
)

# Guardar modelo
model.save("nvidia_multitask_model_rev5.h5")

import random
import matplotlib.pyplot as plt

def mostrar_predicciones_old(model, X_test, y_inter_test, y_angle_test, num_imagenes=10):
    """
    Muestra imágenes aleatorias del set de prueba junto con sus predicciones y valores reales.

    Parámetros:
        model: modelo entrenado.
        X_test: imágenes de prueba.
        y_inter_test: etiquetas reales de intersección.
        y_angle_test: etiquetas reales de ángulo.
        num_imagenes: número de imágenes a mostrar.
    """
    for i in range(num_imagenes):
        idx = random.randint(0, len(X_test) - 1)
        img = X_test[idx]
        real_inter = y_inter_test[idx]
        real_angle = y_angle_test[idx]

        # Hacemos la predicción (agregando un batch dimension)
        pred_inter, pred_angle = model.predict(img[np.newaxis, ...], verbose=0)
        

        plt.figure(figsize=(4, 3))
        plt.imshow(img)
        plt.title(
            f"Real: Intersección={real_inter}, Ángulo={real_angle:.2f}\n"
            f"Predicho: Intersección={pred_inter[0][0]:.2f}, Ángulo={pred_angle[0][0]:.2f}"
        )
        plt.axis('off')
        plt.show()

import random
import matplotlib.pyplot as plt

def mostrar_predicciones(model, X_test, y_inter_test, y_angle_test, scaler, num_imagenes=50):
    """
    Muestra imágenes aleatorias del set de prueba junto con sus predicciones y valores reales.
    Interpreta el ángulo usando el scaler para invertir la estandarización si está activo.

    Parámetros:
        model: modelo entrenado.
        X_test: imágenes de prueba.
        y_inter_test: etiquetas reales de intersección.
        y_angle_test: etiquetas reales de ángulo (escalados o no).
        scaler: StandardScaler ajustado con los datos de entrenamiento o None.
        num_imagenes: número de imágenes a mostrar.
    """
    for i in range(num_imagenes):
        idx = random.randint(0, len(X_test) - 1)
        img = X_test[idx]
        real_inter = y_inter_test[idx]
        real_angle_scaled = y_angle_test[idx]

        # Hacemos la predicción (agregando un batch dimension)
        pred_inter, pred_angle_scaled = model.predict(img[np.newaxis, ...], verbose=0)

        if scaler is not None:
            # Asegura que los valores sean 2D para inverse_transform
            real_angle = scaler.inverse_transform(np.array(real_angle_scaled).reshape(1, -1))[0][0]
            pred_angle = scaler.inverse_transform(np.array(pred_angle_scaled).reshape(1, -1))[0][0]
        else:
            real_angle = real_angle_scaled
            pred_angle = pred_angle_scaled[0][0]

        plt.figure(figsize=(4, 3))
        plt.imshow(img)
        plt.title(
            f"Real: Intersección={real_inter}, Ángulo={real_angle:.3f}\n"
            f"Predicho: Intersección={pred_inter[0][0]:.2f}, Ángulo={pred_angle:.3f}"
        )
        plt.axis('off')
        plt.show()

#mostrar_predicciones(model, X_test, y_inter_test, y_angle_test_scaled, scaler)
mostrar_predicciones(model, X_test, y_inter_test, y_angle_test_scaled if USE_SCALER else y_angle_test, scaler)