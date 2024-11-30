import numpy as np
from tkinter import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Datos de entrenamiento: [edad, peso, nivel]
# Etiquetas: 0 = Cardio, 1 = Fuerza
datos_entrenamiento = np.array([
    [20, 60, 0],  # Edad, Peso, Nivel (0: Básico)
    [25, 75, 1],  # Nivel (1: Intermedio)
    [30, 85, 2],  # Nivel (2: Avanzado)
    [40, 70, 0],  # Básico
    [35, 90, 2]   # Avanzado
])

# Las características (edad, peso, nivel) son las primeras tres columnas
X = datos_entrenamiento  # Aquí ya estamos usando 3 características (edad, peso, nivel)
y = datos_entrenamiento[:, -1]  # Las etiquetas son las últimas columnas (Cardio/Fuerza)

# Crear el modelo de red neuronal
model = Sequential()
model.add(Dense(10, input_dim=3, activation='relu'))  # Capa de entrada con 3 características
model.add(Dense(8, activation='relu'))  # Capa oculta
model.add(Dense(1, activation='sigmoid'))  # Capa de salida (Cardio o Fuerza)

# Compilar el modelo
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Entrenar el modelo
model.fit(X, y, epochs=100, verbose=0)

# Función para predecir el tipo de ejercicio
def predecir_ejercicio():
    try:
        # Obtener la edad, peso y nivel desde las entradas
        edad = int(edad_entry.get())  # Edad
        peso = int(peso_entry.get())  # Peso
        nivel = nivel_entry.get().lower()  # Nivel (en minúsculas para validarlo)

        # Validación de nivel
        if nivel not in ["básico", "intermedio", "avanzado"]:
            resultado_label.config(text="Por favor, ingresa un nivel válido (Básico, Intermedio, Avanzado).")
            return

        # Mapeo del nivel a valores numéricos
        niveles = {"básico": 0, "intermedio": 1, "avanzado": 2}
        nivel_num = niveles[nivel]

        # Realizar la predicción con el modelo entrenado
        # Pasamos la entrada con 3 características (edad, peso, nivel)
        prediccion = model.predict(np.array([[edad, peso, nivel_num]]))  # Entrada con 3 características

        # Si la predicción es menor que 0.5, es Cardio, sino es Fuerza
        if prediccion < 0.5:
            resultado_label.config(text="Recomendación: Cardio")
        else:
            resultado_label.config(text="Recomendación: Fuerza")
            
    except ValueError:
        resultado_label.config(text="Por favor, ingresa valores numéricos válidos.")
    except Exception as e:
        resultado_label.config(text=f"Error: {str(e)}")

# Interfaz gráfica con tkinter
root = Tk()
root.title("Recomendador de Ejercicios con Keras")
root.geometry("400x300")

# Etiquetas y campos de entrada para Edad, Peso y Nivel
Label(root, text="Edad:").pack()
edad_entry = Entry(root)
edad_entry.pack()

Label(root, text="Peso (kg):").pack()
peso_entry = Entry(root)
peso_entry.pack()

Label(root, text="Nivel (Básico, Intermedio, Avanzado):").pack()
nivel_entry = Entry(root)
nivel_entry.pack()

# Botón para realizar la predicción
Button(root, text="Recomendar Ejercicio", command=predecir_ejercicio).pack(pady=10)

# Label para mostrar el resultado
resultado_label = Label(root, text="", font=("Arial", 12))
resultado_label.pack()

# Iniciar la ventana de tkinter
root.mainloop()
