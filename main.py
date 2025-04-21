"""
from transformers import pipeline

# Cargar un modelo de análisis de sentimientos
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Probar con un texto
result = classifier("Odio los modelos de intelgencia artificial!")
print(result)


from transformers import pipeline

# Cargar un modelo preentrenado de Hugging Face
generador_texto = pipeline("text-generation", model="gpt2")

# Generar texto a partir de una frase inicial
texto_generado = generador_texto("Hola, soy una IA que", max_length=50, num_return_sequences=1)

print(texto_generado)

"""
"""
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os

# Cargar dataset de recetas
df = pd.read_csv("recetas.csv")

# Modelo NLP para comparar similitud
modelo = SentenceTransformer("all-MiniLM-L6-v2")

# Embeddings de todas las recetas
embeddings_recetas = modelo.encode(df["ingredientes"].tolist(), convert_to_tensor=True)

# Archivo donde se guardarán las elecciones de los usuarios
archivo_interacciones = "interacciones.csv"

# Crear archivo si no existe
if not os.path.exists(archivo_interacciones):
    pd.DataFrame(columns=["usuario", "ingredientes", "receta"]).to_csv(archivo_interacciones, index=False)

def recomendar_receta(ingredientes_usuario, usuario):
    # Convertir los ingredientes ingresados en un embedding
    embedding_input = modelo.encode(ingredientes_usuario, convert_to_tensor=True)

    # Calcular similitudes entre el input y las recetas del dataset
    similitudes = util.pytorch_cos_sim(embedding_input, embeddings_recetas)

    # Obtener la receta más similar
    indice_receta = similitudes.argmax().item()
    receta_recomendada = df.iloc[indice_receta]["receta"]

    # Guardar la interacción del usuario
    registrar_interaccion(usuario, ingredientes_usuario, receta_recomendada)

    return receta_recomendada

def registrar_interaccion(usuario, ingredientes, receta):
    #Guarda las elecciones de los usuarios para que el sistema aprenda de ellas 
    interacciones = pd.read_csv(archivo_interacciones)
    
    # Agregar la nueva interacción
    nueva_interaccion = pd.DataFrame([{"usuario": usuario, "ingredientes": ingredientes, "receta": receta}])
    interacciones = pd.concat([interacciones, nueva_interaccion], ignore_index=True)

    # Guardar el archivo actualizado
    interacciones.to_csv(archivo_interacciones, index=False)

# Ejemplo de uso con un usuario llamado "Diego"

usuario = input("Introduce tu nombre: ")
ingredientes = input("Introduce los ingredientes que tienes (separados por comas): ")

print(f"Receta recomendada para {usuario}: {recomendar_receta(ingredientes, usuario)}")
"""

"""
import pandas as pd
import torch
from transformers import pipeline, set_seed
import os

# Cargar o crear dataset de recetas
archivo_recetas = "recetas2.csv"
if not os.path.exists(archivo_recetas):
    pd.DataFrame(columns=["ingredientes", "receta"]).to_csv(archivo_recetas, index=False)

# Cargar modelo de generación de texto
set_seed(42)  # Para generar respuestas consistentes
generador = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)

def generar_receta(ingredientes):
    #Genera una nueva receta basada en los ingredientes dados 
    prompt = f"Receta con {ingredientes}:\n"
    
    # Generar texto con el modelo
    receta_generada = generador(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    
    # Limpiar la receta generada (remover contenido innecesario)
    receta_final = receta_generada.split("\n")[0]  
    
    # Guardar la nueva receta en la base de datos
    registrar_receta(ingredientes, receta_final)
    
    return receta_final

def registrar_receta(ingredientes, receta):
    #Guarda las recetas generadas en la base de datos 
    df = pd.read_csv(archivo_recetas)
    nueva_receta = pd.DataFrame([{"ingredientes": ingredientes, "receta": receta}])
    df = pd.concat([df, nueva_receta], ignore_index=True)
    df.to_csv(archivo_recetas, index=False)

# Ejemplo de uso
ingredientes = "pollo, ajo, cebolla, tomate"
print(f"Receta generada:\n{generar_receta(ingredientes)}")
"""


#INTENTO NUMERO 3
import tkinter as tk
from tkinter import *
from tkinter import filedialog

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os

root = tk.Tk()
root.title("Generador de recetas")
root.resizable(0, 0)
root.background = "#f0f0f0"

title = Label(root, text="Generador de recetas", font=("Helvetica", 25, "bold"), fg="#000", background="#fcfcfc")
title.pack(pady=30)

usuario_label = tk.Label(root, text="Nombre de usuario:", font=("Helvetica", 12), background="#fcfcfc", fg="#000")
usuario_label.pack(pady=10)
usuario_text = tk.Entry(root, width=30, font=("Helvetica", 12), background="#ffffff", fg="#000")
usuario_text.pack(pady=10)

ingredientes_label = tk.Label(root, text="Ingredientes (separados por comas):", font=("Helvetica", 12), background="#fcfcfc", fg="#000")
ingredientes_label.pack(pady=10)
ingredientes_text = tk.Entry(root, width=30, font=("Helvetica", 12), background="#ffffff", fg="#000")
ingredientes_text.pack(pady=10)

porcion_label = tk.Label(root, text="Tamaño de la porción:", font=("Helvetica", 12), background="#fcfcfc", fg="#000")       
porcion_label.pack(pady=10)
porcion_text = tk.Entry(root, width=30, font=("Helvetica", 12), background="#ffffff", fg="#000")
porcion_text.pack(pady=10)

dificiltad_label = tk.Label(root, text="Dificultad (fácil, intermedio, difícil):", font=("Helvetica", 12), background="#fcfcfc", fg="#000")
dificiltad_label.pack(pady=10)
dificultad_text = tk.Entry(root, width=30, font=("Helvetica", 12), background="#ffffff", fg="#000")
dificultad_text.pack(pady=10)


# Cargar dataset de recetas (aquí se asume que ya tienes ingredientes y recetas)
df = pd.read_csv("recetas.csv")

# Modelo NLP para comparar similitud
modelo = SentenceTransformer("all-MiniLM-L6-v2")

# Embeddings de todas las recetas
embeddings_recetas = modelo.encode(df["ingredientes"].tolist(), convert_to_tensor=True)

# Archivo donde se guardarán las elecciones de los usuarios
archivo_interacciones = "interacciones.csv"

# Crear archivo si no existe
if not os.path.exists(archivo_interacciones):
    pd.DataFrame(columns=["usuario", "ingredientes", "receta", "porcion", "dificultad", "instrucciones"]).to_csv(archivo_interacciones, index=False)

def recomendar_receta(ingredientes_usuario, usuario, porcion, dificultad):
    # Convertir los ingredientes ingresados en un embedding
    embedding_input = modelo.encode(ingredientes_usuario, convert_to_tensor=True)

    # Calcular similitudes entre el input y las recetas del dataset
    similitudes = util.pytorch_cos_sim(embedding_input, embeddings_recetas)

    # Obtener la receta más similar
    indice_receta = similitudes.argmax().item()
    receta_recomendada = df.iloc[indice_receta]

    # Guardar la interacción del usuario con más detalles
    registrar_interaccion(usuario, ingredientes_usuario, receta_recomendada["receta"], porcion, dificultad, receta_recomendada["instrucciones"])

    # Formatear la receta con todos los detalles
    receta_formateada = (
        f"Receta recomendada para {usuario}:\n\n"
        f"Receta: {receta_recomendada['instrucciones']}\n"
        f"Tamaño de la porción: {porcion}\n"
        f"Dificultad: {dificultad}\n"
        f"Instrucciones:\n{receta_recomendada['receta']}\n"
        f"Ingredientes: {receta_recomendada['ingredientes']}\n"
    )
    print(receta_formateada)  # Mostrar la receta en la consola
    return receta_formateada

def registrar_interaccion(usuario, ingredientes, receta, porcion, dificultad, instrucciones):
    # Guarda las elecciones de los usuarios para que el sistema aprenda de ellas 
    interacciones = pd.read_csv(archivo_interacciones)
    
    # Agregar la nueva interacción
    nueva_interaccion = pd.DataFrame([{
        "usuario": usuario, 
        "ingredientes": ingredientes, 
        "receta": receta, 
        "porcion": porcion, 
        "dificultad": dificultad, 
        "instrucciones": instrucciones
    }])
    interacciones = pd.concat([interacciones, nueva_interaccion], ignore_index=True)

    # Guardar el archivo actualizado
    interacciones.to_csv(archivo_interacciones, index=False)



#Botón para guardar los datos y generar la receta
boton = tk.Button(root, text="Generar receta", font=("Helvetica", 12), background="#4CAF50", fg="#fff", command=lambda: recomendar_receta(ingredientes_text.get(), usuario_text.get(), porcion_text.get(), dificultad_text.get()))
boton.pack(pady=20)

result = recomendar_receta(ingredientes_text.get(), usuario_text.get(), porcion_text.get(), dificultad_text.get())
texto_resultado = tk.Text(root, width=50, height=10, font=("Helvetica", 12), background="#ffffff", fg="#000")
texto_resultado.pack(pady=10)

texto_resultado.insert(tk.END, result)
texto_resultado.config(state=tk.DISABLED)  


# Ejemplo de uso con un usuario llamado "Diego"

#usuario = input("Introduce tu nombre: ")
#ingredientes = input("Introduce los ingredientes que tienes (separados por comas): ")
#porcion = input("Introduce el tamaño de la porción: ")
#dificultad = input("Introduce la dificultad (facil, intermedio, dificil): ")

#print(f"\n{recomendar_receta(ingredientes, usuario, porcion, dificultad)}")

frame = Frame(root, background="#ffffff", padx=10, pady=10)
frame.config(cursor="")    
frame.config(relief="groove")
frame.config(bd=5)         
frame.place(x=40, y=150)

root.mainloop()

#INTENTO NUMERO 4
"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Cargar el modelo y el tokenizador GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Función para generar recetas
def generar_receta(ingredientes_usuario):
    # Crear un prompt claro que guíe la IA a generar una receta
    prompt = (f"Genera una receta completa con los ingredientes siguientes: {ingredientes_usuario}. "
              "El resultado debe incluir un título del platillo, los ingredientes necesarios y las instrucciones "
              "paso a paso para prepararlo.\n\n")
    
    # Convertir el prompt a tokens
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generar la salida del modelo
    outputs = model.generate(inputs, 
                             max_length=200, 
                             num_return_sequences=1, 
                             no_repeat_ngram_size=3, 
                             temperature=0.7, 
                             do_sample=True, 
                             pad_token_id=tokenizer.eos_token_id)

    # Decodificar la salida generada
    receta_generada = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return receta_generada

# Ejemplo de uso
ingredientes = "huevo, leche, harina, azúcar, mantequilla"
receta = generar_receta(ingredientes)
print(f"Receta generada:\n{receta}")

"""
