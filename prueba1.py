import tkinter as tk
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os
from tkinter import messagebox
import pygame

# Inicializar pygame mixer para reproducir audio
pygame.mixer.init()
#musica de fondo
def reproducir_musica_fondo():
    try:
        pygame.mixer.music.load("fondo.ogg")  # Do Robots Dream of Eternal Sleep?-heaven piece her
        pygame.mixer.music.play(loops=-1)  
    except Exception as e:
        messagebox.showwarning("Audio", f"No se pudo reproducir la música de fondo: {e}")

# Configuración de la ventana principal
root = tk.Tk()
root.title("Generador de recetas")
root.resizable(0, 0)
#modificado
reproducir_musica_fondo() 
root.iconbitmap("icono.ico")
root.geometry("620x680")
root.configure(bg="#1E1E1E")

# Widgets de interfaz
label_font = ("Helvetica", 12)
entry_font = ("Helvetica", 12)

# Título
title = tk.Label(root, text="Generador de recetas", font=("Helvetica", 25, "bold"), fg="#FFD166", bg="#2C2C2C")
title.pack(pady=30)

# Nombre de usuario
usuario_label = tk.Label(root, text="Nombre de usuario:", font=label_font, bg="#2C2C2C", fg="#FFFFFF")
usuario_label.pack(pady=10)
usuario_text = tk.Entry(root, width=30, font=entry_font, bg="#3A3A3A", fg="#FFFFFF", insertbackground="#FFFFFF")
usuario_text.pack(pady=10)

# Ingredientes
ingredientes_label = tk.Label(root, text="Ingredientes (separados por comas):", font=label_font, bg="#2C2C2C", fg="#FFFFFF")
ingredientes_label.pack(pady=10)
ingredientes_text = tk.Entry(root, width=30, font=entry_font, bg="#3A3A3A", fg="#FFFFFF", insertbackground="#FFFFFF")
ingredientes_text.pack(pady=10)

# Cargar dataset
try:
    df = pd.read_csv("recetas.csv", encoding="latin1")
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings_recetas = modelo.encode(df["ingredientes"].tolist(), convert_to_tensor=True)
except Exception as e:
    messagebox.showerror("Error", f"Error al cargar recursos: {str(e)}")
    root.destroy()

# Ruta del archivo de interacciones
archivo_interacciones = "interacciones.csv"
if not os.path.exists(archivo_interacciones):
    pd.DataFrame(columns=["usuario", "ingredientes", "receta", "instrucciones"]).to_csv(archivo_interacciones, index=False)

def registrar_interaccion(usuario, ingredientes, receta, instrucciones):
    nueva_interaccion = pd.DataFrame([{
        "usuario": usuario,
        "ingredientes": ingredientes,
        "receta": receta,
        "instrucciones": instrucciones
    }])
    nueva_interaccion.to_csv(archivo_interacciones, mode='a', header=False, index=False)

def recomendar_receta():
    usuario = usuario_text.get().strip()
    ingredientes_usuario = ingredientes_text.get().strip()

    if not all([usuario, ingredientes_usuario]):
        messagebox.showwarning("Campos incompletos", "Por favor, completa todos los campos.")
        return

    try:
        # Reproducir audio de boton
        pygame.mixer.music.load("audio_receta.ogg") #taunt-pizza tower sfx
        pygame.mixer.music.play()
    except Exception as e:
        messagebox.showwarning("Audio", f"No se pudo reproducir el audio: {e}")

    try:
        embedding_input = modelo.encode(ingredientes_usuario, convert_to_tensor=True)
        similitudes = util.pytorch_cos_sim(embedding_input, embeddings_recetas)
        similitudes = similitudes.squeeze()

        indice_max = similitudes.argmax().item()
        receta_recomendada = df.iloc[indice_max]

        registrar_interaccion(usuario, ingredientes_usuario, receta_recomendada["receta"], receta_recomendada["instrucciones"])

        receta_formateada = (
            f"Receta recomendada para {usuario}:\n\n"
            f"Receta: {receta_recomendada['receta']}\n"
            f"Ingredientes: {receta_recomendada['ingredientes']}\n"
            f"Instrucciones: {receta_recomendada['instrucciones']}\n"
        )

        texto_resultado.config(state=tk.NORMAL)
        texto_resultado.delete(1.0, tk.END)
        texto_resultado.insert(tk.END, receta_formateada)
        texto_resultado.config(state=tk.DISABLED)

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Botón
boton = tk.Button(root, text="Generar receta", font=label_font, bg = "#FF6B35", fg = "#FFFFFF", activebackground = "#FF884F", activeforeground = "#FFFFFF", command=recomendar_receta)
boton.pack(pady=20)

# Área de resultados
texto_resultado = tk.Text(root, width=60, height=15, font=entry_font, bg = "#2C2C2C", fg = "#FFFFFF", insertbackground="#FFFFFF")
texto_resultado.pack(pady=10)
texto_resultado.config(state=tk.DISABLED)

root.mainloop()