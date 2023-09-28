import os

import es_core_news_sm


def setUltimoModeloUsado(lista, modelo):
    """
        Función para establecer el último modelo usado en primera posición de la lista.
        Parámetros:
            - lista: Lista de modelos disponibles.
            - modelo: Modelo que queremos situar en primera posición.
        Devuelve:
            - lista2: Lista final con el modelo en primera posición.
    """
    lista.remove(modelo)
    lista2 = []
    lista2.append(modelo)
    for elemento in lista:
        lista2.append(elemento)
    return lista2

def clasificar(texto,modelo):
    """
        Función previa a clasificar un texto.
        Parámetros:
            - texto: String con el texto a clasificar.
            - modelo: String con el modelo a usar en la clasificación.
    """
    print(modelo[0])
    if modelo[0]=="S" or modelo[0]=="N":  # Si es SVM o Naive
        print("Modelo NO BERT")
        return clasificarNOBERT(texto, modelo)
    else:
        print("Modelo BERT")
        return clasificarBERT(texto, modelo)


def clasificarBERT(texto, modelo):
    """
        Función para clasificar un texto con un modelo BERT.
        Parámetros:
            - texto: String con el texto a clasificar.
            - modelo: String con el modelo a usar en la clasificación.
        Devuelve:
            - etiqueta: String con la etiqueta predicha.
            - score: Entero con el score de la predicción.
    """
    import os
    from transformers import pipeline, AutoTokenizer
    os.chdir("ModelosDefinitivos")

    checkpoint=os.listdir(os.getcwd()+"/"+modelo)[-2] # Elegimos último checkpoint.
    print(checkpoint)
    # Definimos modelo con los mismos parámetros que los del entrenamiento.
    classifier = pipeline("text-classification", truncation=True, model=modelo+"\\"+checkpoint,
                          tokenizer=AutoTokenizer.from_pretrained(modelo+"\\"+checkpoint), batch_size=16,
                          )
    # Procesamiento del texto
    from procesaTexto import procesaTexto
    listaAuxiliar=[]
    listaAuxiliar.append(texto)
    procesador = procesaTexto(listaAuxiliar, listaAuxiliar) # La inicializamos con la lista auxiliar simplemente para usar una función.
    textoProcesado=procesador.procesamiento2PorParámetro(listaAuxiliar)[0]
    print(textoProcesado)

    os.chdir('..')
    salida = str(classifier(textoProcesado)).split() # Obtenemos la salida del clasificador.
    print(salida)
    etiqueta = ""
    score = ""
    if salida[1] == "'NO":  # Extraemos y formateamos la etiqueta.
        etiqueta = "NO ODIO"
        str(salida[4]).replace("}", "")
        str(salida[4]).replace("]", "")
        score = str(salida[4]).replace("}", "")
        score = score.replace("]", "")
    else:
        etiqueta = "ODIO"
        score = str(salida[3]).replace("}", "")
        score = score.replace("]", "")
    return etiqueta, score

def clasificarNOBERT(texto, modelo):
    """
        Función para clasificar un texto con un modelo NO BERT.
        Parámetros:
            - texto: String con el texto a clasificar.
            - modelo: String con el modelo a usar en la clasificación.
        Devuelve:
            - etiqueta: String con la etiqueta predicha.
            - score: Entero con el score de la predicción. (Siempre será vacía en este caso)
    """
    from joblib import dump, load
    os.chdir("ModelosDefinitivos") # Buscamos el modelo.
    os.chdir(modelo)
    print(os.getcwd())
    contenidoCarpeta=os.listdir(os.getcwd())
    print(contenidoCarpeta)
    model = load(contenidoCarpeta[0]) # Cargamos el modelo con joblib
    vectorizer=load(contenidoCarpeta[1])
    os.chdir("..") # Retrocedemos una carpeta.
    os.chdir("..") # Retrocedemos una carpeta.
    etiqueta = ""
    score = ""

    # Procesamiento del texto.
    from procesaTexto import procesaTexto
    listaAuxiliar = []
    listaAuxiliar.append(texto)
    procesador = procesaTexto(listaAuxiliar,
                              listaAuxiliar)  # La inicializamos con la lista auxiliar simplemente para usar una función.
    textoProcesado = procesador.procesamiento2PorParámetro(listaAuxiliar)[0]
    print(textoProcesado)

    def tokenizar(texto): # Función de tokenización, es la misma que la entrenamiento BERT.

        nlp = es_core_news_sm.load()
        texto = nlp(texto)
        tokens = []
        textoProcesado = ""
        for token in texto:
            if token.text.isalpha() and not token.is_stop:  # Quitamos signos de puntuación y stop words.
                tokens.append(token.lemma_.lower())  # Reducimos a la raiz cada palabra y la pasamos a minúscula.
        textoProcesado = " ".join(tokens)
        # print(texto,"-",textoProcesado,end="\n")
        return textoProcesado

    textoProcesado=tokenizar(textoProcesado)
    textoFit = vectorizer.transform([textoProcesado]) # Adaptamos el texto al vector de características.

    prediction = model.predict(textoFit)
    print(prediction)
    if prediction==1: # Extraemos etiquetas
        etiqueta = "NO ODIO"
        score = ""
    else:
        etiqueta = "ODIO"
        score = ""

    return etiqueta, score

def getUltimoModeloUsado(self):
    """
        Getter del ultimo modelo usado.
        Devuelve:
            - String con el último modelo usado.
    """
    return self.ultimoModeloUsado
