import os

from sklearn.metrics import classification_report, make_scorer
from sklearn.model_selection import KFold, StratifiedShuffleSplit, cross_val_score
from transformers import pipeline, AutoTokenizer


class modelo:

    def __init__(self, delitosodio, delitosnodio,configuracion):
        """
           Constructor de la clase
           Parámetros:
                - delitosodio: Vector de delitos de Odio
                - delitosnodio: Vector de delitos de No odio
                - configuracion: Instancia de la clase con configuración con los datos de parametros.txt
        """
        self.configuracion = configuracion # Instancia de la clase de configuración con los datos de parámetros.txt
        self.modelosBert=["distilbert-base-uncased","dccuchile/bert-base-spanish-wwm-cased",
                          "PlanTL-GOB-ES/roberta-base-bne","bertin-project/bertin-roberta-base-spanish"
                          ,"bert-base-multilingual-cased","xlm-roberta-large"] # Lista de cadenas con los modelos BERT Disponibles
        self.modelosNOBert=["SVM","Naive"] # Lista de cadenas con los modelos No BERT disponibles
        self.delitosodio = delitosodio # Lista de denuncias de ODIO
        self.delitosnodio = delitosnodio # Lista de denuncias de NO ODIO

        self.metricas = [] # Vector para guardar las métricas de la ejecución.
        self.tiempoValidacionCruzada=0 # Entero para almacenar el tiempo que ha tardado en realizar una validación cruzada.
        self.modeloRuta=self.configuracion.modeloPreentrenado.replace("/","") # String para almacenar una ruta del modelo sin /.
        self.conjuntoEntrenamientoParaPrueba=[] # Vector para almacenar el corpus con el que se ha realizado un k fold de la validación cruzada.
        self.conjuntoValidacionParaPrueba=[] # Vector para almacenar los archivos reservados para prueba en un k fold de la validación cruzada.
        if self.configuracion.numDenunciasPorTipo==-1: # Inicialización de la cardinalizad de cada tipo de denuncia.
            self.numNoOdio=len(self.delitosnodio) # Si el parámetro está a -1 nos quedamos con todas la denuncias de cada tipo.
            self.numOdio=len(self.delitosodio)
        else: # Si el parámetro es diferente guardamos ese número de denuncias.
            self.numNoOdio = self.configuracion.getNumDenunciasPorTipo
            self.numOdio = self.configuracion.getNumDenunciasPorTipo


    def funcionTokenizadora(self, textos):
        """
            Función tokenizadora del modelo
            Parámetros:
                -textos: Lista con los textos a tokenizar
            Devuelve:
                - Instancia del tokenizador con los textos.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.configuracion.getModeloPreentrenado)
        return self.tokenizer(textos["text"], truncation=True)

    def etiquetar(self, numArchivos):
        """
            Función para asignar la etiqueta correspondiente a cada texto y almacenar la dupla en un vector.
            Parámetros:
                - numArchivos: Entero con el número de denuncias de cada tipo a tener en cuenta
        """
        print("Num delitos odio",len(self.delitosodio))
        print("Num delitos no odio", len(self.delitosnodio))
        listaTextos = []
        # Almacenamos una dupla con los campos texto y etiqueta en un vector listaTextos
        if numArchivos == -1:  # -1 cuando se quiere etiquetar el corpus completo.
            for archivo in self.delitosodio:
                textoEtiquetado = {
                    'label': 0,  # Delitos de ODIO etiqueta 0.
                    'text': archivo
                }
                listaTextos.append(textoEtiquetado)
            numOdio = len(listaTextos)
            for archivo in self.delitosnodio:
                textoEtiquetado = {
                    'label': 1, # Delitos de NO ODIO etiqueta 1.
                    'text': archivo
                }
                listaTextos.append(textoEtiquetado)
            numNoOdio = len(listaTextos)
        else:  # Para cualquier número distinto de -1 se nos quedaremos con las primeras numArchivos denuncias.
            import itertools
            for archivo in itertools.islice(self.delitosodio, numArchivos):
                textoEtiquetado = {
                    'label': 0,  # Delitos de ODIO etiqueta 0.
                    'text': archivo
                }
                listaTextos.append(textoEtiquetado)
            numOdio=len(listaTextos)

            for archivo in itertools.islice(self.delitosnodio, numArchivos):
                textoEtiquetado = {
                    'label': 1, # Delitos de NO ODIO etiqueta 1.
                    'text': archivo
                }
                listaTextos.append(textoEtiquetado)
            numNoOdio = len(listaTextos)

        print("--------------Textos totales--------------", len(listaTextos),"| ODIO",numOdio,"/",len(self.delitosodio),"| NO ODIO",numNoOdio-numOdio,"/",len(self.delitosnodio))

        return listaTextos
    def muestraMetricas(self, etiquetas, predicciones):
        """
            Función que muestra el recall, precision y f1 dado un cojunto de etiquetas y prediccciones. Devuelve un reporte de métricas.
            Parámetros:
                - etiquetas: Vector de enteros con las etiquetas originales.
                - predicciones: Vector de enteros con las etiquetas predichas por el sistema.
        """
        from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
        # Se calculan las métricas con el módulo metrics de Sk-learn.
        print("")
        print("")
        print("------------------------METRICAS-------------------------")
        print(f"Precision", precision_score(etiquetas, predicciones, average='macro'))
        print(f"Macro-f1", f1_score(etiquetas, predicciones, average='macro'))
        print(f"Recall", recall_score(etiquetas, predicciones, average='macro'))
        reporte = classification_report(etiquetas, predicciones)
        print(reporte)
        print("")
        return reporte
    def metricasComputo(self, prediccion):
        """
            Función para calcular métricas en mitad de una ejecución de un modelo BERT. Función orientativa no representativa del rendimiento del modelo.
            Parámetros:
                - prediccion: Variable transformers que contiene los datos de una predicción y de las etiquetas originales.
        """
        # Evaluar
        import numpy as np
        from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
        # Extracción de los datos con Numpy
        predicciones, etiquetas = prediccion
        predicciones = np.argmax(predicciones, axis=1)

        print("Validación", etiquetas)
        print("\n")
        print("Predicciones", predicciones)
        # Cálculo de las métricas con el módulo metrics de Sk-learn.
        recall = recall_score(etiquetas, predicciones, average='macro')
        accuracy = accuracy_score(etiquetas, predicciones)
        precision = precision_score(etiquetas, predicciones, average='macro')
        f1score = f1_score(etiquetas, predicciones, average='macro')

        reporte=self.muestraMetricas(etiquetas, predicciones)

        return {'recall': recall,  'accuracy': accuracy,
                 'precision': precision
            , 'f1': f1score, 'reporte': reporte}


    def entrenar(self, numArchivos):
        """
            Función de decisión del tipo de entrenamiento a realizar.
            Parámetros:
                - numArchivos: Entero con el número de archivos a etiquetar en el entrenamiento.
        """

        listaTextos = self.etiquetar(numArchivos)

        if self.configuracion.getModeloPreentrenado in self.modelosBert:  # Si se trata de un modelo BERT
            # Conversión de la lista etiquetada a panda
            import pandas as pd
            dataFrame = pd.DataFrame(listaTextos)

            # Conversión de Panda a Dataset
            from datasets import Dataset
            dataset = Dataset.from_pandas(dataFrame)

            # Tokenizacion
            datasetTokenizado = dataset.map(self.funcionTokenizadora, batched=True)
            self.entrenarBERT(datasetTokenizado, datasetTokenizado, self.metricasComputo)
        if self.configuracion.getModeloPreentrenado in self.modelosNOBert: # Si se trata de un modelo No Bert
            self.entrenarNoBERT(listaTextos)

    def entrenarNoBERT(self, listaTextos):
        """
            Función para entrenar un modelo no BERT.
            Parámetros:
                - listaTextos: Lista con las denuncias etiquedas a clasificar.
        """
        print("---------------------------------------Entrenando Modelo NO BERT------------------------------------")
        textos=[]
        etiquetas=[]
        import spacy.cli
        spacy.cli.download("es_core_news_sm")
        import es_core_news_sm
        def tokenizar(texto): # Función de tokenización usando spacy

            nlp = es_core_news_sm.load()
            texto = nlp(texto)
            tokens = []
            textoProcesado = ""
            for token in texto:
                if token.text.isalpha() and not token.is_stop:  # Quitamos signos de puntuación y stop words
                    tokens.append(token.lemma_.lower())  # Reducimos a la raiz cada palabra y la pasamos a minúscula
            textoProcesado = " ".join(tokens)
            # print(texto,"-",textoProcesado,end="\n")
            return textoProcesado


        for elemento in listaTextos: # Separamos en dos vectores, vector etiquetas y vector de textos
            textos.append(tokenizar(elemento['text']))
            etiquetas.append(elemento['label'])
        # Condiciones para seleccionar el tipo de modelo según la variable del fichero de parámetros
        if self.configuracion.getModeloPreentrenado == "SVM":
            from sklearn import svm
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer()  # Se usa un tfidf para ambos
            modelo = svm.SVC() # Creamos un modelo Support Machine Vector

        if self.configuracion.getModeloPreentrenado == "Naive":
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer()   # Se usa un tfidf para ambos
            modelo = MultinomialNB(alpha=0) # Instanciamos un modelo Naive Bayes
        print("Entrenando Modelo")
        textosFit = vectorizer.fit_transform(textos)
        model = modelo
        model.fit(textosFit, etiquetas)  # Entrenamos el modelo

        from joblib import dump, load
        # Guardamos el modelo creado en una carpeta para su posterior acceso desde la web
        os.chdir("Modelos")
        nombreCarpeta=self.configuracion.getModeloPreentrenado+"-"+self.configuracion.getComentario # Creamos una carpeta única para el modelo
        os.mkdir(nombreCarpeta)
        os.chdir(nombreCarpeta)
        dump(model, 'Modelo.joblib')  # Función de la biblioteca joblib para persistir el modelo
        dump(vectorizer,'Vectorizador.joblib')
        os.chdir("..")
        os.chdir("..")
    def entrenarBERT(self, datasetTokenizadoTrain, datasetTokenizadoVal, metricas_Computo):
        """
            Función para entrenar un modelo BERT.
            Parámetros:
                - datasetTokenizadoTrain: Conjunto de entrenamiento en formato correcto para alimentar al modelo.
                - datasetTokenizadoVal: Conjunto de validación en formato correcto para alimentar al modelo.
                - metricas_Computo: Función de calculo de las métricas en mitad de un entrenamiento.
        """
        print("---------------------------------------Entrenando Modelo  BERT------------------------------------")

        # Etiquetas
        label1 = {0: "ODIO", 1: "NO ODIO"}  # Definimos las etiquetas de forma numérica
        label2 = {"ODIO": 0, "NO ODIO": 1}

        from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

        model = AutoModelForSequenceClassification.from_pretrained( # Definimos un modelo con sus etiquetas
            self.configuracion.getModeloPreentrenado, num_labels=2, id2label=label1, label2id=label2
        )
        from transformers import DataCollatorWithPadding
        # Parámetros de entrenamiento
        '''
            - evaluation-strategy: Estrategia de evaluación, puede ser por épocas.
            - save_strategy: Estrategia de guardado, puede ser por épocas.
            - learning_rate: Ratio de entrenamiento del modelo.
            - per_device_train_batch_size: Tamaño de los lotes del conjunto de entrenamiento.
            - per_devive_val_barch_size: Tamaño de los lotes del conjunto de validación.
            - num_train_epochs: Número de épocas del entrenamiento, veces que se entrena.
            - weight_decay: Caída o deterioro de los pesos.
            - load_best_model_at_end: Variable que indica si se carga el mejor modelo al final.
            - push_to_hub: Variable que indica si se sube al sitio web el modelo.
            -output_dir: Directorio en el que guardar el modelo.
        '''
        parametrosEntrenamiento = TrainingArguments(weight_decay=0.01,num_train_epochs=2,learning_rate=2e-5,load_best_model_at_end=True,
            push_to_hub=False,per_device_train_batch_size=16,per_device_eval_batch_size=16,
            output_dir="../TFG/Modelos/" + self.modeloRuta + "_" + self.configuracion.getComentario,evaluation_strategy="epoch",save_strategy="epoch",
        )
        # Parámetros del entrenador
        '''
            - model: Nombre del modelo con el que entrenar.
            - args: Parámetros de entrenamiento.
            - train_dataset: Conjunto de datos de entrenamiento.
            - val_dataset: Conjunto de datos de validación.
            - tokenizer: Tokenizador.
            - data_collator: Colector de datos.
            - compute_metrics: Instancia de las variables de cómputo.
        '''
        entrenador = Trainer(model=model, args=parametrosEntrenamiento, data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer), compute_metrics=metricas_Computo, train_dataset=datasetTokenizadoTrain,
                             eval_dataset=datasetTokenizadoVal, tokenizer=self.tokenizer,
                             )

        entrenador.train()
        metricas = entrenador.evaluate()
        print(metricas)

        ruta = "../TFG/Metricas/" + self.modeloRuta + "Metricas.csv"
        import sys
        # Métricas no representativas de la validación, únicamente utilizadas para medir el rendimiento de los parámetros
        salida = sys.stdout
        sys.stdout = open(ruta, 'w')
        print(metricas)
        sys.stdout = salida

        return model

    def escribeLineaCSV(self, ruta, linea):
        """
            Función para realizar la escritura de una línea en un archivo csv
            Parámetros:
                - ruta: String con la ruta del csv.
                - linea: String con la línea a escribir en dichi csv.
        """
        from csv import writer # Se utiliza la bliblioteca csv
        with open(ruta, 'a', newline='') as f:
            objetoDeEscritura = writer(f)
            objetoDeEscritura.writerow("")
            objetoDeEscritura.writerow("")
            objetoDeEscritura.writerow(linea)
            f.close()

    def calculaMediaMetricas(self, k):
        """
            Función para calcular el rendimiento medio de un modelo en la validación cruzada
            Parámetros:
                - k: Numero de divisiones en la validación k-fold, es decir, el número de métricas sobre las que realizar la media.
        """
        import pandas as pd
        vectorMetricas = []
        ruta = "../TFG/Metricas/" + self.modeloRuta + "Metricas.csv" # Definimos ruta y escribimos el nombre del modelo.
        self.escribeLineaCSV(ruta, [self.configuracion.getModeloPreentrenado.upper()])

        for metrica in self.metricas:  # Bucle para convertir cada métrica recibida en un formato iterable como una lista
            vector = []
            import pandas as pd
            metrica.to_csv(ruta, mode='a')
            print(metrica)
            print("")
            i = 0
            j = 0
            while i < 5:
                j = 0
                while j < 4:
                    vector.append(metrica.iloc[i, j])
                    j = j + 1
                i = i + 1

            vectorMetricas.append(vector) #Convertimos cada metrica a un vector y lo almacenamos en otro vector de metricas iterables.

        vSum = []
        for posicion in vectorMetricas[0]: #Inicializamos vector de sumas finales
            vSum.append(0)
        import numpy as np
        for elemento in vectorMetricas:
            elemento[8] = 99 #Posicones vacias las declaramos como número para evitar fallo al interar
            elemento[9] = 99 #Posiciones vacías las declaramos como número para evitar fallo al interar
            i = 0
            for posicion in elemento: # Cálculo de la suma
                vSum[i] = vSum[i] + elemento[i]
                i = i + 1

        # print(vSum)
        i = 0
        for elemento in vSum: # Cálculo de la media.
            if vSum[i] == 0:
                vSum[i] = 0
            else:
                vSum[i] = vSum[i] / k
            i = i + 1
        # print(vSum)

        datos = {  # Definimos una estructura para el visualizado de la informaciónn obteniendola de los vectores calculados.
            'precision': [vSum[0], vSum[4], "-", vSum[12], vSum[16]],
            'recall': [vSum[1], vSum[5], "-", vSum[13], vSum[17]],
            'f1-score': [vSum[2], vSum[6], vSum[10], vSum[14], vSum[18]],
            'support': [vSum[3], vSum[7], vSum[11], vSum[15], vSum[19]]

        }
        # Visualización y guardado de resultados
        metrica = pd.DataFrame(datos, index=['0', '1', 'accuracy', 'macro avg', 'weighted avg'])
        print("------------------------MEDIA DE LAS ", k, " ITERACIONES---------------------------(" + self.configuracion.getModeloPreentrenado +"_" + self.configuracion.getComentario + ")")
        print(metrica)
        rutaMedia = "../TFG/Metricas/" + "Media_Metricas.csv"

        self.escribeLineaCSV(rutaMedia, [self.configuracion.getModeloPreentrenado.upper(), self.configuracion.getComentario.upper(), "Tiempo", self.tiempoValidacionCruzada, "K-folds", self.configuracion.getK, "NumTextos", self.configuracion.getNumDenunciasPorTipo])

        metrica.to_csv(rutaMedia, mode='a')

    def extraeMetricas(self, reporte):
        """
            Función para transformar un reporte de métricas en un vector indexado que es almacenado en la variable métricas.
            Parámetros:
                - reporte: String con el reporte a trasnformar en un formato iterable
        """
        # Extracción de las líneas
        lineas = reporte.split('\n')
        linea1 = lineas[2]
        linea2 = lineas[3]
        linea3 = lineas[5]
        linea4 = lineas[6]
        linea5 = lineas[7]
        print(linea3)
        if linea2.split()[0] == "1":
            aux = linea2
        # Corte en vectores de palabras por línea
        vectorLinea1 = linea1.split()
        vectorLinea2 = linea2.split()
        vectorLinea3 = linea3.split()
        vectorLinea4 = linea4.split()
        vectorLinea5 = linea5.split()

        import pandas as pd
        # print(vectorLinea5)
        # Creación de la estructura en base al acceso de los vectores anteriormente calculados.
        datos = {'precision': [float(vectorLinea1[1]), float(vectorLinea2[1]), "-", float(vectorLinea4[2]),
                               float(vectorLinea5[2])],
                 'recall': [float(vectorLinea1[2]), float(vectorLinea2[2]), "-", float(vectorLinea4[3]),
                            float(vectorLinea5[3])],
                 'f1-score': [float(vectorLinea1[3]), float(vectorLinea2[3]), float(vectorLinea3[1]),
                              float(vectorLinea4[4]), float(vectorLinea5[4])],
                 'support': [int(vectorLinea1[4]), int(vectorLinea2[4]), int(vectorLinea3[2]), float(vectorLinea4[5]),
                             float(vectorLinea5[5])]
                 }
        # Definimos un dataframe con las cabeceras.
        dataframeCabeceras = pd.DataFrame(datos, index=['0', '1', 'accuracy', 'macro avg', 'weighted avg'])

        self.metricas.append(dataframeCabeceras)

    def validacionCruzadaBERT(self, numTextos, k):  # cross validation k-fold
        """
            Función para realizar la validación cruzada de un modelo BERT.
            Validación k fold donde se divide el corpus en k subconjuntos, n-1 se usan para el entrenamiento y el restante para la validación
            Parámetros:
                - numTextos: Entero con el número de denuncias de cada tipo a tratar.
                - k: Entero con el número de divisiones/iteraciones en una validación k fold.
        """
        listaTextos = self.etiquetar(numTextos)

        import time
        inicio=time.time()

        from sklearn.model_selection import KFold, StratifiedKFold
        import pandas as pd
        # Se divide en conjunto en numDivisiones subconjuntos, 1 se usará para la validación y los demás para train
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.configuracion.getSemilla)  # K fold que mantiene proporcion de etiquetas
        texts = []
        eti = []
        for elemento in listaTextos:
            texts.append(elemento['text'])
            eti.append(elemento['label'])

        # Extraemos los índices resultantes de la validación k fold
        iteracion = 0
        for indices_train, indices_val in kf.split(texts, eti):
            import shutil  # Reseteamos carpeta de modelos
            if os.path.exists("../TFG/Modelos/"): # Para ahorrar espacio en caso de tener varias ejecuciones
                shutil.rmtree("../TFG/Modelos/")
            train=[]
            val=[]

            contadorClase0Entrenamiento = 0
            contadorClase1Entrenamiento = 0
            contadorClase0Validacion = 0
            contadorClase1Validacion = 0
            # Conteo del número de índices de entrenamiento de cada clase
            for i in indices_train:
                print("Train:",listaTextos[i]['text'])
                train.append(listaTextos[i])

                if listaTextos[i]['label'] == 0:
                    contadorClase0Entrenamiento = contadorClase0Entrenamiento + 1
                else:
                    contadorClase1Entrenamiento = contadorClase1Entrenamiento + 1

            # Conteo del número de índices de validación de cada clase
            for i in indices_val:
                print("Val:", listaTextos[i]['text'])
                val.append(listaTextos[i])

                if listaTextos[i]['label'] == 0:
                    contadorClase0Validacion = contadorClase0Validacion + 1
                else:
                    contadorClase1Validacion = contadorClase1Validacion + 1

            print("Iteración=", iteracion)
            iteracion = iteracion + 1
            print("Índices entrenemiento:", indices_train)
            print("Etiquetas 0:", contadorClase0Entrenamiento, "Etiquetas 1:", contadorClase1Entrenamiento)
            print("Índices validación:", indices_val)
            print("Etiquetas 0:", contadorClase0Validacion, "Etiquetas 1:", contadorClase1Validacion)

            # Conversión a panda
            from datasets import Dataset
            datasetPandaTrain = pd.DataFrame(train)
            datasetPandaVal = pd.DataFrame(val)

            # Conversión de Panda a Dataset

            datasetDatasetTrain= Dataset.from_pandas(datasetPandaTrain)
            datasetDatasetVal = Dataset.from_pandas(datasetPandaVal)

            # Tokenizacion
            datasetTokenizadoTrain = datasetDatasetTrain.map(self.funcionTokenizadora, batched=True)
            datasetTokenizadoVal = datasetDatasetVal.map(self.funcionTokenizadora, batched=True)
            # Le pasamos al entrenamiento como conjunto validador el de entrenamiento también para evitar el leak de clases de validación
            model = self.entrenarBERT(datasetTokenizadoTrain, datasetTokenizadoTrain, self.metricasComputo)
            # Definimos el clasificador con el modelo entrenado
            classifier = pipeline("text-classification", truncation=True, model=model, tokenizer=self.tokenizer,
                                  batch_size=16)

            i = 0
            predicciones = []
            etiquetas = []
            # Predecimos todas las etiquetas de los textos del conjunto de validación
            while i < len(listaTextos):
                if i in indices_val:
                    etiquetas.append(listaTextos[i]['label'])
                    clasificacion = classifier(listaTextos[i]['text'])
                    print(clasificacion)
                    print(str(clasificacion)[12:16])
                    if str(clasificacion)[12:16] == 'ODIO': # Extracción de etiquetas
                        predicciones.append(0)
                    else:
                        predicciones.append(1)
                i = i + 1
            print("Indices val", indices_val)
            print("Etiq", etiquetas)
            print("Pred", predicciones)
            report = classification_report(etiquetas, predicciones) # Genración del reporte de métricas
            print(report)

            self.extraeMetricas(report)
            print("Nueva metrica")

        fin=time.time()
        self.tiempoValidacionCruzada = fin - inicio
        self.calculaMediaMetricas(k)
        self.crearCarpetasValidacionCruzada(indices_train, indices_val)  # Guardado de la carpeta de validación cruzada


    def validacionCruzadaNoBERT(self, numTextos, k, vectorizador, modelo):
        """
            Función para realizar la validación cruzada de un modelo No BERT.
            Validación k fold donde se divide el corpus en k subconjuntos, n-1 se usan para el entrenamiento y el restante para la validación
            Parámetros:
                - numTextos: Entero con el número de denuncias de cada tipo a tratar.
                - k: Entero con el número de divisiones/iteraciones en una validación k fold.
                - vectorizador: Variable con el vectorizador
                - modelo: Variable con el modelo
        """
        self.modelo="SVM-TFIDVectorizer"
        import random
        import time
        inicio=time.time()
        import spacy.cli
        spacy.cli.download("es_core_news_sm")
        import es_core_news_sm
        listaTextos = self.etiquetar(numTextos)

        random.shuffle(listaTextos)

        from sklearn.model_selection import KFold, StratifiedKFold
        import pandas as pd
        # Se divide en conjunto en numDivisiones subconjuntos, 1 se usará para la validación y los demás para train
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.configuracion.getSemilla)  # Mantiene proporcion de etiquetas

        texts = []
        etiq = []
        for elemento in listaTextos: # Se separan en dos vectores distintos las etiquetas y los textos
            texts.append(elemento['text'])
            etiq.append(elemento['label'])

        iteracion = 0
        for indices_train, indices_val in kf.split(texts, etiq): # Obtenemos los índices
            for i in indices_train:
                print("Train:",listaTextos[i]['text'])
            for i in indices_val:
                print("Val:", listaTextos[i]['text'])
            self.conjuntoEntrenamientoParaPrueba = indices_train  # El último conjunto usado en la validación cruzada sera el usado para la prueba
            self.conjuntoValidacionParaPrueba = indices_val
            print("Iteración=", iteracion)
            iteracion = iteracion + 1
            print("Índices entrenemiento:", indices_train)
            contadorEntrenamiento0 = 0
            contadorEntrenamiento1 = 0
            contadorValidacion0 = 0
            contadorValidacion1 = 0

            import spacy.cli
            spacy.cli.download("es_core_news_sm")
            import es_core_news_sm
            def tokenizar(texto):  # Función de tokenización

                nlp = es_core_news_sm.load()
                texto = nlp(texto)
                tokens = []
                textoProcesado = ""
                for token in texto:
                    if token.text.isalpha() and not token.is_stop:  # Quitamos signos de puntuación y stop words
                        tokens.append(token.lemma_.lower())  # Reducimos a la raiz cada palabra y la pasamos a minúscula
                textoProcesado = " ".join(tokens)
                # print(texto,"-",textoProcesado,end="\n")
                return textoProcesado

            trainFinalTexto = []
            trainFinalEtiqueta = []
            valFinalTexto = []
            valFinalEtiqueta = []

            # Conteo de las etiquetas de entrenamiento por tipo
            for i in indices_train:
                # print(listaTextos[i])
                if listaTextos[i]['label'] == 0:
                    contadorEntrenamiento0 = contadorEntrenamiento0 + 1
                else:
                    contadorEntrenamiento1 = contadorEntrenamiento1 + 1
                #print("Indice train", i)
                trainFinalTexto.append(tokenizar(listaTextos[i]['text']))
                trainFinalEtiqueta.append(listaTextos[i]['label'])

            print("Etiquetas 0:", contadorEntrenamiento0, "Etiquetas 1:", contadorEntrenamiento1)
            print("Índices validación:", indices_val)
            # Conteo de las etiquetas de validación por tipo
            for i in indices_val:
                # print(listaTextos[i])
                if listaTextos[i]['label'] == 0:
                    contadorValidacion0 = contadorValidacion0 + 1
                else:
                    contadorValidacion1 = contadorValidacion1 + 1
                #print("Indice val", i)
                valFinalTexto.append(tokenizar(listaTextos[i]['text']))
                valFinalEtiqueta.append(listaTextos[i]['label'])
            print("Etiquetas 0:", contadorValidacion0, "Etiquetas 1:", contadorValidacion1)
            infactibilidad =False #Comprobación de que no entrenamos con modelos de validación
            for elemento in trainFinalTexto:
                for elemento2 in valFinalTexto:
                    if elemento==elemento2:
                        infactibilidad=True

            # Modelo, entrenamiento
            vectorizer = vectorizador
            textosFit = vectorizer.fit_transform(trainFinalTexto)  # Textos ajustados al vectorizador
            etiq=trainFinalEtiqueta  # Cojunto de etiquetas
            model = modelo
            model.fit(textosFit, etiq)

            i = 0
            predicciones = []
            etiquetas = []
            for elemento in valFinalTexto:  # Generación de prediccciones
                Y = vectorizer.transform([elemento]) # Se pasa a una lista.
                prediction = model.predict(Y)
                predicciones.append(prediction[0])
            #print("Indices val", indices_val)
            print("Etiq", valFinalEtiqueta)
            print("Pred", predicciones)
            report = classification_report(valFinalEtiqueta, predicciones) # Generación del reporte
            print(report)

            self.extraeMetricas(report)
            print("Se ha entrenado con texto de validacion?", infactibilidad)

        fin=time.time()
        self.tiempoValidacionCruzada=fin-inicio
        self.calculaMediaMetricas(k)
        self.crearCarpetasValidacionCruzada(indices_train, indices_val)

    def validaciónCruzada(self):
        """
            Función para invocar una función de validación cruzada u otra en función del fichero de parámetros
        """
        print(self.modeloRuta)
        print(self.configuracion.getK)
        print(self.configuracion.getNumDenunciasPorTipo)
        print("Validación Cruzada Modelo: "+self.configuracion.getModeloPreentrenado)
        if self.configuracion.getModeloPreentrenado in self.modelosBert: # Si se trata de un modelo BERT
            self.validacionCruzadaBERT(self.configuracion.getNumDenunciasPorTipo, self.configuracion.getK)
        else: # Si no se trata de un modelo BERT

            if self.configuracion.getModeloPreentrenado=="SVM":  # Support Machine Vector
                from sklearn import svm
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer()
                modelo = svm.SVC()
                self.validacionCruzadaNoBERT(self.configuracion.getNumDenunciasPorTipo, self.configuracion.getK, vectorizer, modelo)
            if self.configuracion.getModeloPreentrenado=="Naive": # Naice Bayes
                from sklearn.naive_bayes import MultinomialNB
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer()
                modelo = MultinomialNB(alpha=0)
                self.validacionCruzadaNoBERT(self.configuracion.getNumDenunciasPorTipo, self.configuracion.getK, vectorizer, modelo)
            else:
                print("Modelo incorrecto")

    def crearCarpetasValidacionCruzada(self, train, val):
        """
            Función para crear las carpetas de la la última división k fold.
            Crea una carpeta con el corpus entrenamiento de una validación cruzada (la última k) y sus respectivos archivos de validación.
        """
        print(os.getcwd())
        os.chdir("./Corpus")
        carpetas=os.listdir()
        print(carpetas)
        for indice in train:   # Buscamos todos los archivos de entrenamiento
            print(indice)
            import shutil
            if indice < self.numNoOdio:  # Indices de NO ODIO
                os.chdir(carpetas[0])
                print("noodio")
                print(indice)
                print(os.getcwd())
                listaTextos=os.listdir() # Se realiza una copia del archivo con la función shutil
                carpetaInicial = r'C:\Users\Pc\Documents\GitHub\TFG\Corpus\NOODIO'
                carpetaFinal = r'C:\Users\Pc\Documents\GitHub\TFG\PruebaEntrenamiento\NOODIO'
                shutil.copyfile(os.path.join(carpetaInicial,listaTextos[indice]), os.path.join(carpetaFinal,listaTextos[indice]))
                os.chdir("..")
                print(os.getcwd())
            else:  # Índices de ODIO
                os.chdir(carpetas[1])
                print("odio")
                print(indice)
                print(os.getcwd())
                listaTextos = os.listdir() # Se realiza una copia del archivo con la función shutil
                carpetaInicial = r'C:\Users\Pc\Documents\GitHub\TFG\Corpus\ODIO'
                carpetaFinal = r'C:\Users\Pc\Documents\GitHub\TFG\PruebaEntrenamiento\ODIO'
                print("indice",indice)
                ind=indice-self.numNoOdio # Descartamos los índices de No Odio ya que hemos cambiado de carpeta
                print("indice en carpeta",ind)
                print("nombre",listaTextos[indice-self.numNoOdio])
                shutil.copyfile(os.path.join(carpetaInicial, listaTextos[indice-self.numNoOdio]), os.path.join(carpetaFinal, listaTextos[indice-self.numNoOdio]))
                os.chdir("..")
                print(os.getcwd())

        print(train)
        for indice in val: # Buscamos todos los archivos de validación
            print(indice)
            import shutil
            if indice < self.numNoOdio:  # Indices de NO ODIO
                os.chdir(carpetas[0])
                print("noodio")
                print(indice)
                print(os.getcwd())
                listaTextos=os.listdir()
                # Se realiza una copia del archivo con la función shutil
                carpetaInicial = r'C:\Users\Pc\Documents\GitHub\TFG\Corpus\NOODIO'
                carpetaFinal = r'C:\Users\Pc\Documents\GitHub\TFG\PruebaValidacion\NOODIO'
                shutil.copyfile(os.path.join(carpetaInicial,listaTextos[indice]), os.path.join(carpetaFinal,listaTextos[indice]))
                os.chdir("..")
                print(os.getcwd())
            else:  # Indices de ODIO
                os.chdir(carpetas[1])
                print("odio")
                print(indice)
                print(os.getcwd())
                listaTextos = os.listdir() # Se realiza una copia del archivo con la función shutil
                carpetaInicial = r'C:\Users\Pc\Documents\GitHub\TFG\Corpus\ODIO'
                carpetaFinal = r'C:\Users\Pc\Documents\GitHub\TFG\PruebaValidacion\ODIO'
                print("indice",indice)
                ind = indice - self.numNoOdio   # Descartamos los índices de No Odio ya que hemos cambiado de carpeta
                print("indice en carpeta",ind)
                print("nombre",listaTextos[indice - self.numNoOdio])

                shutil.copyfile(os.path.join(carpetaInicial, listaTextos[indice-self.numNoOdio]), os.path.join(carpetaFinal, listaTextos[indice-self.numNoOdio]))
                os.chdir("..")
                print(os.getcwd())
        print(val)
