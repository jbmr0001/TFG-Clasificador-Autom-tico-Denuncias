import os

from lecturaArchivos import lecturaArchivos
from configuracion import configuracion
from procesaTexto import procesaTexto
from modelo import modelo

import wandb
#wandb.login(key="5f790623c06f92b02135f7c4e163290e127fd49b")
os.environ["WANDB_DISABLED"] = "true"


# ----------------------------GARGA DE LOS ARCHIVOS DEL CORPUS-------------------------
configuracion = configuracion()
configuracion.lecturaParametros()
print("Modo: ",configuracion.modo)
print("Procesamiento: ",configuracion.procesamiento)
print("Semilla: ",configuracion.semilla)
print("Número de denuncias por tipo: ",configuracion.numDenunciasPorTipo)
lectura = lecturaArchivos(configuracion)
lectura.cargar()

# ----------------------------PRE-PROCESAMIENTO DE CORPUS-------------------------
limpia = procesaTexto(lectura.getArchivosODIO, lectura.getArchivosNOODIO)
if configuracion.procesamiento == 1:
    limpia.procesamiento()
if configuracion.procesamiento == 2:
    limpia.procesamiento2()
if configuracion.procesamiento == 3:
    limpia.procesamiento3()
if configuracion.procesamiento == 4:
    limpia.procesamiento4(tenerEnCuentaStopWords=False)
if configuracion.procesamiento == 5:
    limpia.procesamiento4(tenerEnCuentaStopWords=True)

#lectura.muestraArchivosODIO
#lectura.muestraArchivosNOODIO

#limpia.mostrarPalabrasRepetidasYFrecuenciaPorClase()

# ----------------------------EJECUCIÓN DEL MODELO-------------------------
modelo = modelo(lectura.getArchivosODIO, lectura.getArchivosNOODIO,configuracion)
if configuracion.modo=="Entrenamiento":
    modelo.entrenar(configuracion.getNumDenunciasPorTipo)#-1 == Con todos los archivos
if configuracion.modo=="Validacion":
    modelo.validaciónCruzada()





