# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:22:43 2019

@author: Waldo Hasperué
"""

import simulador
import bpn
import numpy as np

circuito = 'Circuito1.jpg'

#%%
###############################################################################
#                                                                             #
#                       Generar datos para el entrenamiento                   #
#                                                                             #
###############################################################################

# Establecer la cantidad y la dirección de cada sensor de distancia al borde.
# El valor 0 corresopnde al frente del auto
sensores = [-45, 0, 45]

# La conducción del vehículo es manual. El auto seguirá el cursor del mouse
simulador.conduccionManual(sensores, circuito)


#%%
###############################################################################
#                                                                             #
#                              Entrenar el modelo                             #
#                                                                             #
###############################################################################

def entrenarModelo(data, ocultas, alfa, momento, fun_oculta, fun_salida, max_ite, cota_error):
    if(data == None):
        print ("ERROR: No se establecieron los datos de entrenamiento")
        return None
        
    X = data.copy()

    X = np.random.permutation(X)
    T = X[:, -2:]
    P = X[:, 0:(data.shape[1]-2)]

    column_min_values = np.min(P, axis=0)
    column_max_values = np.max(P, axis=0)
    P = (P - column_min_values) / (column_max_values - column_min_values)
    P = P * 2 -1
    
    modelo = bpn.train(P, T, T, ocultas, alfa, momento, fun_oculta, fun_salida, max_ite, cota_error, True)
       
    modelo = list(modelo)
    modelo[4] = fun_oculta 
    modelo[5] = fun_salida 
    
    return (modelo, column_min_values, column_max_values)

def continuarEntrenamiento(modelo, data, ocultas, alfa, momento, fun_oculta, fun_salida, max_ite, cota_error):
    if(data == None):
        print ("ERROR: No se establecieron los datos de entrenamiento")
        return None
        
    X = data.copy()

    X = np.random.permutation(X)
    T = X[:, -2:]
    P = X[:, 0:(data.shape[1]-2)]

    column_min_values = np.min(P, axis=0)
    column_max_values = np.max(P, axis=0)
    P = (P - column_min_values) / (column_max_values - column_min_values)
    P = P * 2 -1
    
    w_O = modelo[0][0]
    b_O = modelo[0][1]
    w_S = modelo[0][2]
    b_S = modelo[0][3]
    modelo = bpn.train_con_pesos(P, T, T, ocultas, alfa, momento, fun_oculta, fun_salida, max_ite, cota_error, True, w_O, b_O, w_S, b_S)
       
    modelo = list(modelo)
    modelo[4] = fun_oculta 
    modelo[5] = fun_salida 
    
    return (modelo, column_min_values, column_max_values)

#%%
# Pedir los datos para tener una copia (por si se desea tener varios conjuntos de entrenamiento)
datos = simulador.obtenerDatosDeSensores()

#%%
# Parámetros de la red
ocultas = 20
alfa = 0.01
momento = 0
max_ite = 1500
cota_error = 0.00001

fun_oculta = 'tansig' 
fun_salida = 'purelin'

#%%
# Entrenamiento
modelo = entrenarModelo(datos, ocultas, alfa, momento, fun_oculta, fun_salida, max_ite, cota_error)
#modelo = continuarEntrenamiento(modelo, datos, ocultas, alfa, momento, fun_oculta, fun_salida, max_ite, cota_error)


#%%
###############################################################################
#                                                                             #
#                               Testear el modelo                             #
#                                                                             #
###############################################################################

# Establecer la función que simula al conductor autómata
# Como parámetro recibe las distancias detectadas por los sensores
# Debe devolver como salida la velocidad y dirección de giro
def fConductor(distancias):
    global modelo
    
    column_min_values = modelo[1]
    column_max_values = modelo[2]
    
    distancias = (distancias - column_min_values) / (column_max_values - column_min_values)
    distancias = distancias * 2 - 1
    
    w_O = modelo[0][0]
    b_O = modelo[0][1]
    w_S = modelo[0][2]
    b_S = modelo[0][3]
    fun_oculta = modelo[0][4]
    fun_salida = modelo[0][5]
    
    neta_oculta = w_O.dot(distancias[np.newaxis].T) + b_O
    salida_oculta = eval('bpn.' + fun_oculta + '(neta_oculta)')
    neta_salida = w_S.dot(salida_oculta) + b_S
    salida_salida = eval('bpn.' + fun_salida + '(neta_salida)')

    return salida_salida

#%%
# Simular la conducción del vehículo usando al modelo como "conductor"    
simulador.conduccionAutonoma(sensores, circuito, fConductor)
