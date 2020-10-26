# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 09:49:37 2017

@author: Waldo Hasperué

Función train
-------------
Parámetros:
       P: es la matriz con los datos de los patrones con los cuales
           entrenar el perceptrón. Los ejemplos deben estar en filas.
       T: es un vector con la clase esperada para cada ejemplo. Los
           valores de las clases deben ser 0 (cero) y 1 (uno)
       alfa: velocidad de aprendizaje
       max_itera: la cantidad máxima de iteraciones en las cuales se va a
           ejecutar el algoritmo
       dibujar: si vale True (y los datos son en dos dimensiones) dibuja los
           datos y la recta discriminante.

Devuelve:
       W: la matriz de pesos W del percpetrón entrenado
       b: valor del bias (W0) del perceptrón entrenado
       ite: número de iteraciones ejecutadas durante el algoritmo. Si
           devuelve el mismo valor que MAX_ITERA es porque no pudo finalizar
           con el entrenamiento

Ejemplo de uso:
       [W, b, ite] = train(P, T, 0.25, 250, True);
       
*******************************************************************************
       
Función plot
-------------       
Parámetros:
       P: es una matriz con los datos de los patrones con los cuales
           entrenar el perceptrón. Los ejemplos deben estar en filas.
       T: es un vector con la clase esperada para cada ejemplo. Los
           valores de las clases deben ser 0 (cero) y 1 (uno)
       W: la matriz de pesos W del percpetrón entrenado
       b: valor del bias (W0) del perceptrón entrenado
       title: (Opcional) el título del gráfico

Ejemplo de uso:
       plot(P, T, W, b);


"""

import numpy as np
import matplotlib.pyplot as plt

def plot(P, T, W, b, title=""):
    plt.clf()
    
    #ceros
    x0=[]
    y0=[]
    x1=[]
    y1=[]
    for i in range(len(T)):
        if T[i] == 0:
            x0.append(P[i, 0])
            y0.append(P[i, 1])
        else:
            x1.append(P[i, 0])
            y1.append(P[i, 1])
            
    plt.scatter(x0, y0, marker='+', color='b')            
    plt.scatter(x1, y1, marker='o', color='g')
    
    #ejes
    minimos = np.min(P, axis=0)
    maximos = np.max(P, axis=0)
    diferencias = maximos - minimos
    minimos = minimos - diferencias * 0.1
    maximos = maximos + diferencias * 0.1
    plt.axis([minimos[0], maximos[0], minimos[-1], maximos[1]])
    plt.axis('off')
    
    #recta discriminante
    m = W[0,0] / W[0,1] * -1
    n = b / W[0,1] * -1
    x1 = minimos[0]
    y1 = x1.dot(W[0])
    x2 = maximos[0]
    y2 = x2.dot(W[0])
    plt.plot([x1, x2],[y1, y2], color='r')
    
    plt.title(title)
    
    plt.draw()
    plt.pause(0.0001)   

def train(P, T, alfa, max_itera, dibujar):
        
    (cant_patrones, cant_atrib) = P.shape
    W = np.random.rand(1, cant_atrib)
    b = np.random.rand()
    
    ite = 0
    otra_vez = True
    
    plt.ion()
    plt.show()
    
    while ((ite <= max_itera) and otra_vez):
        otra_vez = False
        ite = ite + 1
        
        for patr in range(cant_patrones):
            salida = b + W.dot(P[patr, :][np.newaxis].T) 
            if salida >= 0:
                salida = 1
            else:
                salida = 0
      
            factor = alfa * (T[patr] - salida)
            if (factor != 0):
                otra_vez = True
                W = W + factor * P[patr, :][np.newaxis]
                b = b + factor
                
        
        if dibujar and (cant_atrib == 2):        
            plot(P, T, W, b, 'Iteración: ' + str(ite))

    if dibujar and (cant_atrib == 2):        
        plot(P, T, W, b, 'Iteración: ' + str(ite))
        
    return (W, b, ite)