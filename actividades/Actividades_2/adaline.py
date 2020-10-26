# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:02:48 2017

@author: Waldo Hasperué

Función train
-------------
Parámetros:
       P: es una matriz con los datos de los patrones con los cuales
           entrenar el perceptrón. Los ejemplos deben estar en filas.
       T: es un vector con la clase esperada para cada ejemplo. Los
           valores de las clases deben ser acordes a la función de transferencia 
           utilizada:
               - para logsig: 0 (cero) y 1 (uno) 
               - para tansig: -1(menos uno) y 1 (uno)
       alfa: velocidad de aprendizaje
       max_itera: la cantidad máxima de iteraciones en las cuales se va a
           ejecutar el algoritmo
       cota_error: error promedio mínimo que se espera alcanzar como condición 
           de fin del algoritmo
       funcion: un string con el nombre de la función de transferencia a utilizar
           - 'logsig'
           - 'tansig'
       dibujar: si vale True (y los datos son en dos dimensiones) dibuja los
           ejemplos y la recta discriminante.

Devuelve:
       W: la matriz de pesos W del percpetrón entrenado
       b: valor del bias (W0) del perceptrón entrenado
       ite: número de iteraciones ejecutadas durante el algoritmo. Si
           devuelve el mismo valor que MAX_ITERA es porque no pudo finalizar
           con el entrenamiento
       error_prom: error promedio cometido en la última iteración del algoritmo

Ejemplo de uso:
       [W, b, ite, error_prom] = adaline.train(P, T, 0.01, 1000, 0.001, 'tansig', True)    
       
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
       title: el título que aparecerá en la gráfica

Ejemplo de uso:
       plot(P, T, W, b, 'Entrenamiento final del Adaline');


"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec

def plot(P, T, W, b, title = ''):
    plt.clf()
    
    gs = gridspec.GridSpec(1, 2)
    ax = plt.subplot(gs[0, 0])
    
    #Gráfica en 3D
    plt.title(title)
    bx = plt.subplot(gs[0, 1], projection='3d')
    
    #ceros
    x=[]
    y=[]
    z=[]
    for i in range(len(T)):
        if T[i] == 0:
            x.append(P[i, 0])
            y.append(P[i, 1])
            z.append(0)
    ax.scatter(x, y, marker='+', color='b')
    bx.scatter(x, y, z, marker='+', color='b')
    
    #unos
    x=[]
    y=[]
    z=[]
    for i in range(len(T)):
        if T[i] == 1:
            x.append(P[i, 0])
            y.append(P[i, 1])
            z.append(1)
    ax.scatter(x, y, marker='o', color='g')
    bx.scatter(x, y, z, marker='o', color='g')
    
    #ejes
    minimos = np.min(P, axis=0)
    maximos = np.max(P, axis=0)
    diferencias = maximos - minimos
    minimos = minimos - diferencias * 0.1
    maximos = maximos + diferencias * 0.1
    ax.axis([minimos[0], maximos[0], minimos[-1], maximos[1]])
    bx.axis([minimos[0], maximos[0], minimos[-1], maximos[1]])
    bx.set_zlim3d(0,1)
    
    #recta discriminante
    m = W[0,0] / W[0,1] * -1
    n = b / W[0,1] * -1
    x1 = minimos[0]
    y1 = x1 * m + n
    x2 = maximos[0]
    y2 = x2 * m + n
    ax.plot([x1, x2],[y1, y2], color='r')
    
    #sigmoide discriminante
    x = np.linspace(minimos[0], maximos[0], 30)
    y = np.linspace(minimos[-1], maximos[1], 30)

    X, Y = np.meshgrid(x, y)
    a1 = X * W[0,0]
    b1 = Y * W[0,1]
    c1 = a1 + b1 + b 
    Z = logsig(c1)
    bx.plot_wireframe(X, Y, Z)
    
    plt.draw()
    plt.pause(0.0001) 
    
def purelin(x):
    return x

def dpurelin(x):
    return np.ones_like(x)
    
def logsig(x):
    return 1 / (1 + np.exp(-x))

def dlogsig(x):
    return logsig(x) * (1 - logsig(x))

def tansig(x):
    return np.tanh(x)

def dtansig(x):
    return 1.0 - np.square(tansig(x))
    
def train_con_pesos(P, T, alfa, max_itera, cota_error, funcion, dibujar, W, b):
    (cant_patrones, cant_atrib) = P.shape    
    
    T2 = T.copy();
    T2 = np.floor((T2 + 1) / 2)

    ite = 0;
    error_prom = 1
    
    while (ite < max_itera) and (error_prom > cota_error):
        SumaError = 0
        for p in range(cant_patrones): 
           neta = b + W.dot(P[p, :])
           
           salida = eval(funcion + '(neta)')            
           errorPatron = T[p] - salida
           SumaError = SumaError + errorPatron ** 2
           
           derivada = eval('d' + funcion + '(neta)')
           
           grad_b = -2 * errorPatron * derivada;
           grad_W = -2 * errorPatron * derivada * P[p, :]

           b = b - alfa * grad_b;
           W = W - alfa * grad_W;         
        
        error_prom = SumaError / cant_patrones
        ite = ite + 1
        print(ite, error_prom)   
        
        if dibujar and (cant_atrib == 2):        
            plot(P, T2, W, b, 'Iteración: ' + str(ite) + ' - Error promedio: ' + str(error_prom))
        
    return (W, b, ite, error_prom)

def train(P, T, alfa, max_itera, cota_error, funcion, dibujar):
    (cant_patrones, cant_atrib) = P.shape    
    W = np.random.rand(1, cant_atrib)
    b = np.random.rand()
    return train_con_pesos(P, T, alfa, max_itera, cota_error, funcion, dibujar, W, b)
