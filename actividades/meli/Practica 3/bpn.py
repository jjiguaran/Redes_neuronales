# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:02:48 2017

@author: Waldo Hasperué

Función train
-------------
Parámetros:
       P: es una matriz con los datos de los patrones con los cuales
           entrenar la red neuronal. Los ejemplos deben estar en filas.
       T: es una matriz con la salida esperada para cada ejemplo. Esta matriz 
           debe tener tantas columnas como neuronas de salida tenga la red
       T2: clases con su valor original (0 .. n-1) (Solo es utilizado para graficar)
       ocultas: la cantidad de neuronas ocultas que tendrá la red    
       alfa: velocidad de aprendizaje
       momento: término de momento
       fun_oculta: función de activación en las neuronas de la capa oculta
       fun_salida: función de activación en las neuronas de la capa de salida
       MAX_ITERA: la cantidad máxima de iteraciones en las cuales se va a
           ejecutar el algoritmo
       cota_error: diferencia de error (entre una iteración y la anterior) mínima aceptada para finalizar con el algoritmo
       dibujar: si vale True (y los datos son en dos dimensiones) dibuja los
           ejemplos y las rectas discriminantes.

Devuelve:
       w_O: la matriz de pesos de las neuronas de la capa oculta
       b_O: vector de bias de las neuronas de la capa oculta
       w_S: la matriz de pesos de las neuronas de la capa de salida
       b_S: vector de bias de las neuronas de la capa de salida
       ite: número de iteraciones ejecutadas durante el algoritmo
       error_prom: errorPromedio finalizado el algoritmo

Ejemplo de uso:
       (w_O, b_O, w_S, b_S, ite, error_prom) = train(P, T, T2, 10, 0.25, 1.2, 'logsig', 'tansig', 25000, 0.001, True);
       
*******************************************************************************

Función plot
-------------       
Parámetros:
       P: es una matriz con los datos de los patrones con los cuales
           entrenar el multiperceptrón. Los ejemplos deben estar en filas.
       T: es un vector con la clase esperada para cada ejemplo. Los
           valores de las clases deben ser 0 (cero) y 1 (uno)
       W: la matriz de pesos W del percpetrón entrenado
       b: valor del bias (W0) del perceptrón entrenado
       title: el título que aparecerá en la gráfica

Ejemplo de uso:
       plot(P, T, W, b, 'Entrenamiento final del multiperceptrón');
       
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(0)

marcadores = {0:('+','b'), 1:('o','g'), 2:('x', 'y'), 3:('*', 'm'), 4:('.', 'r'), 5:('+', 'k')}

def plot(P, T, W, b, title = '', **extras):
    plt.clf()
        
    (neuronas, cant_patrones) = W.shape
    if(neuronas <= 2):
        gs = gridspec.GridSpec(1, 2)
        ax = plt.subplot(gs[0, 0])    
        bx = plt.subplot(gs[0, 1])
    else:
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])    
    
    #Ejemplos
    for class_value in np.unique(T):
        x = []
        y = []
        for i in range(len(T)):
            if T[i] == class_value:
                x.append(P[i, 0])
                y.append(P[i, 1])
        ax.scatter(x, y, marker=marcadores[class_value][0], color=marcadores[class_value][1])
    
    #ejes
    minimos = np.min(P, axis=0)
    maximos = np.max(P, axis=0)
    diferencias = maximos - minimos
    minimos = minimos - diferencias * 0.1
    maximos = maximos + diferencias * 0.1
    ax.axis([minimos[0], maximos[0], minimos[-1], maximos[1]])
    
    #rectas discriminantes
    x1 = minimos[0]
    x2 = maximos[0]
    for neu in range(neuronas):
        m = W[neu,0] / W[neu,1] * -1
        n = b[neu] / W[neu,1] * -1
        y1 = x1 * m + n
        y2 = x2 * m + n
        ax.plot([x1, x2],[y1, y2], color='r')
        
        
    if(neuronas <=2):
        # espacio de la capa oculta
        fun_oculta = extras["fun_oculta"]
        for class_value in np.unique(T):
            x = []
            y = []
            for i in range(len(T)):
                if T[i] == class_value:
                    neta_oculta = W.dot(P[i,:][np.newaxis].T) + b
                    salida_oculta = eval(fun_oculta + '(neta_oculta)')
                    if(len(salida_oculta) == 1):
                        x.append(0 + np.random.random() / 4 - 0.125)
                        y.append(salida_oculta)
                    else:
                        x.append(salida_oculta[0,0])
                        y.append(salida_oculta[1,0])
            bx.scatter(x, y, marker=marcadores[class_value][0], color=marcadores[class_value][1])
    
        #rectas discriminantes
        if (fun_oculta == "logsig"):
            x1 = -0.1
            x2 = 1.1
        else:
            x1 = -1.1
            x2 = 1.1
        bx.axis([x1, x2, x1, x2])
        w2 = extras["w2"]
        b2 = extras["b2"]
        neuronas2, dim = w2.shape
        for neu in range(neuronas2):
            if(dim == 2):
                m = w2[neu,0] / w2[neu,1] * -1
                n = b2[neu] / w2[neu,1] * -1
            else:
                m = w2[neu][0]
                n = b2[neu][0]
                
            y1 = x1 * m + n
            y2 = x2 * m + n
            bx.plot([x1, x2],[y1, y2], color='r')
        
    plt.suptitle(title)
    
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

def softmax(x):
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)

def dsoftmax(x):
    s = softmax(x)
    n = s.shape[0]
    jacobian_m = np.zeros((n,n))
    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i] * (1-s[i])
            else: 
                jacobian_m[i][j] = -s[i]*s[j]
    return jacobian_m
    
def train_con_pesos(P, T, T2, ocultas, alfa, momento, fun_oculta, fun_salida, max_itera, cota_error, dibujar, w_O, b_O, w_S, b_S):
    (cant_patrones, cant_atrib) = P.shape
    
    momento_w_S = np.zeros(w_S.shape)
    momento_b_S = np.zeros(b_S.shape)
    momento_w_O = np.zeros(w_O.shape)
    momento_b_O = np.zeros(b_O.shape)

    ite = 0;
    error_prom = cota_error + 1
    ultimoError = cota_error +1
    anteultimoError = cota_error +2
    
    while (ite < max_itera) and (error_prom > cota_error):
        suma_error = 0
        for p in range(cant_patrones): 
            neta_oculta = w_O.dot(P[p,:][np.newaxis].T) + b_O
            salida_oculta = eval(fun_oculta + '(neta_oculta)')
            neta_salida = w_S.dot(salida_oculta) + b_S
            salida_salida = eval(fun_salida + '(neta_salida)')
           
            error_ejemplo = T[p,:] - salida_salida.T[0]
            suma_error = suma_error + np.sum(error_ejemplo**2)

            delta_salida = error_ejemplo[np.newaxis].T * eval('d' + fun_salida + '(neta_salida)')
            delta_oculta = eval('d' + fun_oculta + '(neta_oculta)') * w_S.T.dot(delta_salida)
            
            w_S = w_S + alfa * delta_salida * salida_oculta.T + momento * momento_w_S
            b_S = b_S + alfa * delta_salida + momento * momento_b_S
             
            w_O = w_O + alfa * delta_oculta * P[p,:] + momento * momento_w_O
            b_O = b_O + alfa * delta_oculta + momento * momento_b_O
           
            momento_w_S = alfa * delta_salida * salida_oculta.T + momento * momento_w_S
            momento_b_S = alfa * delta_salida + momento * momento_b_S            
            
            momento_w_O = alfa * delta_oculta * P[p,:].T + momento * momento_w_O
            momento_b_O = alfa * delta_oculta + momento * momento_b_O
            
        error_prom = suma_error / cant_patrones
        
        anteultimoError = ultimoError
        ultimoError = error_prom
        
        ite = ite + 1
        print(ite, error_prom, abs(ultimoError - anteultimoError))   
        
        if dibujar and (cant_atrib == 2):        
            plot(P, T2, w_O, b_O, 'Iteración: ' + str(ite) + ' - Error promedio: ' + str(error_prom), fun_oculta = fun_oculta, w2 = w_S, b2 = b_S)
        
    return (w_O, b_O, w_S, b_S, ite, error_prom)

def train(P, T, T2, ocultas, alfa, momento, fun_oculta, fun_salida, max_itera, cota_error, dibujar):
    (cant_patrones, cant_atrib) = P.shape
    (cant_patrones, cant_salidas) = T.shape
    
    w_O = np.random.rand(ocultas, cant_atrib) - 0.5
    b_O = np.random.rand(ocultas,1) - 0.5
    w_S = np.random.rand(cant_salidas, ocultas) - 0.5
    b_S = np.random.rand(cant_salidas,1) - 0.5
        
    return train_con_pesos(P, T, T2, ocultas, alfa, momento, fun_oculta, fun_salida, max_itera, cota_error, dibujar, w_O, b_O, w_S, b_S)
    
