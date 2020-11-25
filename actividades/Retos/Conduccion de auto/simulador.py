# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:22:43 2019

@author: Waldo Hasperué
"""

import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
import numpy as np

_velocidad = 0.1         # Proporción de la distancia al nuevo punto
_velocidadGiro = 0.3      # Proporción de la orientación al nuevo punto

_umbralDeBorde = 20

_anguloRotacion = 0
_mover = False
_simular = False
_colision = False

_imagenCircuito = None
_imagenAuto = mpimg.imread('auto.jpg')
                    
_imgplot = None
_ventana = None

_posActual = ( 0,  0)
_posInicio = (0, 0)
_sensoresDis = []
_sensores = [-45, 0, 45]

_data = None

def _imgToPixel(event):
    return (event.xdata, event.ydata)
    
def _repaint():
    global _posActual, _imagenAuto, _imagenCircuito, _anguloRotacion, _mover, _sensoresDis, _simular, _pasosDados, _colision, _umbralDeBorde
    
    canvas = _imagenCircuito.copy()
    autos = _imagenAuto.shape
    
    if(_mover):
        sprite = _imagenAuto.copy()
        angulo = 360 - _anguloRotacion
        sprite = ndimage.rotate(sprite, angulo, reshape=False)

        # Dibujo el auto, detectando colisión
        for i in range(autos[0]):
            for j in range(autos[1]):
                ceros = 0
                for k in range(autos[2]):
                    # En la imagen están invertidos los x e y
                    x = _posActual[1] - autos[0] // 2
                    y = _posActual[0] - autos[1] // 2
                        
                    if(sprite[i][j][k] != 255) and (sprite[i][j][k] != 0):   
                        if canvas[x+i][y+j][k] < _umbralDeBorde:
                            ceros+= 1
                        
                        canvas[x+i][y+j][k] = sprite[i][j][k]
                if ceros > 1:
                    # Colisión contra el borde
                    _colision = True
        
        # sensores
        for s in _sensoresDis:
            for i in range(7):
                i = i - 3
                for j in range(7):
                    j = j - 3
                    canvas[s[1]+i][s[0]+j][0] = 255
                    canvas[s[1]+i][s[0]+j][1] = 0
                    canvas[s[1]+i][s[0]+j][2] = 0
        
        # PosActual
        for i in range(7):
            i = i - 3
            for j in range(7):
                j = j - 3
                canvas[_posActual[1]+i][_posActual[0]+j][0] = 0
                canvas[_posActual[1]+i][_posActual[0]+j][1] = 0
                canvas[_posActual[1]+i][_posActual[0]+j][2] = 255
                    
    plt.clf()
    plt.imshow(canvas)  
    plt.axis('off')
    text = ""
    if _colision:
        text = "CHOQUE !!!\n"
    if(not _mover):
        plt.title(text  +'Haga click en algún lugar del circuito para indicar el punto de partida')
    else:
        if not _simular:
            plt.title(text  +'Mueva el mouse, el auto seguirá el cursor. Haga click para finalizar la captura de datos')
        else:
            plt.title(text + str(_pasosDados) + ' pasos logrados.\n(Haga click para detener la simulación)')
    plt.draw()
    
def _simularRecorrido():
    global _posActual, _anguloRotacion, _ventana, _imagenCircuito, _mover, _sensoresDis, _fConductor, _pasosDados, _colision
    
    _mover = True
    _colision = False
    
    _anguloRotacion = 0
    _pasosDados = 0
    
    while _mover:
        if not _colision:
            _sensoresDis = _sensar()
        
            dataTest = []
            for i in range(len(_sensores)):
                dataTest.append(_distanciaEntre(_posActual, _sensoresDis[i]))

            respuesta = _fConductor(dataTest)
        
            _anguloRotacion = _anguloRotacion + respuesta[1]
            el_x, el_y = _puntoA(_posActual, respuesta[0], _anguloRotacion)
            _posActual = (el_x, el_y)
        
            _pasosDados+= 1

        _repaint()
        
        plt.pause(0.0001) 
        
def _on_close(event):
    global _mover
    _mover = False

def _onPressSim(event):
    global _posActual, _mover, _colision
    
    if _mover:
        _mover = False
        _colision = False
        _repaint()
    else:
        _posActual = _imgToPixel(event)    
        _posActual = (_posActual[0] + _imagenAuto.shape[1], _posActual[1])
        _simularRecorrido()        
    
def _onPress(event):
    global _posActual, _imgplot, _mover, _data, _ventana , _imagenAuto, _anguloRotacion, _simular , _sensores
        
    d = _imgplot.get_cursor_data(event)
    if(d == None):
        return

    _mover = not _mover    
    if(_mover):
        # Start record
        _posActual = _imgToPixel(event)    
        _posActual = (_posActual[0] + _imagenAuto.shape[1], _posActual[1])
        
        _anguloRotacion = 0
        _data = np.zeros((1, len(_sensores) + 2))
        _repaint()
    else:
        plt.close(_ventana)
        _ventana = None
        _data = _data[1:, :]
    
def _distanciaEntre(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
def _puntoA(punto, distancia, angulo):
    if (int(angulo) == 90) or (int(angulo) == -90):
        angulo = angulo + 0.5
    pendiente = math.tan(math.radians(angulo))
    el_x = math.sqrt( distancia**2 / (1 + pendiente**2 ))  + punto[0]
    el_y = pendiente * (el_x - punto[0]) + punto[1]
    
    if(angulo <= 90) or (angulo >= 270):
        (el_x, el_y) = (punto[0] - (el_x - punto[0]), punto[1] - (el_y - punto[1]))
    return (int(el_x), int(el_y))
        
def _desplazarA(destino):
    global _posActual, _velocidad, _anguloRotacion, _velocidadGiro, _distanciaRecorrida, _giroVolante
        
    dx = (destino[0] - _posActual[0]) * _velocidad
    dy = (destino[1] - _posActual[1]) * _velocidad
    posNueva = (_posActual[0] + int(dx), _posActual[1] + int(dy))

    # Cálculo de la variación del ángulo de rotación
    if(posNueva[0] - _posActual[0]) == 0:
        if(posNueva[1] < _posActual[1]):
            angulo = 90
        else:
            angulo = 270
    else:
        pendiente = (posNueva[1] - _posActual[1]) / (posNueva[0] - _posActual[0])
        angulo = math.degrees ( math.atan(pendiente)    )
        if(posNueva[0] > _posActual[0]):
            angulo = angulo + 180
        elif(angulo < 0):
            angulo = 270 + (90 + angulo)
            if(angulo >= 360):
                angulo = angulo - 360
    
    if(_anguloRotacion >= 0) and (_anguloRotacion <= 90) and (angulo < 360) and (angulo >= 270):
        angulo = angulo - 360
    elif(_anguloRotacion >= 270) and (_anguloRotacion <= 360) and (angulo < 90) and (angulo >= 0):
        angulo = angulo + 360
    _giroVolante = (angulo - _anguloRotacion) * _velocidadGiro
    _anguloRotacion = _anguloRotacion + _giroVolante
    if(_anguloRotacion > 360):
        _anguloRotacion = _anguloRotacion - 360
    if(_anguloRotacion < 0):
        _anguloRotacion = _anguloRotacion + 360
    
    #HAcer el desplazamiento
    _distanciaRecorrida = _distanciaEntre(_posActual, posNueva)
    
    el_x, el_y = _puntoA(_posActual, _distanciaRecorrida, _anguloRotacion)
    
    _posActual = (el_x, el_y)

def _sensarA(angulo, frente):
    global _imagenCircuito, _umbralDeBorde
    
    distancia = 0
    ok = True
    while ok:
        (x,y) = _puntoA(frente, distancia, angulo)
        if(x < 0) or (x > _imagenCircuito.shape[1]):
            # Salí de los límites
            ok = False
            (x,y) = frente
        elif(y < 0) or (y > _imagenCircuito.shape[0]):
            # Salí de los límites
            ok = False
            (x,y) = frente
        elif(_imagenCircuito[y][x][0] < _umbralDeBorde):
            # detección borde
            ok = False
        else:
            distancia = distancia + 1
    return (x,y)

def _sensar():
    global _anguloRotacion, _posActual, _sensores, _imagenAuto
           
    frenteSensores = _puntoA(_posActual, _imagenAuto.shape[0] * 0.35, _anguloRotacion)
    
    distancias = []
    for s in _sensores:
        angulo = _anguloRotacion + s
        sensor = _sensarA(angulo, frenteSensores)
        
        distancias.append(sensor)
    
    return distancias
    
def _onMove(event):
    global _posActual, _mover, _sensoresDis, _data, _distanciaRecorrida, _giroVolante, _imgplot, _simular, _sensores
    
    if not _mover:
        return
    if _simular:
        return
    
    d = _imgplot.get_cursor_data(event)
    if(d == None):
        return
    
    destino = _imgToPixel(event)
    
    _desplazarA(destino)
    
    _sensoresDis = _sensar()
    
    row = np.zeros((1, len(_sensores) + 2))
    for i in range(len(_sensores)):
        row[0,i] = _distanciaEntre(_posActual, _sensoresDis[i])
    row[0,len(_sensores)] = _distanciaRecorrida
    row[0,len(_sensores)+1] = _giroVolante
    
    _data = np.vstack((_data, row))

    _repaint()
    
def _onResize(event):
    _repaint()

# Funciones públicas

def obtenerDatosDeSensores():
    global _data
    
    if not (_data is None):
        return _data.copy()
    else:
        return None

def conduccionManual(sensores, circuito):
    global _imagenCircuito, _imgplot, _ventana, _mover, _simular , _sensores, _colision 
    
    _sensores = sensores
    _mover = False
    _simular = False
    _colision = False
    _imagenCircuito = mpimg.imread(circuito)

    if(_ventana != None):
        plt.close(_ventana)
    _ventana, ax = plt.subplots()
    _imgplot = plt.imshow(_imagenCircuito)   
    _repaint()

    img = _imgplot.get_figure()
    _ventana.canvas.mpl_connect('button_press_event', _onPress)
    img.canvas.mpl_connect('motion_notify_event', _onMove)
    _ventana.canvas.mpl_connect('resize_event', _onResize)
    _ventana.canvas.mpl_connect('close_event', _on_close)

    
def conduccionAutonoma(sensores, circuito, fConductor):
    global _fConductor, _simular , _ventana, _imgplot, _mover, _imagenCircuito, _sensores, _colision
    
    _sensores = sensores
    _fConductor = fConductor
    _simular = True
    _colision = False
    _mover = False
    
    _imagenCircuito = mpimg.imread(circuito)
    if(_ventana != None):
        plt.close(_ventana)    
    _ventana, ax = plt.subplots()    
    _imgplot = plt.imshow(_imagenCircuito)   
    _repaint()
    
    _ventana.canvas.mpl_connect('button_press_event', _onPressSim)
    _ventana.canvas.mpl_connect('resize_event', _onResize)
    _ventana.canvas.mpl_connect('close_event', _on_close)
    