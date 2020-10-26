# -*- coding: utf-8 -*-
"""
Created on Oct 24 2020

@author: Waldo Hasperué
"""

import numpy as np
from matplotlib.pyplot import imread


def AbrirImagen(archivo):
    datos = np.array(imread(archivo))
    sh=np.shape(datos)
    datos2 = []
    for i in range(sh[0]):
        for j in range(sh[1]):
            color = sum(datos[i][j])
            ok=False
            if(color==255): #Negro
                color=0
                ok=True
            if(color==510): #Rojo
                color=1
                ok=True
            if(ok):
                datos2.append([j, 1000 - i, color])
                #datos2.append(color)
    return np.array(datos2)


if __name__ == '__main__':
    archivo = r'Imagen.bmp'
    print(AbrirImagen(archivo))



