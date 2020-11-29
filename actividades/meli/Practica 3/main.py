# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:35:35 2017

@author: Waldo Hasperué
"""

import bpn
from ImagenPuntos import AbrirImagen
import numpy as np

root_path = "D:\\Facultad\\Catedras\\UBA\\Teorías\\04 - BPN\\Python\\"

file_path = root_path + 'Imagen 1.bmp'

X = AbrirImagen(file_path)
column_count = 3;

T = X[:, column_count - 1]
P = X[:, 0:(column_count - 1)]

ocultas = 1
alfa = 0.01
momento = 0
max_ite = 60
cota_error = 0.0001

fun_oculta = 'tansig'
fun_salida = 'logsig'

T_matriz = np.concatenate(([T==0], [T==1]), axis=0).astype(int).T
T_original = np.array(X[:, column_count - 1])

(w1, b1, w2, b2, ite, error) = bpn.train(P, T_matriz, T_original, ocultas, alfa, momento, fun_oculta, fun_salida, max_ite, cota_error, True)
