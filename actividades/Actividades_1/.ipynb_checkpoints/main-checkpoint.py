from perceptron import train, plot
from ImagenPuntos import AbrirImagen

root_path = "D:\\Facultad\\Catedras\\UBA\\Teorías\\02 - Perceptrón\\Python\\"

file_path = root_path + 'Imagen 1.bmp'

X = AbrirImagen(file_path)
column_count = 3;

T = X[:, column_count - 1]
P = X[:, 0:(column_count - 1)]

alfa = 0.1
max_ite = 300

(W, b, ite) = train(P, T, alfa, max_ite, False)
print(ite)

plot(P, T, W, b)

