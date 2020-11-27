import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def equals_vector(x, y):
    return np.all(x==y)

def verificar_igualdad(x,y):
    iguales=equals_vector(x, y)
    if iguales:
        print("Los vectores x e y son iguales:")
    else:
        print("Los vectores x e y son distintos:")

    print("x: ", x)
    print("y: ", y)


def plot_regresion_lineal_univariada(w,b,x,y,title=""):
    # genero una ventana de dibujo con una sola zona de dibujo (1,1)
    f,ax_data=plt.subplots(1,1)
    # dibujo el conjunto de datos como pares x,y y color azul
    ax_data.scatter(x, y, color="blue")
    # establezco el titulo principal
    f.suptitle(f"{title}")
    # Dibujo la recta dada por los parámetros del modelo (w,b)
    x_pad = 10
    min_x, max_x = x.min() - x_pad, x.max() + x_pad
    ax_data.plot([min_x, max_x], [min_x * w + b, max_x * w + b], color="red",label=f"w={w:.5f}, b={b:.5f}")
    # agrego una leyenda con la etiqueta del parametro `label`
    ax_data.legend()
    # Establezco las etiquetas de los ejes x e y
    ax_data.set_xlabel("x (Horas estudiadas)")
    ax_data.set_ylabel("y (Nota)")



def plot_regresion_lineal(w,b,x,y,title=""):
    # genero una ventana de dibujo con una sola zona de dibujo (1,1)
    # que permita graficos en 3D
    figure = plt.figure(figsize=(10, 10), dpi=100)
    ax_data = figure.add_subplot(1, 1, 1, projection='3d')

    #dibujo el dataset en 3D (x1,x2,y)
    x1=x[:,0]
    x2=x[:,1]
    ax_data.scatter(x1,x2, y, color="blue")
    figure.suptitle(title)

    # Dibujo el plano dado por los parametros del modelo (w,b)
    # Este codigo probablemente no sea facil de entender
    # si no tenes experiencia con calculos en 3D
    detail = 0.05
    # genero coordenadas x,y de a pares, las llamo xx e yy
    xr = np.arange(x.min(), x.max(), detail)
    yr = np.arange(y.min(), 10, detail)
    xx, yy = np.meshgrid(xr, yr)
    # calculo las coordenadas z en base a xx, yy, y el modelo (w,b)
    zz = xx * w[0] + yy * w[1] + b
    # dibujo la superficie dada por los puntos (xx,yy,zz)
    surf = ax_data.plot_surface(xx, yy, zz, cmap='Reds', alpha=0.5, linewidth=0, antialiased=True)

    # Establezco las etiquetas de los ejes
    ax_data.set_xlabel("x1 (Horas estudiadas)")
    ax_data.set_ylabel("x2 (Promedio)")
    ax_data.set_zlabel("y (Nota)")
    # Establezco el titulo del grafico
    ax_data.set_title("(Horas estudiadas x Promedio) vs Nota")


# imprime los puntos para un dataset bidimensional junto con la frontera de decisión del modelo
def plot_frontera_de_decision_2D(modelo, x, y, x_test=0, y_test=0, title="",detail=0.05):

    assert x.shape[1]==2,f"x debe tener solo dos variables de entrada (tiene {x.shape[1]})"
    # nueva figura
    plt.figure()
    # gráfico con la predicción aprendida
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, detail),
                         np.arange(y_min, y_max, detail))

    Z = np.c_[xx.ravel(), yy.ravel()]

    Z = modelo.predict(Z)
    Z = Z.argmax(axis=1)  # para Keras
    titulo = f"{title}: regiones de cada clase"
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)  # ,  cmap='RdBu')
#    plt.colorbar()
    plt.title(titulo)

    # puntos con las clases
    plt.scatter(x[:, 0], x[:, 1], marker="o", c=y, label= 'Training')
    if (isinstance(x_test, int)==False):
        plt.scatter(x_test[:,0], x_test[:,1], marker="+",c=y_test, alpha=0.8, s=80, label= 'Testing') 
        plt.legend()

# graficar curvas de entrenamiento
def plot_training_curves(history, acc=True):
    # summarize history for accuracy
    if (acc):
        plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show(), plt.grid()
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show(), plt.grid()

# Crea y Grafica una matriz de confusión
# PARAM:
#       real_target = vector con valores esperados
#       pred_target = vector con valores calculados por un modelo
#       classes = lista de strings con los nombres de las clases.
def plot_confusion_matrix(real_target, pred_target, classes=[],  normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    from sklearn.metrics import confusion_matrix
    import itertools
    if (len(classes)==0):
        classes= [str(i) for i in range(int(max(real_target)+1))]  # nombres de clases consecutivos
    cm= confusion_matrix(real_target, pred_target)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

#    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# dividir dataset en training y testing de forma aleatoria    
def dividir_train_test(X, Y, test_size= 0.2):
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= test_size)
    return X_train, X_test, Y_train, Y_test

# Grafica la curva ROC y la curva precision-recall para el modelo y los datos pasados como argumentos 
def plot_ROC_curve(modelo, x, y):
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
    # probabilidades para los datos
    y_score = modelo.predict_proba(x)[:,1] # se queda con la clase 1
    # Create true and false positive rates
    false_positive_rate, true_positive_rate, threshold = roc_curve(y, y_score)
    # Precision-recall curve
    precision, recall, _ = precision_recall_curve(y, y_score)
    
    # ROC
    plt.figure()
    plt.title('ROC. Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, label='ROC curve (area = %0.2f)' % roc_auc_score(y, y_score))
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.legend(loc="lower right")
    plt.ylabel('True Positive Rate (Recall)')
    plt.xlabel('False Positive Rate (1- Especificidad)')
    plt.show()
    
    # precision-recall curve
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve')   
    
def print_classification_report(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, f1_score,  recall_score, precision_score, accuracy_score

    n_clases= (max(y_true)+1)
    # probs to class number
    y_pred = np.argmax(y_pred,axis = 1) 
    print("   Accuracy: %.2f    soporte: %d" % (accuracy_score(y_true, y_pred), y_true.shape[0]))
    if (n_clases==2):
        print("  Precision: %.2f" % precision_score(y_true, y_pred) )
        print("     Recall: %.2f" % recall_score(y_true, y_pred ))
        print("  f-measure: %.2f" % f1_score(y_true, y_pred))
        
        
        