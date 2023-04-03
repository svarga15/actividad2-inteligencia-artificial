import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

celsius = np.array([-23, -8, 0, 23, 34, 65, 70], dtype=float)
fahrenheit = np.array([-9.4, 17, 32, 53.6, 93.2, 149, 158], dtype=float)

capas = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capas])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)


def entrenamiento():
    print("Inicio entrenamiento...")
    historialGradica = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
    print("Modelo se entreno!")
    verGraficaDeAprendizaje(historialGradica)


def ejecutarPrediccion():
    print("Provemos!")
    resultado = modelo.predict([9])
    print("Resultado es " + str(resultado) + " fahrenheit")


def verGraficaDeAprendizaje(historial):
    plt.xlabel("# base")
    plt.ylabel("Magnitud de pérdida")
    plt.plot(historial.history["loss"])
    plt.show()


if __name__ == '__main__':
    entrenamiento()
    ejecutarPrediccion()
