import tensorflow as tf

# Definindo as dimensões n x m
n = 3  # número de linhas
m = 4  # número de colunas

# Criando um array n x m em TensorFlow
array_tf = tf.constant([[1, 2, 3, 4],
                        [5, 6, -20, 8],
                        [9, 10, -110, 90]])

# Definindo a função de mapeamento para retornar a constante 1
def map_to_constant_one(x):
    return tf.cond(x >= 20, lambda: 20, lambda: tf.cond(x <= -20, lambda: -20, lambda: x))

# Aplicando a função de mapeamento a cada elemento do tensor
mapped_array_tf = tf.map_fn(lambda x: tf.map_fn(map_to_constant_one, x), array_tf)

# Imprimindo o array original e o array mapeado
print("Array Original:")
print(array_tf.numpy())

print("\nArray Mapeado para Constante 1:")
print(mapped_array_tf.numpy())
