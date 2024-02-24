import tensorflow as tf

tf.executing_eagerly()

x = tf.constant(3.0)

with tf.GradientTape() as g:
    g.watch(x)
    y = x*x
    dy_dx = g.gradient(y, x)

print(dy_dx)

x = tf.constant(5.0)

with tf.GradientTape() as g:
    g.watch(x)
    y = x*x*x
    dy_dx = g.gradient(y, x)

print(dy_dx)

x = tf.constant(3.0)

with tf.GradientTape() as g:
    g.watch(x)
    with tf.GradientTape() as gg:
        gg.watch(x)
        y = x*x
    dy_dx = gg.gradient(y, x)
d2y_dx2 = g.gradient(dy_dx, x)

print(dy_dx)

print(d2y_dx2)

x = tf.constant(3.0)

with tf.GradientTape(persistent=True) as g:
    g.watch(x)
    y = x*x
    z = y*y
dz_dx = g.gradient(z, x)
dy_dx = g.gradient(y, x)

print(dz_dx)

print(dy_dx)