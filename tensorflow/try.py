import tensorflow as tf

y_true = tf.Variable([[0., 2., 3.],
                      [0., 5., 6.]], )
y_pred = tf.Variable([[7., 8., 9.], 
                      [10., 11., 12.]],)


print(y_pred[..., 2:])
print(y_true[..., 0:1])

farklarinin_koku = tf.where(tf.equal(y_true[..., 0], tf.constant(0.0)), tf.constant([0.0]), tf.sqrt(tf.abs(y_pred[..., 2:] - y_true[..., 2:])))
#farklarinin_karesi = tf.where(tf.equal(y_true[..., 0], tf.constant(0.0)), tf.constant(0.0), tf.square(y_pred[..., 0:2] - y_true[..., 0:2]))

#print(farklarinin_karesi)
print(farklarinin_koku)

print(tf.reduce_mean(y_true, 1))
#print(tf.add(x1, x2))
#print(tf.add(farklarinin_koku, farklarinin_karesi))
#print(tf.reduce_sum(x1))