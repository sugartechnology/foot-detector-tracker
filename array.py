import tensorflow as tf
import numpy as np

# a = np.arange(40.0).reshape(2, 5, 2, 2)
# a = np.array(np.split(a, 2, axis=3))
# print(a.shape)

if __name__ == "__main__":
    c1 = tf.constant(
        np.array(np.random.randint(2, size=32).astype(np.float32)).reshape(2, 2, 2, 4)
    )
    c2 = tf.constant(
        np.array(np.random.randint(2, size=32).astype(np.float32)).reshape(2, 2, 2, 4)
    )

    i = c1

    u = c2

    print("i")
    print(i)
    i = tf.reduce_sum(i, axis=(1, 2))
    print("i.reduce_sum")
    print(i)

    print("u")
    print(u)
    u = tf.reduce_sum(u, axis=(1, 2))
    print("u.reduce_sum")
    print(u)

    iou = (i + 1e-6) / (u + 1e-6)
    print("iou")
    print(iou)
    print("======")
    iou0 = tf.reduce_sum(iou, axis=0)
    print(iou0)
    print("iou1")
    iou1 = tf.reduce_mean(iou)
    print(iou1)

    print("iou0")
    iou0 = tf.reduce_mean(iou0)
    print(iou0)
