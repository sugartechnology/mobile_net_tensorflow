import matplotlib.pyplot as plt
import tensorflow as tf


def display(display_list, title, results):

    for i in range(len(display_list)):
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))

        res = results[i]
        print('{} {}'.format(title[i], res))
        cx = res[1] * 224
        cy = (1 - res[2]) * 224
        plt.scatter(cx, cy)
        plt.scatter(cx + res[3] * 224, cy - res[4] * 224)
        plt.scatter(cx + res[5] * 224, cy - res[6] * 224)
        plt.scatter(cx + res[7] * 224, cy - res[8] * 224)
        plt.scatter(cx + res[9] * 224, cy - res[10] * 224)
        plt.axis('off')
        plt.show()

    
