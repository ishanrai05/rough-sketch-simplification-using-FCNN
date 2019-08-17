import matplotlib.pyplot as plt
from PIL import Image

def show_samples(images):
    fig = plt.figure(figsize=(20, 5))
    columns, rows = 8, 2
    ax = []
    k = 0
    for i in range(columns*rows):
        img = Image.open(images[k])
        k += 1
        # create subplot and append to ax
        ax.append( fig.add_subplot(rows, columns, i+1))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap="gray")
    plt.tight_layout(True)
    plt.show()  # finally, render the plot