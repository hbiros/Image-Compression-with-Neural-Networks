import matplotlib.pyplot as plt

def show_data(x, y, n=5, height=64, width=64, title_1="Original", title_2="Reconstructed"):
    fig, (ax1, ax2)= plt.subplots(2,n, figsize=(10,4))
    for i in range(n):
        ax1[i].imshow(x[i], extent=[0, 1, 0, 1])
        ax2[i].imshow(y[i], extent=[0, 1, 0, 1])
        ax1[i].get_xaxis().set_visible(False)
        ax1[i].get_yaxis().set_visible(False)
        ax2[i].get_xaxis().set_visible(False)
        ax2[i].get_yaxis().set_visible(False)
    midpoint = n // 2
    ax1[midpoint].set_title(title_1, position=(0.5, 0.5), fontsize=14)
    ax2[midpoint].set_title(title_2, position=(0.5, 0.5), fontsize=14)
    plt.show()