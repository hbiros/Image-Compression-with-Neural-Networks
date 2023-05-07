import matplotlib.pyplot as plt

def show_data(x, n=5, height=64, width=64, title=""):
    plt.figure(figsize=(10,3))
    for i in range(n):
        ax = plt.subplot(2,n,i+1)
        plt.imshow(x[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.suptitle(title, fontsize=16)
