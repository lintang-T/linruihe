import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def save_curve(data_dict):

    for k, v in data_dict.items():
        plt.plot(v, label=k)

    plt.legend()
    plt.title("Few-shot Comparison")
    plt.savefig("outputs/acc_curve.png")
    plt.close()


def save_tsne(feat, label):

    z = TSNE(2).fit_transform(feat)

    plt.scatter(z[:,0], z[:,1], c=label, s=10)
    plt.title("t-SNE")
    plt.savefig("outputs/tsne.png")