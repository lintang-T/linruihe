import matplotlib.pyplot as plt


def save_comparison(results):

    fig, ax = plt.subplots(
        figsize=(8,4)
    )

    ax.axis("off")

    rows = []

    for method in results:

        row = [
            method,
            f"{results[method]['1']:.3f}",
            f"{results[method]['5']:.3f}",
            f"{results[method]['10']:.3f}"
        ]

        rows.append(row)

    table = ax.table(

        cellText=rows,

        colLabels=[
            "Method",
            "1-shot",
            "5-shot",
            "10-shot"
        ],

        loc="center"
    )

    plt.savefig(
        "outputs/comparison_table.png"
    )

    plt.close()