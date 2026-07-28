import matplotlib.pyplot as plt
from typing import Any


def main(
    records: dict[str, Any], 
    title: str, 
    score: str="mse",  
    figsize: tuple[int, int]=(7,3),
) -> None:
    trn_scores = records["trn"]
    val_scores = records["val"]
    best_epoch = records["best_epoch"]
    epoch = range(1,len(trn_scores)+1)

    plt.figure(
        figsize=figsize,
    )

    plt.plot(
        *(epoch, trn_scores),
        label="Train",
    )
    plt.plot(
        *(epoch, val_scores),
        label="Validation",
    )

    plt.axvline(
        x=best_epoch,
        color="red",
        linestyle="-",
        linewidth=2,
        label="Best Epoch",
    )

    plt.title(
        title, 
        fontsize=12, 
        fontweight="bold",
    )
    plt.xlabel(
        "Epoch", 
        fontsize=10,
    )
    plt.ylabel(
        score, 
        fontsize=10,
    )
    plt.legend(
        fontsize=9,
    )

    plt.grid(
        True, 
        linestyle="--", 
        alpha=0.5,
    )

    plt.tight_layout()
    plt.show()