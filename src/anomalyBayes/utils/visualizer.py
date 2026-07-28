import matplotlib.pyplot as plt
from typing import Any


def draw_ax(ax, trn_data, val_data, score, best_epoch):
    epoch = range(1,len(trn_data)+1)

    ax.plot(
        *(epoch, trn_data),
        label="Train",
    )
    ax.plot(
        *(epoch, val_data),
        label="Validation",
    )

    ax.axvline(
        x=best_epoch,
        color="red",
        linestyle="-",
        linewidth=2,
        label="Best Epoch",
    )

    ax.set_title(
        f"Train vs. Validation (Score: {score})", 
        fontsize=12, 
        fontweight="bold",
    )
    ax.set_xlabel(
        "Epoch", 
        fontsize=10,
    )
    ax.set_ylabel(
        score, 
        fontsize=10,
    )
    ax.legend(
        fontsize=9,
    )

    ax.grid(
        True, 
        linestyle="--", 
        alpha=0.5,
    )


def main(
    records: dict[str, Any], 
    suptitle: str, 
    score: str="mse",  
    figsize: tuple[int, int]=(7,3),
) -> None:
    trn_nll = records["trn"]["nll"]
    trn_kld = records["trn"]["kld"]
    val_nll = records["val"]["nll"]
    val_kld = records["val"]["kld"]
    best_epoch = records["best_epoch"]

    NROWS = 2
    NCOLS = 1
    WEIGHTS = figsize[0]
    HEIGHTS = figsize[1]

    fig, axes = plt.subplots(
        nrows=NROWS, 
        ncols=NCOLS, 
        figsize=(WEIGHTS*NCOLS, HEIGHTS*NROWS), 
        sharex=True, 
        sharey=False,
    )

    nll = (trn_nll, val_nll, score, best_epoch)
    kld = (trn_kld, val_kld, "KLD", best_epoch)

    for ax, args in zip(axes, (nll, kld)):
        draw_ax(ax, *args)

    plt.suptitle(
        t=suptitle,
        fontsize=14,
        fontweight="bold",
    )
    
    plt.tight_layout()
    plt.show()
