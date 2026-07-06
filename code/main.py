from dataset import CUBDataset
from model import Encoder
from episode import sample_episode
from train_eval import run_episode
from methods import apply_method
from viz import save_curve, save_tsne
from report import generate_report
from finetune import train_finetune
from comparison import save_comparison
from confusion import save_confusion_matrix

import torch


METHODS = [
    "baseline",
    "mixup",
    "cutmix",
    "finetune"
]

SHOTS = [1, 5, 10]


def main():

    print("Loading CUB dataset...")

    dataset = CUBDataset("CUB_200_2011")
    x, y = dataset.get_data()

    print(f"Loaded {len(x)} images")

    model = Encoder()

    optim = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    results = {}
    curves = {}

    for method in METHODS:

        print(f"\n===== {method} =====")

        # -------------------------
        # Fine-tuning
        # -------------------------
        if method == "finetune":

            acc1 = train_finetune(
                x,
                y,
                shot=1
            )

            acc5 = train_finetune(
                x,
                y,
                shot=5
            )

            acc10 = train_finetune(
                x,
                y,
                shot=10
            )

            results["finetune"] = {
                "1": acc1,
                "5": acc5,
                "10": acc10
            }

            continue

        # -------------------------
        # ProtoNet / Mixup / CutMix
        # -------------------------
        curves[method] = []

        for shot in SHOTS:

            print(f"Running {shot}-shot...")

            acc_list = []

            for i in range(150):

                sx, sy, qx, qy = sample_episode(
                    x,
                    y,
                    k_shot=shot
                )

                sx, sy, _, _ = apply_method(
                    method,
                    sx,
                    sy
                )

                episode = (
                    sx,
                    sy,
                    qx,
                    qy
                )

                loss, acc, feat, label = run_episode(
                    model,
                    episode
                )

                optim.zero_grad()

                loss.backward()

                optim.step()

                acc_list.append(
                    acc.item()
                )

            avg_acc = sum(acc_list) / len(acc_list)

            results.setdefault(
                method,
                {}
            )[str(shot)] = avg_acc

            curves[
                method + f"_shot{shot}"
            ] = acc_list

            print(
                f"{method} {shot}-shot "
                f"ACC={avg_acc:.4f}"
            )

    print("\nGenerating figures...")

    save_curve(curves)

    save_tsne(
        feat.detach().numpy(),
        label.numpy()
    )

    save_comparison(results)

    save_confusion_matrix()

    print("Generating report...")

    generate_report(results)

    print("\nFinished!")
    print(results)


if __name__ == "__main__":
    main()