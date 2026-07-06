import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def build_fewshot_dataset(x, y, shot):

    selected_x = []
    selected_y = []

    classes = torch.unique(y)

    for c in classes:

        idx = (y == c).nonzero(as_tuple=True)[0]

        idx = idx[torch.randperm(len(idx))]

        chosen = idx[:shot]

        selected_x.append(x[chosen])
        selected_y.append(y[chosen])

    selected_x = torch.cat(selected_x)
    selected_y = torch.cat(selected_y)

    return selected_x, selected_y


def train_finetune(
    x,
    y,
    shot=1,
    epochs=2
):

    print(f"\n===== FineTune {shot}-shot =====")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_x, train_y = build_fewshot_dataset(
        x,
        y,
        shot
    )

    print(
        f"Training samples: {len(train_x)}"
    )

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        len(torch.unique(y))
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(
        train_x,
        train_y
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    model.train()

    for epoch in range(epochs):

        total = 0
        correct = 0

        for bx, by in loader:

            bx = bx.to(device)
            by = by.to(device)

            optimizer.zero_grad()

            out = model(bx)

            loss = criterion(
                out,
                by
            )

            loss.backward()

            optimizer.step()

            pred = out.argmax(1)

            correct += (
                pred == by
            ).sum().item()

            total += by.size(0)

        acc = correct / total

        print(
            f"Epoch {epoch+1} "
            f"ACC={acc:.4f}"
        )

    return acc


if __name__ == "__main__":

    from dataset import CUBDataset

    dataset = CUBDataset(
        "CUB_200_2011"
    )

    x, y = dataset.get_data()

    for shot in [1, 5, 10]:

        acc = train_finetune(
            x,
            y,
            shot=shot,
            epochs=2
        )

        print(
            f"\nFinal {shot}-shot "
            f"ACC = {acc:.4f}"
        )