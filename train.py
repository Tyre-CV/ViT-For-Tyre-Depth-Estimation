import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm  # for progress bar

def parse_labels(labels, label_map=None):
    if not torch.is_tensor(labels):
        # labels might be a list/tuple of ints or strings
        if isinstance(labels, (list, tuple)) and labels and isinstance(labels[0], str):
            if label_map is None:
                raise ValueError("`label_map` must be provided to encode string labels to integers.")
            indices = [label_map[l] for l in labels]
            labels = torch.tensor(indices, dtype=torch.long)
        else:
            labels = torch.tensor(labels, dtype=torch.long)
    return labels


def train(model, train_loader, test_loader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), criterion=None, optimizer=None, num_epochs=10, lr=1e-4, save_path="./", label_map=None):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        # Train for one epoch
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit=" batch"):
            # imgs: (B, H, W), labels: (B,)
            imgs   = imgs.to(device)   # -> (B, H, W)
            labels = parse_labels(labels, label_map)
            labels = labels.to(device)  # -> (B,)

            optimizer.zero_grad()

            logits = model(imgs)            # -> (B, num_classes)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch} — train loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, correct = 0.0, 0
            for imgs, labels in tqdm(test_loader, desc="Validation", unit=" batch"):
                imgs = imgs.to(device)
                labels = parse_labels(labels, label_map)
                labels = labels.to(device)
                logits = model(imgs)
                val_loss += criterion(logits, labels).item() * imgs.size(0)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
            val_loss /= len(test_loader.dataset)
            val_acc  = correct / len(test_loader.dataset)
            print(f"         val loss: {val_loss:.4f}, val acc: {val_acc:.4f}")

    # Save model
    hyperparameters = f"lr_{lr}_epochs_{num_epochs}_batch_size_{train_loader.batch_size}_" + \
                      f"embed_dim_{model.embed_dim}_patch_size_{model.patch_size}_patch_dir_{model.patch_dir}"
    model_name = f"vit_{hyperparameters}.pth"
    torch.save(model.state_dict(), os.path.join(save_path, model_name))