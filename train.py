from collections import defaultdict
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm  # for progress bar
from IPython.display import clear_output
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def save_loss(losses, save_dir="./losses"):
    os.makedirs(save_dir, exist_ok=True)  # Create save_dir if it doesn't exist

    # Append new losses to existing file or create a new one
    loss_path = os.path.join(save_dir, 'loss', "losses.json")
    if os.path.exists(loss_path):
        with open(loss_path, 'r') as f:
            existing_losses = json.load(f)
    else:
        existing_losses = defaultdict(list)

    new_losses = {**existing_losses}   
    for key, value in losses.items():
        new_losses[key].append(value)
            
    with open(loss_path, 'w') as f:
        json.dump(new_losses, f, indent=4)

def spooky():
    print("Booo 👻")

def save_attention(attn_maps, labels, epoch, save_dir="./attention"):
    os.makedirs(save_dir, exist_ok=True)
    epoch_path = os.path.join(save_dir, f"training_epoch_{epoch:03d}.json")
    if os.path.exists(epoch_path):
        with open(epoch_path, 'r') as f:
            attn_dict = json.load(f)
    else:
        attn_dict = defaultdict(list)
    
    for i, label in enumerate(labels.tolist()):
        attn = attn_maps[-1][i, 0, 1:].detach().cpu().tolist()
        key = str(label)
        attn_dict[key].append(attn)

    with open(epoch_path, 'w') as f:
        json.dump(attn_dict, f, indent=4)
        

def plot_attention_map(epoch, model, save_dir="./attention", output_dir="./attn_plots"):    
    epoch_path = os.path.join(save_dir, f"training_epoch_{epoch:03d}.json")
    with open(epoch_path, 'r') as f:
        attn_dict = json.load(f)
    os.makedirs(output_dir, exist_ok=True)

    h,w = model.patch_size 
    H,W = model.image_size 
    patch_grid = (H//h, W//w)

    rows, cols = patch_grid
    for lbl, info in attn_dict.items():
        attn_list = info["patch_attn"]
        avg = torch.tensor(attn_list).mean(dim=0).reshape(rows, cols).tolist()
        fig = go.Figure(go.Heatmap(z=avg, colorscale='Viridis'))
        fig.update_layout(title=f"Epoch {epoch} - Class {lbl}")
        out = os.path.join(output_dir, f"attn_epoch{epoch:03d}_class{lbl}.html")
        fig.write_html(out)
    
    
def plot_training_progress(save_path, number_of_epochs=None, losses=None):
    # Load JSON if needed
    if losses is None:
        with open(os.path.join(save_path, 'loss', "losses.json"), 'r') as f:
            losses = json.load(f)
    if number_of_epochs is None:
        number_of_epochs = len(losses['train_loss'])

    # Truncate/pad data
    train_loss = losses['train_loss'][:number_of_epochs]
    test_loss  = losses['test_loss'][:number_of_epochs]
    train_acc  = losses['train_acc'][:number_of_epochs]
    test_acc   = losses['test_acc'][:number_of_epochs]
    epochs     = list(range(1, number_of_epochs + 1))

    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add Loss traces (left axis) with custom colors
    fig.add_trace(
        go.Scatter(
            x=epochs, y=train_loss, mode='lines+markers', name='Train Loss',
            line=dict(color='rgb(0.0, 0.0, 127.5)')
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=epochs, y=test_loss, mode='lines+markers', name='Test Loss',
            line=dict(color='rgb(127.5,127.5,255)')
        ),
        secondary_y=False
    )

    # Add Accuracy traces (right axis) with custom colors
    fig.add_trace(
        go.Scatter(
            x=epochs, y=train_acc, mode='lines+markers', name='Train Acc',
            line=dict(color="#b22222")
        ),
        secondary_y=True
    )
    fig.add_trace(
        go.Scatter(
            x=epochs, y=test_acc, mode='lines+markers', name='Test Acc',
            line=dict(color='#ff6f3c')
        ),
        secondary_y=True
    )

    # Force x-axis from 1 to number_of_epochs with integer ticks
    fig.update_xaxes(
        title_text='Epoch',
        range=[1, number_of_epochs],
        tickmode='linear',
        dtick=1
    )

    # Label y-axes
    fig.update_yaxes(title_text='Loss', secondary_y=False)
    fig.update_yaxes(title_text='Accuracy', secondary_y=True)

    # Polish layout
    fig.update_layout(
        title='Training Progress',
        legend=dict(x=0.01, y=0.99, bordercolor="black", borderwidth=1),
        template='simple_white'
    )

    # Show in browser
    fig.show(renderer="browser")
    

def save_checkpoint(model, optimizer, epoch, batch_size, loss, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)  # Create dir if it doesn't exist
    checkpoint_path = os.path.join(checkpoint_dir, 'model', f"Epoch_{epoch}.pth")
    
    torch.save({
        'epoch': epoch,
        'batch_size': batch_size,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,

    }, checkpoint_path)
    

def train(model, train_loader, test_loader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), criterion=None, optimizer=None, num_epochs=10, lr=1e-4, save_path="./checkpoints", label_map=None):
    
    # Setup
    try:
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        if optimizer is None:
            optimizer = optim.AdamW(model.parameters(), lr=lr)

        model.to(device)

        # Clear all prior outputs (console)
        clear_output(wait=True)

        # Training loop
        for epoch in range(1, num_epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0

            # Train for one epoch
            for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit=" batch"):
                # imgs: (B, H, W), labels: (B,)
                imgs   = imgs.to(device) # -> (B, H, W)
                labels = parse_labels(labels, label_map)
                labels = labels.to(device) # -> (B,)

                optimizer.zero_grad()

                logits = model(imgs) # -> (B, num_classes)
                loss   = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()

                epoch_loss = running_loss / len(train_loader.dataset)
                train_acc = correct / len(train_loader.dataset)
                print(f"Epoch {epoch}:")
                print(f"\t — train loss: {epoch_loss:.4f}, train acc: {train_acc:.4f}")

            # Test loop
            model.eval()
            with torch.no_grad():
                test_loss, correct = 0.0, 0
                for imgs, labels in tqdm(test_loader, desc="Test", unit=" batch"):
                    imgs = imgs.to(device)
                    labels = parse_labels(labels, label_map)
                    labels = labels.to(device)
                    logits, attn_maps = model(imgs, return_attn = True)
                    test_loss += criterion(logits, labels).item() * imgs.size(0)
                    preds = logits.argmax(dim=-1)
                    correct += (preds == labels).sum().item()

                    save_attention(attn_maps, labels, epoch)

            test_loss /= len(test_loader.dataset)
            test_acc  = correct / len(test_loader.dataset)
            print(f"\t — test loss: {test_loss:.4f}, test acc: {test_acc:.4f}")
            
            # Epoch summary
            loss = {
                'train_loss': epoch_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc
            }
            # Save model epoch-wise
            save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, batch_size=imgs.size(0), loss=loss, checkpoint_dir=save_path)
            # Save losses
            save_loss(loss, save_dir="save_path")

            #Output
            # Clear console output for next epoch
            clear_output(wait=True)  
            # Show losses
            plot_training_progress(save_path, number_of_epochs=num_epochs)

        

    except Exception as e:
        # Free GPU memory in case of an error
        try:
            model.to('cpu')  # move model parameters to CPU
            del train_loader, test_loader, model, criterion, optimizer  # delete large tensors and optimizer
            torch.cuda.empty_cache()  # release cached memory
        except Exception:
            pass
        finally:
            raise e
    
    # Free GPU memory
    try:
        model.to('cpu')  # move model parameters to CPU
        del imgs, labels, logits, loss, optimizer  # delete large tensors and optimizer
        torch.cuda.empty_cache()  # release cached memory
    except Exception:
        pass

if __name__ == "__main__":
    pass