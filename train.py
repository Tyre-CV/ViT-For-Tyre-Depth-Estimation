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
    os.makedirs(os.path.join(save_dir, 'loss'), exist_ok=True)  # Create save_dir if it doesn't exist

    # Append new losses to existing file or create a new one
    loss_path = os.path.join(save_dir, 'loss', "losses.json")
    existing_losses = defaultdict(list)

    if os.path.exists(loss_path):
        with open(loss_path, 'r') as f:
            existing_losses = json.load(f)
        for key, value in existing_losses.items():
            existing_losses[key] = value

     
    for key, value in losses.items():
        existing_losses[key].append(value)
            
    with open(loss_path, 'w') as f:
        json.dump(existing_losses, f, indent=4)

def spooky():
    print("Booo 👻")

def save_attention_weights(attn_maps, labels, epoch, save_dir="./attention"):
    os.makedirs(save_dir, exist_ok=True)
    epoch_path = os.path.join(save_dir, f"training_epoch_{epoch:03d}.json")

    attn_dict = defaultdict(list)

    if os.path.exists(epoch_path):
        with open(epoch_path, 'r') as f:
            loaded_dict = json.load(f)
        for key, value in loaded_dict.items():
            attn_dict[key] = value
    
    for i, label in enumerate(labels.tolist()):
        attn = attn_maps[-1][i, 0, 1:].detach().cpu().tolist()
        key = str(label)
        attn_dict[key].append(attn)

    with open(epoch_path, 'w') as f:
        json.dump(attn_dict, f, indent=4)
        

def plot_attention_map(epoch, model, save_dir="./attention", output_dir="./attn_plots"):    
    epoch_path = os.path.join(save_dir, f"training_epoch_{epoch:03d}.json")
    if not os.path.exists(epoch_path):
        raise FileNotFoundError(f"Attention map file for epoch {epoch} not found at {epoch_path}.")
    
    with open(epoch_path, 'r') as f:
        attn_dict = json.load(f)
    if not os.path.exists(output_dir):
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
    
    
def plot_training_progress(save_path, number_of_epochs=None, losses=None, renderer=None):
    # Load JSON if needed
    if losses is None:
        # Check if the losses file exists
        if not os.path.exists(os.path.join(save_path, 'loss', "losses.json")):
            raise FileNotFoundError(f"Losses file not found at {os.path.join(save_path, 'loss', 'losses.json')}.")
        with open(os.path.join(save_path, 'loss', "losses.json"), 'r') as f:
            losses = json.load(f)
    if number_of_epochs is None:
        number_of_epochs = len(losses['train_loss'])

    # Truncate/pad data
    train_loss = losses['train_loss'][:number_of_epochs]
    test_loss  = losses['test_loss_all'][:number_of_epochs]
    train_acc  = losses['train_acc'][:number_of_epochs]
    test_acc   = losses['test_acc_all'][:number_of_epochs]
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
    if renderer is None:
        fig.show()
    else:
        fig.show(renderer=renderer)
    

def save_checkpoint(model, optimizer, epoch, batch_size, loss, checkpoint_dir="checkpoints", history_length=6):
    os.makedirs(os.path.join(checkpoint_dir, 'models'), exist_ok=True)  # Create dir if it doesn't exist

    # Check how many models are already saved
    existing_models = [f for f in os.listdir(os.path.join(checkpoint_dir, 'models')) if f.endswith('.pth')]
    # If we have more than history_length models, remove the oldest one
    if len(existing_models) >= history_length:
        existing_models.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        oldest_model = existing_models[0]
        os.remove(os.path.join(checkpoint_dir, 'models', oldest_model))

    checkpoint_path = os.path.join(checkpoint_dir, 'models', f"Epoch_{epoch}.pth")
    
    torch.save({
        'epoch': epoch,
        'batch_size': batch_size,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,

    }, checkpoint_path)
    

def train(model, train_loader, test_loader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), criterion=None, optimizer=None, num_epochs=10, lr=1e-4, save_path="./checkpoints", label_map=None, save_attention=False, history_length=6):
    
    # Setup
    try:
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        if optimizer is None:
            optimizer = optim.AdamW(model.parameters(), lr=lr)

        # Ensure save_path exists
        os.makedirs(save_path, exist_ok=True)

        # multi-GPU support
        model = nn.DataParallel(model)

        model.to(device)

        reverse_label_map = {v: k for k, v in label_map.items()} if label_map else None

        # Clear all prior outputs (console)
        clear_output(wait=True)
        print(f"Using {torch.cuda.device_count()} GPUs for training.")

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

                logits, attn_maps = model(imgs) # -> (B, num_classes)
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
                test_loss_all, correct_all = 0.0, 0
                # Label-Wise loss and accuracy
                label_wise_loss = {lbl: 0.0 for lbl in label_map.values()} if label_map else None
                label_wise_correct = {lbl: 0 for lbl in label_map.values()} if label_map else None                 
                for imgs, labels in tqdm(test_loader, desc="Test", unit=" batch"):
                    imgs = imgs.to(device)
                    labels = parse_labels(labels, label_map)
                    labels = labels.to(device)
                    logits, attn_maps = model(imgs, return_attn = True)
                    test_loss_all += criterion(logits, labels).item() * imgs.size(0)
                    preds = logits.argmax(dim=-1)
                    correct_all += (preds == labels).sum().item()
                    if label_wise_loss is not None:
                        for bucket_label in label_map.values():
                            mask = (labels == bucket_label)
                            if mask.any():
                                label_wise_loss[bucket_label] += criterion(logits[mask], labels[mask]).item() * mask.sum().item()
                                label_wise_correct[bucket_label] += (preds[mask] == labels[mask]).sum().item()

                    if save_attention and epoch == num_epochs:
                        # Save attention maps for the last layer
                        save_attention_weights(attn_maps, labels, epoch)

            test_loss_all /= len(test_loader.dataset)
            test_acc_all  = correct_all / len(test_loader.dataset) if len(test_loader.dataset) > 0 else 0
            print(f"\t — test loss: {test_loss_all:.4f}, test acc: {test_acc_all:.4f}")

            # Print label-wise loss and accuracy
            if label_wise_loss is not None:
                for bucket_label, loss in label_wise_loss.items():
                    acc = label_wise_correct[bucket_label] / (labels == bucket_label).sum().item() if (labels == bucket_label).sum().item() > 0 else 0
                    print(f"\t\t —— {reverse_label_map[bucket_label] if reverse_label_map else bucket_label}: loss = {loss:.4f}, acc = {acc:.4f}")
            
            # Epoch summary
            loss = {
                'train_loss': epoch_loss,
                'train_acc': train_acc,
                'test_loss_all': test_loss_all,
                'test_acc_all': test_acc_all,
                'test_loss_buckets': label_wise_loss if label_wise_loss else [],
                'test_acc_buckets': label_wise_correct if label_wise_correct else []
            }
            # Save model epoch-wise
            save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, batch_size=imgs.size(0), loss=loss, checkpoint_dir=save_path)
            # Save losses
            save_loss(loss, save_dir=save_path)

            #Output
            # Clear console output for next epoch
            # clear_output(wait=True)  
        # Show losses
        plot_training_progress(save_path, number_of_epochs=num_epochs)

        

    except Exception as e:
        # Free GPU memory in case of an error
        try:
            model.to('cpu')  # move model parameters to CPU
            del train_loader, test_loader, model, criterion, optimizer  # delete large tensors and optimizer
            torch.cuda.empty_cache()  # release cached memory
        except Exception as e_free:
            print("Error while freeing GPU memory.")
            print(e_free)
        finally:
            raise e
    
    # Free GPU memory
    try:
        model.to('cpu')  # move model parameters to CPU
        del imgs, labels, logits, loss, optimizer  # delete large tensors and optimizer
        torch.cuda.empty_cache()  # release cached memory
    except Exception as e_free:
        print("Error while freeing GPU memory.")
        print(e_free)

if __name__ == "__main__":
    pass