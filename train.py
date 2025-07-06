from collections import defaultdict
import os
import re
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
    # Prepare output path
    loss_dir  = os.path.join(save_dir, 'loss')
    os.makedirs(loss_dir, exist_ok=True)
    loss_path = os.path.join(loss_dir, "losses.json")

    # Load existing or initialise
    if os.path.exists(loss_path):
        with open(loss_path, 'r') as f:
            existing = json.load(f)
    else:
        existing = {}

    # Ensure every top‑level list exists
    # (in case someone renames keys later)
    for key in ['train_loss', 'train_acc', 'test_loss_all', 'test_acc_all']:
        existing.setdefault(key, [])

    # Append the global scalars
    for key in ['train_loss', 'train_acc', 'test_loss_all', 'test_acc_all']:
        if key in losses:
            existing[key].append(losses[key])

    # Flatten & append per-label buckets
    # losses['test_loss_buckets'] and losses['test_acc_buckets']
    for bucket_label, bucket_loss in losses.get('test_loss_buckets', {}).items():
        lbl_key = f"test_loss_label_{bucket_label}"
        existing.setdefault(lbl_key, []).append(bucket_loss)

    for bucket_label, bucket_acc in losses.get('test_acc_buckets', {}).items():
        lbl_key = f"test_acc_label_{bucket_label}"
        existing.setdefault(lbl_key, []).append(bucket_acc)

    # Write back out
    with open(loss_path, 'w') as f:
        json.dump(existing, f, indent=4)

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
    
    
def lighten_hex_color(hex_color, amount=0.5):
    hex_color = hex_color.lstrip('#')
    r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f'#{r:02x}{g:02x}{b:02x}'

def plot_training_progress(
    save_path,
    number_of_epochs=None,
    losses=None,
    include='batch',       # 'batch' | 'label' | 'both'
    renderer=None
):
    # Load JSON if not passed in
    loss_file = os.path.join(save_path, 'loss', "losses.json")
    if losses is None:
        if not os.path.exists(loss_file):
            raise FileNotFoundError(f"Losses file not found at {loss_file}.")
        with open(loss_file, 'r') as f:
            losses = json.load(f)

    # Determine epoch range
    if number_of_epochs is None:
        number_of_epochs = len(losses.get('train_loss', []))
    epochs = list(range(1, number_of_epochs + 1))

    # Prepare overall curves
    train_loss = losses['train_loss'][:number_of_epochs]
    test_loss  = losses['test_loss_all'][:number_of_epochs]
    train_acc  = losses['train_acc'][:number_of_epochs]
    test_acc   = losses['test_acc_all'][:number_of_epochs]

    # Detect label‑wise keys
    #    Keys in JSON are like 'test_loss_label_{label}' and 'test_acc_label_{label}'
    loss_keys = [k for k in losses if k.startswith('test_loss_label_')]
    # extract the label id from key
    labels = [re.findall(r'test_loss_label_(.+)', k)[0] for k in loss_keys]
    # sort labels naturally
    labels = sorted(labels, key=lambda x: int(x) if x.isdigit() else x)

    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add overall (batch) curves if requested
    if include in ('batch', 'both'):
        # use fixed colors for batch curves
        fig.add_trace(go.Scatter(
            x=epochs, y=train_loss, mode='lines+markers', name='Train Loss'
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=epochs, y=test_loss,  mode='lines+markers', name='Test Loss'
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=epochs, y=train_acc,  mode='lines+markers', name='Train Acc'
        ), secondary_y=True)
        fig.add_trace(go.Scatter(
            x=epochs, y=test_acc,   mode='lines+markers', name='Test Acc'
        ), secondary_y=True)

    # Add per‑label curves if requested
    if include in ('label', 'both') and labels:
        # pick a qualitative palette
        palette = px.colors.qualitative.Plotly
        for i, label in enumerate(labels):
            color = palette[i % len(palette)]
            light_color = lighten_hex_color(color, amount=0.6)

            # fetch data
            lkey = f'test_loss_label_{label}'
            akey = f'test_acc_label_{label}'
            lbl_loss = losses.get(lkey, [])[:number_of_epochs]
            lbl_acc  = losses.get(akey, [])[:number_of_epochs]

            # add loss
            fig.add_trace(go.Scatter(
                x=epochs, y=lbl_loss,
                mode='lines+markers',
                name=f'Loss [{label}]',
                line=dict(color=color),
            ), secondary_y=False)

            # add accuracy
            fig.add_trace(go.Scatter(
                x=epochs, y=lbl_acc,
                mode='lines+markers',
                name=f'Acc [{label}]',
                line=dict(color=light_color),
            ), secondary_y=True)

    # Axis formatting
    fig.update_xaxes(
        title_text='Epoch',
        range=[1, number_of_epochs],
        tickmode='linear',
        dtick=1
    )
    fig.update_yaxes(title_text='Loss',     secondary_y=False)
    fig.update_yaxes(title_text='Accuracy', secondary_y=True)

    # Layout
    fig.update_layout(
        title='Training Progress',
        legend=dict(x=0.01, y=0.99, bordercolor="black", borderwidth=1),
        template='simple_white',
        height=600, width=900
    )

    # Show
    if renderer:
        fig.show(renderer=renderer)
    else:
        fig.show()
    

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
    
def train(
    model,
    train_loader,
    test_loader,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    criterion=None,
    optimizer=None,
    num_epochs=10,
    lr=1e-4,
    save_path="./checkpoints",
    label_map=None,
    save_attention=False,
    history_length=6
):
    # Setup
    try:
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        if optimizer is None:
            optimizer = optim.AdamW(model.parameters(), lr=lr)

        os.makedirs(save_path, exist_ok=True)
        model = nn.DataParallel(model)
        model.to(device)

        reverse_label_map = {v: k for k, v in label_map.items()} if label_map else None

        clear_output(wait=True)
        print(f"Using {torch.cuda.device_count()} GPUs for training.")

        for epoch in range(1, num_epochs + 1):
            ############
            # TRAINING #
            ############
            model.train()
            running_loss = 0.0
            correct      = 0

            for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit=" batch"):
                imgs   = imgs.to(device)
                labels = parse_labels(labels, label_map).to(device)

                optimizer.zero_grad()
                logits, _ = model(imgs, return_attn=False)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()

            epoch_loss = running_loss / len(train_loader.dataset)
            train_acc  = correct      / len(train_loader.dataset)
            print(f"Epoch {epoch}:")
            print(f"\t— train loss: {epoch_loss:.4f}, train acc: {train_acc:.4f}")

            ############
            #   TEST   #
            ############
            model.eval()
            test_loss_all   = 0.0
            correct_all     = 0

            # Initialise per‑label accumulators
            if label_map:
                label_wise_loss    = {bucket_label: 0.0 for bucket_label in label_map.values()}
                label_wise_correct = {bucket_label: 0   for bucket_label in label_map.values()}
                label_wise_count   = {bucket_label: 0   for bucket_label in label_map.values()}
            else:
                label_wise_loss = label_wise_correct = label_wise_count = None

            with torch.no_grad():
                for imgs, labels in tqdm(test_loader, desc="Test", unit=" batch"):
                    imgs   = imgs.to(device)
                    labels = parse_labels(labels, label_map).to(device)

                    logits, attn_maps = model(imgs, return_attn=save_attention)
                    batch_size = imgs.size(0)

                    # Global test metrics
                    test_loss_all += criterion(logits, labels).item() * batch_size
                    preds = logits.argmax(dim=-1)
                    correct_all += (preds == labels).sum().item()

                    # Per‑label metrics
                    if label_map:
                        for bucket_label in label_map.values():
                            mask = (labels == bucket_label)
                            if not mask.any():
                                continue
                            n = mask.sum().item()
                            label_wise_count[bucket_label]   += n
                            label_wise_loss[bucket_label]    += (
                                criterion(logits[mask], labels[mask]).item() * n
                            )
                            label_wise_correct[bucket_label] += (preds[mask] == labels[mask]).sum().item()

                    # Save attention maps on the last epoch if requested
                    if save_attention and epoch == num_epochs:
                        save_attention_weights(attn_maps, labels, epoch)

            # Finalize global test metrics
            test_loss_all /= len(test_loader.dataset)
            test_acc_all   = correct_all / len(test_loader.dataset)
            print(f"\t— test loss: {test_loss_all:.4f}, test acc: {test_acc_all:.4f}")

            # Print per‑label results
            if label_map:
                for bucket_label in label_map.values():
                    cnt = label_wise_count[bucket_label]
                    if cnt > 0:
                        avg_loss = label_wise_loss[bucket_label] / cnt
                        acc      = label_wise_correct[bucket_label] / cnt
                    else:
                        avg_loss = 0.0
                        acc      = 0.0
                    name = reverse_label_map[bucket_label] if reverse_label_map else str(bucket_label)
                    print(f"\t\t—— {name}: loss = {avg_loss:.4f}, acc = {acc:.4f}")

            # Save checkpoint & losses
            metrics = {
                'train_loss':   epoch_loss,
                'train_acc':    train_acc,
                'test_loss_all': test_loss_all,
                'test_acc_all':  test_acc_all,
                'test_loss_buckets':  label_wise_loss    if label_map else {},
                'test_acc_buckets':   label_wise_correct if label_map else {}
            }
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                batch_size=batch_size,
                loss=metrics,
                checkpoint_dir=save_path,
                history_length=history_length
            )
            save_loss(metrics, save_dir=save_path)

        # Plot overall training curves at the end
        plot_training_progress(save_path, number_of_epochs=num_epochs)

    except Exception as e:
        # Attempt to free GPU memory on error
        try:
            model.to('cpu')
            del train_loader, test_loader, model, criterion, optimizer
            torch.cuda.empty_cache()
        except:
            pass
        finally:
            raise e

    # Finally free GPU memory
    try:
        model.to('cpu')
        del imgs, labels, logits, loss, optimizer
        torch.cuda.empty_cache()
    except:
        pass


if __name__ == "__main__":
    pass 
