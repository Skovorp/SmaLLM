import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import List, Optional, Any
from tqdm import tqdm
import os
import wandb
from copy import copy

from dataset import TextDataset
from transformer_model import TransformerLanguageModel
from utils import CosineAnnealingWithWarmupLR

# from dotenv import load_dotenv   -- I want to use this so bad, but extra dependencies are banned!!!
# load_dotenv()

def training_epoch(model, optimizer: torch.optim.Optimizer, scheduler, criterion: nn.Module,
                   loader: DataLoader, epoch_size: int, tqdm_desc: str):
    device = next(model.parameters()).device
    train_loss = 0.0
    seen_objects = 0
    seen_tokens = 0

    model.train()
    for step_num, (indices, lengths) in tqdm(enumerate(loader), total=epoch_size):
        if step_num == epoch_size:
            break
        optimizer.zero_grad()
        indices = indices[:, :lengths.max()].to(device)
        logits = model(indices, lengths)
        loss = criterion(logits[:, :-1].transpose(1, 2), indices[:, 1:])

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        train_loss += loss.item() * indices.shape[0]
        wandb.log({'train_loss': loss.item()})
        seen_objects += indices.shape[0]
        seen_tokens += torch.sum(lengths).item()
    print(f"Saw {seen_tokens:_} tokens this epoch")
    train_loss /= seen_objects
    return train_loss


@torch.no_grad()
def validation_epoch(model, criterion: nn.Module, loader: DataLoader, tqdm_desc: str):
    val_loss = 0.0
    device = next(model.parameters()).device

    model.eval()
    for indices, lengths in tqdm(loader, desc=tqdm_desc):
        indices = indices[:, :lengths.max()].to(device)
        logits = model(indices, lengths)
        loss = criterion(logits[:, :-1].transpose(1, 2), indices[:, 1:])
        val_loss += loss.item() * indices.shape[0]

    val_loss /= len(loader.dataset)
    print("val loss:", val_loss)

    return val_loss


def train(model, optimizer: torch.optim.Optimizer, scheduler: Optional[Any], criterion,
          train_loader: DataLoader, val_loader: DataLoader, save_path: str, num_epochs: int, num_examples=1):
    train_losses, val_losses = [], []
    
    examples_table = wandb.Table(columns=['epoch', 'example'])
    for epoch in range(1, num_epochs + 1):
        train_loss = training_epoch(
            model, optimizer, scheduler, criterion, train_loader, cfg['training']['epoch_size'],
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        print(f"Epoch {epoch}/{num_epochs}. train_loss: {train_loss}")
        wandb.log({
            "epoch": epoch, 
            # "train_loss": train_loss,
            "lr": optimizer.param_groups[0]['lr']})

        val_loss = validation_epoch(model, criterion, val_loader, tqdm_desc=f'Validation {epoch}/{num_epochs}')
        # print(f"val_loss: {val_loss}")
        wandb.log({'val_loss': val_loss})
        example = model.inference(temp=2)
        print(example)
        examples_table.add_data(epoch, example)
        wandb.log({'examples': copy(examples_table)})
        torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    SEED = 123
    torch.manual_seed(SEED)


    config_path = '/home/ubuntu/SmaLLM/config.yaml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    os.environ["WANDB_API_KEY"] = "9ce4b10bf35b56281619a07601ec9c274604c9f6"
    wandb.init(
    project="smallm",
    config=cfg
)

    # data
    print(cfg)
    train_set = TextDataset(**cfg['dataset'], limit=cfg['dataset']['train_limit'], train=True)
    print("VS", train_set.vocab_size)
    val_set = TextDataset(**cfg['dataset'], limit=cfg['dataset']['val_limit'], train=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg['training']['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg['training']['batch_size'], shuffle=False)
    print("len val set", len(val_set))
    print("len train set", len(train_set))
    print("len val loader", len(val_loader))
    print("len train loader", len(train_loader))

    # stuff
    model = TransformerLanguageModel(train_set, **cfg['model']).to(torch.device('cuda'))
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])
    scheduler = CosineAnnealingWithWarmupLR(
        optimizer,
        warmup_steps=cfg['training']['warmup_steps'],
        max_steps=int(cfg['training']['num_epochs'] * len(train_loader))
    )
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.pad_id)

    train(model, optimizer, scheduler, criterion, train_loader, val_loader, cfg['training']['save_path'], cfg['training']['num_epochs'])
    wandb.finish()
