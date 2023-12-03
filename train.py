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
from benchmark_model import RNNLanguageModel
from transformer_model import TransformerLanguageModel
from utils import CosineAnnealingWithWarmupLR

# from dotenv import load_dotenv   -- I want to use this so bad, but extra dependencies are banned!!!
# load_dotenv()

def training_epoch(model, optimizer: torch.optim.Optimizer, scheduler, criterion: nn.Module,
                   loader: DataLoader, epoch_size: int, tqdm_desc: str):
    """
    Process one training epoch
    :param model: language model to train
    :param optimizer: optimizer instance
    :param criterion: loss function class
    :param loader: training dataloader
    :param tqdm_desc: progress bar description
    :return: running train loss
    """
    device = next(model.parameters()).device
    train_loss = 0.0

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

    train_loss /= len(loader.dataset)
    return train_loss


@torch.no_grad()
def validation_epoch(model, criterion: nn.Module,
                     loader: DataLoader, tqdm_desc: str):
    """
    Process one validation epoch
    :param model: language model to validate
    :param criterion: loss function class
    :param loader: validation dataloader
    :param tqdm_desc: progress bar description
    :return: validation loss
    """
    device = next(model.parameters()).device
    val_loss = 0.0

    model.eval()
    for indices, lengths in tqdm(loader):
        """
        YOUR CODE HERE (âŠƒï½¡â€¢Ìâ€¿â€¢Ì€ï½¡)âŠƒâ”âœ¿âœ¿âœ¿âœ¿âœ¿âœ¿
        Process one validation step: calculate loss.
        Accumulate sum of losses for different batches in val_loss
        """
        indices = indices[:, :lengths.max()].to(device)
        logits = model(indices, lengths)[:, :-1]
        loss = criterion(logits.transpose(1, 2), indices[:, 1:])
        val_loss += loss.item() * indices.shape[0]

    val_loss /= len(loader.dataset)
    return val_loss


def train(model, optimizer: torch.optim.Optimizer, scheduler: Optional[Any], criterion,
          train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, num_examples=1):
    train_losses, val_losses = [], []
    
    examples_table = wandb.Table(columns=['epoch', 'example'])
    for epoch in range(1, num_epochs + 1):
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg['training']['batch_size'], shuffle=True)
        train_loss = training_epoch(
            model, optimizer, scheduler, criterion, train_loader, cfg['training']['epoch_size'],
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        print(f"Epoch {epoch}/{num_epochs}. train_loss: {train_loss}")
        wandb.log({
            "epoch": epoch, 
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]['lr']})
        # val_loss = validation_epoch(
        #     model, criterion, val_loader,
        #     tqdm_desc=f'Validating {epoch}/{num_epochs}'
        # )

        
        # train_losses += [train_loss]
        # val_losses += [val_loss]
        examples_table.add_data(epoch, model.inference())
        wandb.log({'examples': copy(examples_table)})
        # print('Example:', model.inference())
        # print(num_examples)
        # for _ in range(num_examples):
        #     print(model.inference())


if __name__ == "__main__":
    SEED = 123
    torch.manual_seed(SEED)


    config_path = '/home/ubuntu/smaLLLLLLLm/config.yaml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    os.environ["WANDB_API_KEY"] = "9ce4b10bf35b56281619a07601ec9c274604c9f6"
    wandb.init(
    project="smallm",
    config=cfg,
    mode='disabled'
)

    # data
    train_set = TextDataset(**cfg['dataset'], train=True)
    val_set = TextDataset(**cfg['dataset'], train=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg['training']['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg['training']['batch_size'], shuffle=False)

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

    train(model, optimizer, scheduler, criterion, train_loader, val_loader, cfg['training']['num_epochs'])
    wandb.finish()
