from transformer_model import TransformerLanguageModel
from dataset import TextDataset
import yaml
import torch
from torch import nn

config_path = '/home/ubuntu/SmaLLM/config.yaml'
with open(config_path) as f:
    cfg = yaml.safe_load(f)


val_set = TextDataset(**cfg['dataset'], train=False)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg['training']['batch_size'], shuffle=False)

# stuff
model = TransformerLanguageModel(val_set, **cfg['model']).to(torch.device('cuda'))
model.load_state_dict(torch.load('/home/ubuntu/SmaLLM/saved/model_1.pth'))

print(len(val_set))
inds, lengths = (next(iter(val_loader)))
print(inds.shape)
inds, lengths = inds.to(torch.device('cuda')), lengths.to(torch.device('cuda'))
res = model(inds, lengths)

criterion = nn.CrossEntropyLoss(ignore_index=3)
criterion_bad = nn.CrossEntropyLoss()

print(inds[1])
print(res[1].argmax(dim=1))


print(criterion(res[1, :-1], inds[1, 1:]))
print(criterion_bad(res[1, :-1], inds[1, 1:]))
print(criterion_bad(res[1, :lengths[1]-1], inds[1, 1:lengths[1]]))


