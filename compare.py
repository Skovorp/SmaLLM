from transformer_model import TransformerLanguageModel
from dataset import TextDataset
import yaml
import torch
from torch import nn
import transformers


transformers.set_seed(123)
gpt2 = transformers.pipeline('text-generation', model='gpt2-xl')

config_path = '/home/ubuntu/SmaLLM/config.yaml'
with open(config_path) as f:
    cfg = yaml.safe_load(f)


val_set = TextDataset(**cfg['dataset'], train=False)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg['training']['batch_size'], shuffle=False)

# stuff
model = TransformerLanguageModel(val_set, **cfg['model']).to(torch.device('cuda'))
model.load_state_dict(torch.load('/home/ubuntu/SmaLLM/saved/model_fix_night_run.pth'))

model.eval()
prefix = 'Once upon a time'
for i in range(50):
    example = model.inference(prefix=prefix, temp=1)
    print("Smallm: ", example)
    print('\n')
print("=========")

gpt2_results = gpt2(prefix, max_length=256, num_return_sequences=50)
for el in gpt2_results:
    print("GPT2-XL: ", el)
    print('\n')