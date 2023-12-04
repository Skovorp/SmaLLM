import torch
from typing import Type
from torch import nn
from dataset import TextDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions.categorical import Categorical
from utils import PositionalEncoding


class TransformerLanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_dim, hidden_size, n_transformer_blocks, n_heads, **kwargs):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super().__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(self.vocab_size, embed_dim, dataset.pad_id)
        self.pos_encoding = PositionalEncoding(embed_dim, dataset.max_length)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=n_heads,
            dim_feedforward=hidden_size,
            batch_first=True)

        self.decoder_stack = nn.TransformerDecoder(decoder_layer, n_transformer_blocks)
        self.linear = nn.Linear(embed_dim, dataset.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        device = indices.device
        embeds = self.embedding(indices)
        embeds = self.pos_encoding(embeds)

        l = embeds.shape[1]
        # mask = torch.log(torch.triu(torch.ones((l, l)))).to(device)
        mask = torch.triu(torch.ones((l, l), device=device), diagonal=1).bool()

        res = self.decoder_stack(tgt=embeds, memory=embeds, tgt_mask=mask, memory_mask=mask)
        res = self.linear(res)
        return res

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        """
        YOUR CODE HERE (âŠƒï½¡â€¢Ìâ€¿â€¢Ì€ï½¡)âŠƒâ”âœ¿âœ¿âœ¿âœ¿âœ¿âœ¿
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """
        # encode prefix
        device = next(self.embedding.parameters()).device
        tokens = torch.IntTensor([self.dataset.bos_id] + self.dataset.text2ids(prefix)).to(device)

        # 2 stopping conditions: reaching max len or getting <eos> token
        for i in range(tokens.shape[0], self.max_length):
            embeds = self.embedding(tokens)
            # print("emb shape", embeds.shape)
            embeds = self.pos_encoding(embeds)
            l = embeds.shape[0]
            mask = torch.triu(torch.ones((l, l), device=device), diagonal=1).bool()
            # mask = torch.log(torch.triu(torch.ones((l, l)))).to(device)
            # mask = torch.triu(torch.ones((self.max_length, self.max_length)), diagonal=1)
            # print(embeds.shape, mask.shape)
            res = self.decoder_stack(tgt=embeds, memory=embeds, tgt_mask=mask, memory_mask=mask)
            # print(res.shape)
            res = self.linear(res)

            logits = res[i - 1, :]
            new_tokens = Categorical(logits=logits / temp).sample()
            tokens = torch.cat([tokens, torch.unsqueeze(new_tokens, dim=0)])
            if new_tokens.item() == self.dataset.eos_id or new_tokens.item() == self.dataset.unk_id:
                break
            # print(tokens.shape)

            # # process newly obtained token
            # embeds = self.embedding(new_tokens)
            # _, hidden = self.rnn(embeds, hidden)
            # if type(hidden) == tuple:
            #     logits = self.linear(hidden[0])
            # else:
            #     logits = self.linear(hidden)
            # # sample the next token from logits
            
            # tokens = torch.cat([tokens, new_tokens])

        # decode result to a string
        return self.dataset.ids2text(tokens.squeeze()[1:-1])


    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([torch.prod(torch.tensor(p.size())).item() for p in model_parameters])
        return super().__str__() + f"\nTrainable parameters: {params:_}"


if __name__ == "__main__":
    import yaml
    config_path = '/home/ubuntu/smaLLLLLLLm/config.yaml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    train_set = TextDataset(**cfg['dataset'], train=True)
    model = LanguageModel(train_set)

    for bs in [1, 4, 16, 64, 256]:
        indices = torch.randint(high=train_set.vocab_size, size=(bs, train_set.max_length))
        lengths = torch.randint(low=1, high=train_set.max_length + 1, size=(bs,))
        logits = model(indices, lengths)
        assert logits.shape == (bs, lengths.max(), train_set.vocab_size)

    for prefix in ['', 'ÐºÑƒÐ¿Ð¸Ð» Ð¼ÑƒÐ¶Ð¸Ðº ÑˆÐ»ÑÐ¿Ñƒ,', 'ÑÐµÐ» Ð¼ÐµÐ´Ð²ÐµÐ´ÑŒ Ð² Ð¼Ð°ÑˆÐ¸Ð½Ñƒ Ð¸', 'Ð¿Ð¾Ð´ÑƒÐ¼Ð°Ð» ÑˆÑ‚Ð¸Ñ€Ð»Ð¸Ñ†']:
        generated = model.inference(prefix, temp=0.5)
        assert type(generated) == str
        assert generated.startswith(prefix)
        print(f"Generated: {generated}")