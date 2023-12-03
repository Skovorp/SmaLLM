### FILE FROM ANECDOTE HW

import torch
from typing import Type
from torch import nn
from dataset import TextDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions.categorical import Categorical


class RNNLanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super().__init__()
        # super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length
        self.hidden_size = hidden_size
        """
        YOUR CODE HERE (âŠƒï½¡â€¢Ìâ€¿â€¢Ì€ï½¡)âŠƒâ”âœ¿âœ¿âœ¿âœ¿âœ¿âœ¿
        Create necessary layers
        """
        self.embedding = nn.Embedding(self.vocab_size, embed_size, dataset.pad_id)
        self.rnn = rnn_type(embed_size, hidden_size, rnn_layers, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, self.vocab_size),
            nn.LeakyReLU(0.1)
        )

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
        # # This is a placeholder, you may remove it.
        # logits = torch.randn(
        #     indices.shape[0], indices.shape[1], self.vocab_size,
        #     device=indices.device
        # )
        # h0 = torch.zeros(indices.shape[0], indices.shape[1], self.hidden_size)
        embeds = self.embedding(indices)
        packed_embeds = pack_padded_sequence(embeds, lengths=lengths, batch_first=True, enforce_sorted=False)

        outp, _ = self.rnn(packed_embeds)  # h0 is zeros, outp is (batch, max_l, hidden)
        outp, _ = pad_packed_sequence(outp, batch_first=True)
        logits = self.linear(outp)  # (batch, max_l, vocab_size)
        """
        YOUR CODE HERE (âŠƒï½¡â€¢Ìâ€¿â€¢Ì€ï½¡)âŠƒâ”âœ¿âœ¿âœ¿âœ¿âœ¿âœ¿
        Convert indices to embeddings, pass them through recurrent layers
        and apply output linear layer to obtain the logits
        """

        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        # # This is a placeholder, you may remove it.
        # generated = prefix + ', Ð° Ð¿Ð¾Ñ‚Ð¾Ð¼ ÐºÑƒÐ¿Ð¸Ð» Ð¼ÑƒÐ¶Ð¸Ðº ÑˆÐ»ÑÐ¿Ñƒ, Ð° Ð¾Ð½Ð° ÐµÐ¼Ñƒ ÐºÐ°Ðº Ñ€Ð°Ð·.'
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

        # generate hidden for prefix
        embeds = self.embedding(tokens)
        _, hidden = self.rnn(embeds)
        if type(hidden) == tuple:
            logits = self.linear(hidden[0])
        else:
            logits = self.linear(hidden)

        # sample new token from logits
        new_tokens = Categorical(logits=logits / temp).sample()
        tokens = torch.cat([tokens, new_tokens])

        # 2 stopping conditions: reaching max len or getting <eos> token
        while tokens.shape[0] < self.max_length:
            if new_tokens.item() == self.dataset.eos_id or new_tokens.item() == self.dataset.unk_id:
                break

            # process newly obtained token
            embeds = self.embedding(new_tokens)
            _, hidden = self.rnn(embeds, hidden)
            if type(hidden) == tuple:
                logits = self.linear(hidden[0])
            else:
                logits = self.linear(hidden)
            # sample the next token from logits
            new_tokens = Categorical(logits=logits / temp).sample()
            tokens = torch.cat([tokens, new_tokens])

        # decode result to a string
        return self.dataset.ids2text(tokens.squeeze()[1:-1])


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