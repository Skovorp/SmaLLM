import os
import torch
from typing import Union, List, Tuple
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from torch.utils.data import Dataset
import random
import yaml
import json

# PARTS ARE FROM HOMEWORK LAST YEAR ABOUT ANECDOTE GENERATION

class TextDataset(Dataset):
    VAL_RATIO = 0.05

    def __init__(self, data_dir: str, train: bool = True, sp_model_prefix: str = None,
                 vocab_size: int = 2000, normalization_rule_name: str = 'nmt_nfkc_cf',
                 model_type: str = 'bpe', max_length: int = 128, limit=None, files_to_use=10, **kwargs):
        """
        Dataset with texts, supporting BPE tokenizer
        :param data_file: txt file containing texts
        :param train: whether to use train or validation split
        :param sp_model_prefix: path prefix to save tokenizer model
        :param vocab_size: sentencepiece tokenizer vocabulary size
        :param normalization_rule_name: sentencepiece tokenizer normalization rule
        :param model_type: sentencepiece tokenizer model type
        :param max_length: maximal length of text in tokens
        """

        texts = []
        for i, data_file in enumerate(os.listdir(data_dir)):
            if i == files_to_use:
                break
            with open(os.path.join(data_dir, data_file), 'r') as f:
                t = json.load(f)
                texts.extend([el['story'] for el in t])

        if not os.path.isfile(sp_model_prefix + '.model'):
            # train tokenizer if not trained yet
            SentencePieceTrainer.train(
                sentence_iterator=iter(texts), 
                vocab_size=vocab_size,
                model_type=model_type, model_prefix=sp_model_prefix,
                normalization_rule_name=normalization_rule_name,
                pad_id = 3
            )
        # load tokenizer from file
        self.sp_model = SentencePieceProcessor(model_file=sp_model_prefix + '.model')


        """
        Split texts to train and validation fixing self.TRAIN_VAL_RANDOM_SEED
        The validation ratio is self.VAL_RATIO
        """
        perm = torch.randperm(len(texts)).tolist()
        val_count = int(self.VAL_RATIO * len(perm))
        val_texts = [texts[i] for i in perm[:val_count]]
        train_texts = [texts[i] for i in perm[val_count:]]

        if limit is not None:
            val_texts = val_texts[:limit]
            train_texts = train_texts[:limit]

        # train_texts, val_texts = None, None
        self.texts = train_texts if train else val_texts
        self.indices = self.sp_model.encode(self.texts)

        self.pad_id, self.unk_id, self.bos_id, self.eos_id = \
            self.sp_model.pad_id(), self.sp_model.unk_id(), \
            self.sp_model.bos_id(), self.sp_model.eos_id()
        self.max_length = max_length
        self.vocab_size = self.sp_model.vocab_size()

    def text2ids(self, texts: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """
        Encode a text or list of texts as tokenized indices
        :param texts: text or list of texts to tokenize
        :return: encoded indices
        """
        return self.sp_model.encode(texts)

    def ids2text(self, ids: Union[torch.Tensor, List[int], List[List[int]]]) -> Union[str, List[str]]:
        """
        Decode indices as a text or list of tokens
        :param ids: 1D or 2D list (or torch.Tensor) of indices to decode
        :return: decoded texts
        """
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()

        return self.sp_model.decode(ids)

    def __len__(self):
        """
        Size of the dataset
        :return: number of texts in the dataset
        """
        return len(self.indices)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        """
        Add specials to the index array and pad to maximal length
        :param item: text id
        :return: encoded text indices and its actual length (including BOS and EOS specials)
        """
        # These are placeholders, you may remove them.
        # indices = torch.randint(high=self.vocab_size, size=(self.max_length, ))
        # length = torch.randint(low=1, high=self.max_length + 1, size=()).item()
        """
        Take corresponding index array from self.indices,
        add special tokens (self.bos_id and self.eos_id) and 
        pad to self.max_length using self.pad_id.
        Return padded indices of size (max_length, ) and its actual length
        """
        seq = self.indices[item]
        seq = seq[:self.max_length - 2]
        length = len(seq) + 2
        inds = torch.LongTensor([self.bos_id] + seq + [self.eos_id] + [self.pad_id] * (self.max_length - length))
        return inds, length


if __name__ == "__main__":
    config_path = '/home/ubuntu/SmaLLM/config.yaml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)


    train_set = TextDataset(**cfg['dataset'], train=True)
    # valid_set = TextDataset(data_file=cfg['dataset_path'], train=False, sp_model_prefix='bpe')

    # print(train_set[0])
    # print(train_set.ids2text(train_set[0][0]))
    # print("---------------")
    # print(train_set[10])
    # print(train_set.ids2text(train_set[10][0]))
    # print("---------------")
    # print(train_set[100])
    # print(train_set.ids2text(train_set[100][0]))
    # print("---------------")
    # print(train_set.pad_id)

    '/home/ubuntu/SmaLLM/saved/model_1.pth'

    # for _ in range(5):
    #     for dataset in (train_set, valid_set):
    #         indices, length = dataset[np.random.randint(len(dataset))]
    #         assert indices.shape == (dataset.max_length,)
    #         assert indices[0].item() == dataset.bos_id
    #         assert (indices == dataset.eos_id).sum().item() == 1

    #         eos_pos = indices.tolist().index(dataset.eos_id)
    #         assert torch.all(indices[eos_pos + 1:] == dataset.pad_id)
    #         assert (indices != dataset.pad_id).sum() == length