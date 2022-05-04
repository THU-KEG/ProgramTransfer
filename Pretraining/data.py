import json
import pickle
import torch
from Pretraining.utils import invert_dict

def load_vocab(path):
    vocab = json.load(open(path))
    vocab['id2function'] = invert_dict(vocab['function2id'])
    return vocab

def collate(batch):
    batch = list(zip(*batch))
    input_ids, token_type_ids, attention_mask, function_ids, relation_pos, relation_id = list(map(torch.stack, batch))
    return input_ids, token_type_ids, attention_mask, function_ids, relation_pos, relation_id

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.input_ids, self.token_type_ids, self.attention_mask, self.function_ids, self.relation_pos, self.relation_id = inputs

    def __getitem__(self, index):
        input_ids = torch.LongTensor(self.input_ids[index])
        token_type_ids = torch.LongTensor(self.token_type_ids[index])
        attention_mask = torch.LongTensor(self.attention_mask[index])
        function_ids = torch.LongTensor(self.function_ids[index])
        relation_pos = torch.LongTensor(self.relation_pos[index])
        relation_id = torch.LongTensor(self.relation_id[index])
        return input_ids, token_type_ids, attention_mask, function_ids, relation_pos, relation_id


    def __len__(self):
        return len(self.input_ids)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, vocab_json, question_pt, batch_size, training=False):
        vocab = load_vocab(vocab_json)
        inputs = []
        with open(question_pt, 'rb') as f:
            for _ in range(6):
                inputs.append(pickle.load(f))
        dataset = Dataset(inputs)

        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate, 
            )
        self.vocab = vocab