import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.modeling.tokenizer import tokenize_text
from src.utils.utils import save_data
from .dataset import Dataset
from .model import Model


def train(dataset, model, args):
    global losses
    seq_len = args['sequence_length']
    model.train()

    dataloader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(args['max_epochs']):
        losses = []
        for batch, (massages, massage_lens) in enumerate(dataloader):
            #             state_h, state_c = model.init_state(seq_len)
            max_massage_len = max(massage_lens).item()
            total_loss = 0
            for m in range(max_massage_len):
                mask = massage_lens > m
                x, y = (
                    torch.tensor(massages[mask, m:(m + seq_len)], dtype=torch.long),
                    torch.tensor(massages[mask, m + seq_len], dtype=torch.long)
                )

                optimizer.zero_grad()

                y_pred, _states = model(x)
                loss = criterion(y_pred, y)

                for state_h, state_c in _states:
                    state_h = state_h.detach()
                    state_c = state_c.detach()

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            losses.append(total_loss / max_massage_len)
            print({'epoch': epoch, 'batch': batch, 'loss': losses[-1]})


def train_model(text):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--max-epochs', type=int, default=10)
    # parser.add_argument('--batch-size', type=int, default=256)
    # parser.add_argument('--sequence-length', type=int, default=4)
    # args = parser.parse_args()

    args = {
        'max_epochs': 5,
        'batch_size': 256,
        'sequence_length': 5,
        'max_len': 100,

    }
    tokenized = text.apply(tokenize_text)
    dataset = Dataset(tokenized, **args)
    model = Model(dataset)

    train(dataset, model, args)

    # model.index_to_word = dataset.index_to_word
    # model.word_to_index = dataset.word_to_index
    from config import DATA_DIR
    save_data(model, DATA_DIR / 'models' / 'massage_model.pickle')
    # print(predict(dataset, model, text='בוקר טוב'))
