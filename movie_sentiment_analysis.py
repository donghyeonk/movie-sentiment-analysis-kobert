from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import math
from gluonnlp.data import SentencepieceTokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from pytorch_transformers import AdamW, WarmupLinearSchedule

# Dataset
# https://github.com/e9t/nsmc.git

# BERT Model
# https://github.com/SKTBrain/KoBERT

# Optimizer
# https://github.com/huggingface/pytorch-transformers#optimizers-bertadam--openaiadam-are-now-adamw-schedules-are-standard-pytorch-schedules


def train(train_loader, device, model, linear, all_params, optimizer, scheduler,
          max_grad_norm, log_interval, epoch):
    model.train()
    linear.train()
    for batch_idx, (input_ids, token_type_ids, input_mask, target) \
            in enumerate(train_loader):
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        input_mask = input_mask.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        _, pooled_output = model(input_ids, token_type_ids, input_mask)
        logits = linear(pooled_output)
        output = F.log_softmax(logits, dim=1)

        loss = F.nll_loss(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
        optimizer.step()
        scheduler.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()

        if (batch_idx + 1) % log_interval == 0 \
                or batch_idx == len(train_loader) - 1:
            batch_len = len(input_ids)
            lr = ''
            for param_group in optimizer.param_groups:
                if 'lr' in param_group:
                    lr = param_group['lr']
                    break
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                  '\tAccuracy: {}/{} ({:.2f}%)\tlr: {:.3e}'.format(
                    datetime.now(),
                    epoch, (batch_idx + 1) * batch_len,
                    len(train_loader.dataset),
                    100. * (batch_idx + 1) / len(train_loader), loss.item(),
                    correct, batch_len, 100. * correct / batch_len,
                    lr))


def test(test_loader, device, model, linear):
    model.eval()
    linear.eval()
    eval_loss = 0.
    correct = 0
    start_t = datetime.now()
    with torch.no_grad():
        for batch_idx, (input_ids, token_type_ids, input_mask, target) \
                in enumerate(test_loader):
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            input_mask = input_mask.to(device)
            target = target.to(device)

            _, pooled_output = model(input_ids, token_type_ids, input_mask)
            logits = linear(pooled_output)
            output = F.log_softmax(logits, dim=1)

            eval_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    eval_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    print('Elapsed time: {}, Test, Avg. Loss: {:.6f}, '
          'Accuracy: {}/{} ({:.2f}%)\n'.format(datetime.now() - start_t,
                                               eval_loss,
                                               correct,
                                               len(test_loader.dataset),
                                               100. * acc))


class MovieDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def batchify(b):
    x_len = [len(e[0]) for e in b]
    batch_max_len = max(x_len)

    x = list()
    tk_type_ids = list()
    x_mask = list()
    y = list()
    for e in b:
        seq_len = len(e[0])
        e0_mask = [1] * seq_len  # 1: MASK
        while len(e[0]) < batch_max_len:
            e[0].append(0)  # 0: '[PAD]'
            e0_mask.append(0)
        assert len(e[0]) == batch_max_len

        e0_tk_type_ids = [0] * batch_max_len  #
        # e0_tk_type_ids[seq_len - 1] = 1

        x.append(e[0])
        tk_type_ids.append(e0_tk_type_ids)
        x_mask.append(e0_mask)
        y.append(e[1])

    x = torch.tensor(x, dtype=torch.int64)
    tk_type_ids = torch.tensor(tk_type_ids, dtype=torch.int64)
    x_mask = torch.tensor(x_mask, dtype=torch.int64)
    y = torch.tensor(y, dtype=torch.int64)

    return x, tk_type_ids, x_mask, y


def get_data(filepath, vocab, sp):
    data = list()
    max_seq_len = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for lidx, l in enumerate(f):
            if 0 == lidx:
                continue
            cols = l[:-1].split('\t')
            # docid = cols[0]
            doc = cols[1]
            label = cols[2]

            token_ids = list()
            token_ids.append(vocab['[CLS]'])
            for t in sp(doc):
                if t in vocab:
                    token_ids.append(vocab[t])
                else:
                    token_ids.append(vocab['[UNK]'])
            token_ids.append(vocab['[SEP]'])

            data.append([token_ids, int(label)])

            if max_seq_len < len(token_ids):
                max_seq_len = len(token_ids)
    print('max_seq_len', max_seq_len)
    return data


def main():
    nsmc_home_dir = 'YOUR_NSMC_DIR'
    train_file = nsmc_home_dir + '/ratings_train.txt'  # 150K
    test_file = nsmc_home_dir + '/ratings_test.txt'  # 50K

    model, vocab = get_pytorch_kobert_model(
        ctx='cuda' if torch.cuda.is_available() else 'cpu')

    lr = 5e-5
    batch_size = 16
    epochs = 5
    max_grad_norm = 1.0
    num_total_steps = math.ceil(150000 / batch_size) * epochs
    num_warmup_steps = num_total_steps // 10
    log_interval = 100
    seed = 2019
    num_workers = 2
    num_classes = 2
    pooler_out_dim = model.pooler.dense.out_features

    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('device', device)

    tok_path = get_tokenizer()
    sp = SentencepieceTokenizer(tok_path)

    train_loader = torch.utils.data.DataLoader(
        MovieDataset(get_data(train_file, vocab, sp)),
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=batchify,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        MovieDataset(get_data(test_file, vocab, sp)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=batchify,
        pin_memory=True
    )

    linear = torch.nn.Linear(pooler_out_dim, num_classes).to(device)

    all_params = list(model.parameters()) + list(linear.parameters())
    optimizer = AdamW(all_params, lr=lr, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps,
                                     t_total=num_total_steps)

    for epoch in range(epochs):
        train(train_loader, device, model, linear, all_params,
              optimizer, scheduler, max_grad_norm, log_interval, epoch)
        print(datetime.now(), 'Testing...')
        test(test_loader, device, model, linear)


if __name__ == '__main__':
    main()
