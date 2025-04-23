"""Training code for the detector model"""

import argparse
import os
import subprocess
import sys
from itertools import count
from multiprocessing import Process

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from tqdm import tqdm
from transformers import *

from dataset import Corpus, EncodedDataset, TuringBenchDataset
from utils import summary, distributed

from sklearn.metrics import precision_recall_fscore_support
from fvcore.nn import FlopCountAnalysis
from typing import Tuple

def setup_distributed(port=29500):
    if not dist.is_available() or not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return 0, 1

    if 'MPIR_CVAR_CH3_INTERFACE_HOSTNAME' in os.environ:
        from mpi4py import MPI
        mpi_rank = MPI.COMM_WORLD.Get_rank()
        mpi_size = MPI.COMM_WORLD.Get_size()

        os.environ["MASTER_ADDR"] = '127.0.0.1'
        os.environ["MASTER_PORT"] = str(port)

        dist.init_process_group(backend="nccl", world_size=mpi_size, rank=mpi_rank)
        return mpi_rank, mpi_size

    dist.init_process_group(backend="nccl", init_method="env://")
    return dist.get_rank(), dist.get_world_size()


def load_datasets(data_dir, real_dataset, fake_dataset, tokenizer, batch_size,
                  max_sequence_length, random_sequence_length, epoch_size=None, token_dropout=None, seed=None, extra_fake_dataset=None):
    if fake_dataset == "TWO":
        real_train, real_valid = real_corpus.train * 2, real_corpus.valid * 2
        fake_corpora = [Corpus(name, data_dir=data_dir) for name in ['xl-1542M', 'xl-1542M-nucleus']]
        fake_train = sum([corpus.train for corpus in fake_corpora], [])
        fake_valid = sum([corpus.valid for corpus in fake_corpora], [])
    elif fake_dataset == "THREE":
        real_train, real_valid = real_corpus.train * 3, real_corpus.valid * 3
        fake_corpora = [Corpus(name, data_dir=data_dir) for name in
                        ['xl-1542M', 'xl-1542M-k40', 'xl-1542M-nucleus']]
        fake_train = sum([corpus.train for corpus in fake_corpora], [])
        fake_valid = sum([corpus.valid for corpus in fake_corpora], [])
    else:
        fake_corpus = Corpus(fake_dataset, data_dir=data_dir)
        fake_train, fake_valid = fake_corpus.train, fake_corpus.valid

        if extra_fake_dataset is not None:
            fake_train_len, fake_valid_len = len(fake_train), len(fake_valid)
            extra_fake_corpus = Corpus(extra_fake_dataset, data_dir=data_dir, max_train_len=fake_train_len, max_valid_len=fake_valid_len)
            extra_fake_train, extra_fake_valid = extra_fake_corpus.train, extra_fake_corpus.valid
            fake_train.extend(extra_fake_train)
            fake_valid.extend(extra_fake_valid)

        if real_dataset != "":
            fake_train_len, fake_valid_len = len(fake_train), len(fake_valid)
            real_corpus = Corpus(real_dataset, data_dir=data_dir, max_train_len=fake_train_len, max_valid_len=fake_valid_len)
            real_train, real_valid = real_corpus.train, real_corpus.valid

    Sampler = DistributedSampler if distributed() and dist.get_world_size() > 1 else RandomSampler

    min_sequence_length = 10 if random_sequence_length else None
    if real_dataset != "":
        train_dataset = EncodedDataset(real_train, fake_train, tokenizer, max_sequence_length, min_sequence_length,
                                    epoch_size, token_dropout, seed)
    else:
        train_dataset = EncodedDataset([], fake_train, tokenizer, max_sequence_length, min_sequence_length,
                                    epoch_size, token_dropout, seed)
    train_loader = DataLoader(train_dataset, batch_size, sampler=Sampler(train_dataset))

    if real_dataset != "":
        validation_dataset = EncodedDataset(real_valid, fake_valid, tokenizer)
    else:
        validation_dataset = EncodedDataset([], fake_valid, tokenizer)
    validation_loader = DataLoader(validation_dataset, batch_size=1,sampler=Sampler(validation_dataset))

    return train_loader, validation_loader

def load_turing_datasets(train_dir, val_dir, tokenizer, batch_size, 
    max_sequence_length, random_sequence_length, epoch_size=None, token_dropout=None, seed=None):
    Sampler = DistributedSampler if distributed() and dist.get_world_size() > 1 else RandomSampler

    min_sequence_length = 10 if random_sequence_length else None
    train_dataset = TuringBenchDataset(train_dir, tokenizer, max_sequence_length, min_sequence_length,
                                   epoch_size, token_dropout, seed)
    train_loader = DataLoader(train_dataset, batch_size, sampler=Sampler(train_dataset), num_workers=0)

    validation_dataset = TuringBenchDataset(val_dir, tokenizer)
    validation_loader = DataLoader(validation_dataset, batch_size=1,sampler=Sampler(validation_dataset))

    return train_loader, validation_loader

def accuracy_sum(logits, labels):
    if list(logits.shape) == list(labels.shape) + [2]:
        # 2-d outputs
        classification = (logits[..., 0] < logits[..., 1]).long().flatten()
    else:
        classification = (logits > 0).long().flatten()
    assert classification.shape == labels.shape
    return (classification == labels).float().sum().item(), classification


def train(model: nn.Module, optimizer, device: str, loader: DataLoader, desc='Train'):
    model.train()

    train_accuracy = 0
    train_epoch_size = 0
    train_loss = 0
    
    train_accs = []
    train_losses = []

    with tqdm(loader, desc=desc, disable=distributed() and dist.get_rank() > 0) as loop:
        for texts, masks, labels in loop:

            texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
            batch_size = texts.shape[0]

            optimizer.zero_grad()
            loss, logits = model(texts, attention_mask=masks, labels=labels, return_dict=False)
            loss.backward()
            optimizer.step()

            batch_accuracy, _ = accuracy_sum(logits, labels)
            train_accuracy += batch_accuracy
            train_epoch_size += batch_size
            train_loss += loss.item() * batch_size

            loop.set_postfix(loss=loss.item(), acc=train_accuracy / train_epoch_size)
            train_accs.append(batch_accuracy / batch_size)
            train_losses.append(loss.item())

    return {
        "train/accuracy": train_accuracy,
        "train/epoch_size": train_epoch_size,
        "train/loss": train_loss,
        "train_losses": torch.tensor(train_losses),
        "train_accs": torch.tensor(train_accs)
    }


def validate(model: nn.Module, device: str, loader: DataLoader, votes=1, desc='Validation'):
    model.eval()

    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0

    all_preds = []
    all_labels = []

    records = [record for v in range(votes) for record in tqdm(loader, desc=f'Preloading data ... {v}',
                                                               disable=False)]
    records = [[records[v * len(loader) + i] for v in range(votes)] for i in range(len(loader))]

    with tqdm(records, desc=desc, disable=distributed() and dist.get_rank() > 0) as loop, torch.no_grad():
        for example in loop:
            losses = []
            logit_votes = []

            for texts, masks, labels in example:
                texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
                batch_size = texts.shape[0]

                loss, logits = model(texts, attention_mask=masks, labels=labels, return_dict=False)
                losses.append(loss)
                logit_votes.append(logits)

            loss = torch.stack(losses).mean(dim=0)
            logits = torch.stack(logit_votes).mean(dim=0)

            batch_accuracy, preds = accuracy_sum(logits, labels)
            validation_accuracy += batch_accuracy
            validation_epoch_size += batch_size
            validation_loss += loss.item() * batch_size

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            loop.set_postfix(loss=loss.item(), acc=validation_accuracy / validation_epoch_size)

    return {
        "validation/accuracy": validation_accuracy,
        "validation/epoch_size": validation_epoch_size,
        "validation/loss": validation_loss
    }

def estimate_total_flops(
    model: nn.Module,
    input_size: Tuple[int],
    N: int,
    batch_size: int,
    epochs: int,
    trainable_ratio = 1
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    dummy_input = torch.randint(50265, (1, *input_size), device=device)
    flops_forward = FlopCountAnalysis(model, dummy_input).total()

    # In training: forward + backward + optimizer step
    # Rule of thumb: backward ≈ 2x forward, optimizer ≈ 1x forward
    flops_per_sample = flops_forward * (1 + 2 * trainable_ratio + 1 * trainable_ratio)  # total ≈ 4x forward flops

    total_steps = (N * batch_size) * epochs
    total_flops = flops_per_sample * batch_size * total_steps

    print(f"FLOPs per sample (forward): {flops_forward:,.0f}")
    print(f"FLOPs per sample (train step): {flops_per_sample:,.0f}")
    print(f"Total training steps: {total_steps}")
    print(f"Total TFLOPs for training: {total_flops/1e12:,.0f}")
    return total_flops

def _all_reduce_dict(d, device):
    # wrap in tensor and use reduce to gpu0 tensor
    output_d = {}
    for (key, value) in sorted(d.items()):
        tensor_input = torch.tensor([[value]]).to(device)
        torch.distributed.all_reduce(tensor_input)
        output_d[key] = tensor_input.item()
    return output_d


def run(max_epochs=None,
        device=None,
        batch_size=24,
        max_sequence_length=128,
        random_sequence_length=False,
        epoch_size=None,
        seed=None,
        data_dir='data',
        real_dataset='',
        fake_dataset='xl-1542M-nucleus',
        extra_fake_dataset=None,
        token_dropout=None,
        large=False,
        learning_rate=2e-5,
        weight_decay=0,
        freeze_params=False,
        **kwargs):
    args = locals()
    rank, world_size = setup_distributed()

    if device is None:
        device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'

    print('rank:', rank, 'world_size:', world_size, 'device:', device)

    import torch.distributed as dist
    if distributed() and rank > 0:
        dist.barrier()

    model_name = 'roberta-large' if large else '/home/hice1/wzhou322/scratch/roberta-old-AI-detector'
    tokenization_utils.logger.setLevel('ERROR')
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name).to(device)
    
    trainable_ratio = 1
    if freeze_params:
        for name, param in model.named_parameters():
            if "classifier" in name or "11" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        trainable_ratio = sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters())
        print("Trainable ratio: ", trainable_ratio)

    if rank == 0:
        # summary(model)
        if distributed():
            dist.barrier()

    if world_size > 1:
        model = DistributedDataParallel(model, [rank], output_device=rank, find_unused_parameters=True)

    train_loader, validation_loader = load_datasets(data_dir, real_dataset, fake_dataset, tokenizer, batch_size, max_sequence_length, random_sequence_length, epoch_size, token_dropout, seed, extra_fake_dataset=extra_fake_dataset)
    # train_dir = os.path.join(data_dir, 'train.csv')
    # val_dir = os.path.join(data_dir, 'valid.csv')
    # train_loader, validation_loader = load_turing_datasets(train_dir, val_dir, tokenizer, batch_size,
    #                                                 max_sequence_length, random_sequence_length, epoch_size,
    #                                                 token_dropout, seed)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    epoch_loop = count(1) if max_epochs is None else range(1, max_epochs + 1)

    logdir = os.environ.get("OPENAI_LOGDIR", "logs")
    os.makedirs(logdir, exist_ok=True)

    estimate_total_flops(model, input_size=(max_sequence_length,), N=len(train_loader), batch_size=batch_size, epochs=max_epochs, trainable_ratio=trainable_ratio)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(logdir) if rank == 0 else None
    best_validation_accuracy = 0

    for epoch in epoch_loop:
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
            validation_loader.sampler.set_epoch(epoch)

        train_metrics = train(model, optimizer, device, train_loader, f'Epoch {epoch}')
        torch.save(train_metrics["train_losses"], "train_losses.pt")
        torch.save(train_metrics["train_accs"], "train_accs.pt")
        train_metrics.pop("train_losses")
        train_metrics.pop("train_accs")
        validation_metrics = validate(model, device, validation_loader)

        # combined_metrics = _all_reduce_dict({**validation_metrics, **train_metrics}, device)

        # combined_metrics["train/accuracy"] /= combined_metrics["train/epoch_size"]
        # combined_metrics["train/loss"] /= combined_metrics["train/epoch_size"]
        # combined_metrics["validation/accuracy"] /= combined_metrics["validation/epoch_size"]
        # combined_metrics["validation/loss"] /= combined_metrics["validation/epoch_size"]

        train_metrics["train/accuracy"] /= train_metrics["train/epoch_size"]
        train_metrics["train/loss"] /= train_metrics["train/epoch_size"]
        validation_metrics["validation/accuracy"] /= validation_metrics["validation/epoch_size"]
        validation_metrics["validation/loss"] /= validation_metrics["validation/epoch_size"]

        print(train_metrics)
        print(validation_metrics)

        if rank == 0:
            # for key, value in combined_metrics.items():
            #     writer.add_scalar(key, value, global_step=epoch)

            if validation_metrics["validation/accuracy"] > best_validation_accuracy:
                best_validation_accuracy = validation_metrics["validation/accuracy"]

                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(dict(
                        epoch=epoch,
                        model_state_dict=model_to_save.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        args=args
                    ),
                    os.path.join(logdir, "best-model.pt")
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--max-sequence-length', type=int, default=512)
    parser.add_argument('--random-sequence-length', action='store_true')
    parser.add_argument('--epoch-size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--real-dataset', type=str, default='')
    parser.add_argument('--fake-dataset', type=str, default='xl-1542M-k40')
    parser.add_argument('--extra-fake-dataset', type=str, default=None)
    parser.add_argument('--token-dropout', type=float, default=None)

    parser.add_argument('--large', action='store_true', help='use the roberta-large model instead of roberta-base')
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--freeze_params', action='store_true')
    args = parser.parse_args()

    nproc = int(subprocess.check_output([sys.executable, '-c', "import torch;"
                                         "print(torch.cuda.device_count() if torch.cuda.is_available() else 1)"]))
    if nproc > 1:
        print(f'Launching {nproc} processes ...', file=sys.stderr)

        os.environ["MASTER_ADDR"] = '127.0.0.1'
        os.environ["MASTER_PORT"] = str(29500)
        os.environ['WORLD_SIZE'] = str(nproc)
        os.environ['OMP_NUM_THREAD'] = str(1)
        subprocesses = []

        for i in range(nproc):
            os.environ['RANK'] = str(i)
            os.environ['LOCAL_RANK'] = str(i)
            process = Process(target=run, kwargs=vars(args))
            process.start()
            subprocesses.append(process)

        for process in subprocesses:
            process.join()
    else:
        run(**vars(args))
