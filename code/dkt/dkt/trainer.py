import math
import os

import numpy as np
import torch
from torch import nn
from torch.nn.functional import sigmoid
import wandb
from sklearn.model_selection import KFold

from .criterion import get_criterion
from .dataloader import get_loaders, get_loaders_kfold, data_augmentation, split_data
from .metric import get_metric
from .model import LSTM, LSTMATTN, BERT, LastQuery, Saint, FixupEncoder, GRUATTN, GRUSaint
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .utils import get_logger, logging_conf, get_save_time

# [찬우] data augmentation 추가 data_augmentation
from .dataloader import data_augmentation


logger = get_logger(logger_conf=logging_conf)


def run(args,
        train_data: np.ndarray,
        model: nn.Module):
    
    # augmentation
    if args.augmentation == 'window':
        train_data = data_augmentation(train_data, args)

    # split data
    train_data, valid_data = split_data(data=train_data, ratio=args.ratio)
    
    # wandb
    wandb.init(project="dkt", config=vars(args))
    wandb.run.name = f"Model:{args.model}"
    
    train_loader, valid_loader = get_loaders(args=args, train=train_data, valid=valid_data)

    # For warmup scheduler which uses step interval
    args.total_steps = int(math.ceil(len(train_loader.dataset) / args.batch_size)) * (
        args.n_epochs
    )
    args.warmup_steps = args.total_steps // 10

    optimizer = get_optimizer(model=model, args=args)
    scheduler = get_scheduler(optimizer=optimizer, args=args)

    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):
        logger.info("Start Training: Epoch %s", epoch + 1)

        # TRAIN
        train_auc, train_acc, train_loss = train(train_loader=train_loader,
                                                 model=model, optimizer=optimizer,
                                                 scheduler=scheduler, args=args)

        # VALID
        auc, acc = validate(valid_loader=valid_loader, model=model, args=args)

        wandb.log(dict(epoch=epoch,
                       train_loss_epoch=train_loss,
                       train_auc_epoch=train_auc,
                       train_acc_epoch=train_acc,
                       valid_auc_epoch=auc,
                       valid_acc_epoch=acc))
        
        if auc > best_auc:
            best_auc = auc
            # nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, "module") else model
            save_checkpoint(state={"epoch": epoch + 1,
                                   "state_dict": model_to_save.state_dict()},
                            model_dir=args.model_dir,
                            model_filename="best_model.pt")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                logger.info(
                    "EarlyStopping counter: %s out of %s",
                    early_stopping_counter, args.patience
                )
                break

        # scheduler
        if args.scheduler == "plateau":
            scheduler.step(best_auc)
            
def run_kfold(args,
        train_data: np.ndarray,
        model: nn.Module):
    
    # augmentation
    if args.augmentation == 'window':
        train_data = data_augmentation(train_data, args)
    
    # k-fold
    kfold = KFold(n_splits=args.kfold_splits, shuffle=args.shuffle, random_state=args.seed)
    logger.info(f"K-Fold Split : {args.kfold_splits}")
    
    for k, (train_idx, valid_idx) in enumerate(kfold.split(train_data)):
        # wandb
        wandb.init(project="dkt", config=vars(args))
        wandb.run.name = f"Fold:{k + 1}_Model:{args.model}"
        logger.info(f"-------------- Start Fold : {k + 1} ---------------")
        
        train_loader, valid_loader = get_loaders_kfold(args, train_data, train_idx, valid_idx)

        # For warmup scheduler which uses step interval
        args.total_steps = int(math.ceil(len(train_idx) / args.batch_size)) * (
            args.n_epochs
        )
        args.warmup_steps = args.total_steps // 10

        optimizer = get_optimizer(model=model, args=args)
        scheduler = get_scheduler(optimizer=optimizer, args=args)

        best_auc = -1
        early_stopping_counter = 0
        for epoch in range(args.n_epochs):
            logger.info("Start Training: Epoch %s", epoch + 1)

            # TRAIN
            train_auc, train_acc, train_loss = train(train_loader=train_loader,
                                                    model=model, optimizer=optimizer,
                                                    scheduler=scheduler, args=args)

            # VALID
            auc, acc = validate(valid_loader=valid_loader, model=model, args=args)

            wandb.log(dict(epoch=epoch,
                        train_loss_epoch=train_loss,
                        train_auc_epoch=train_auc,
                        train_acc_epoch=train_acc,
                        valid_auc_epoch=auc,
                        valid_acc_epoch=acc))
            
            if auc > best_auc:
                best_auc = auc
                # nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
                model_to_save = model.module if hasattr(model, "module") else model
                save_checkpoint(state={"epoch": epoch + 1,
                                    "state_dict": model_to_save.state_dict()},
                                model_dir=args.model_dir,
                                model_filename="best_model.pt")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.patience:
                    logger.info(
                        "EarlyStopping counter: %s out of %s",
                        early_stopping_counter, args.patience
                    )
                    break

            # scheduler
            if args.scheduler == "plateau":
                scheduler.step(best_auc)
                
        wandb.finish()


def train(train_loader: torch.utils.data.DataLoader,
          model: nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          args):
    model.train() # 학습을 알린다고 이해하자

    total_preds = []
    total_targets = []  
    losses = []
    for step, batch in enumerate(train_loader): # train_loader는 batch별로 for문 돔
        # args.device의 default가 cpu라서 꼭 gpu라고 줘야함
        batch = {k: v.to(args.device) for k, v in batch.items()} # ex) batch([('feature', tensor(...)), ('label', tensor(...))])
        preds = model(batch) # [건우] '**'를 사용하기 위해 parameter와 argument의 쌍이 같아햐 하는데 lstm에서 paramete는 data하나기 때문에 '**'안씀
        targets = batch["answerCode"]
        
        loss = compute_loss(preds=preds, targets=targets)
        update_params(loss=loss, model=model, optimizer=optimizer,
                      scheduler=scheduler, args=args)

        if step % args.log_steps == 0:
            logger.info("    Training steps: %s Loss: %.4f", step, loss.item())

        # predictions
        preds = torch.sigmoid(preds[:, -1])
        targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())
        losses.append(loss)

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(targets=total_targets, preds=total_preds)
    loss_avg = sum(losses) / len(losses)
    logger.info("TRAIN AUC : %.4f ACC : %.4f", auc, acc)
    return auc, acc, loss_avg


def validate(valid_loader: nn.Module, model: nn.Module, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        preds = model(batch) # [건우] '**'를 사용하기 위해 parameter와 argument의 쌍이 같아햐 하는데 lstm에서 paramete는 data하나기 때문에 '**'안씀
        targets = batch["answerCode"]

        # predictions
        preds = torch.sigmoid(preds[:, -1])
        targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(targets=total_targets, preds=total_preds)
    logger.info("VALID AUC : %.4f ACC : %.4f", auc, acc)
    return auc, acc


def inference(args, test_data: np.ndarray, model: nn.Module) -> None:
    model.eval()
    _, test_loader = get_loaders(args=args, train=None, valid=test_data)

    total_preds = []
    for step, batch in enumerate(test_loader):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        preds = model(batch) # [건우] '**'를 사용하기 위해 parameter와 argument의 쌍이 같아햐 하는데 lstm에서 paramete는 data하나기 때문에 '**'안씀

        # predictions
        preds = torch.sigmoid(preds[:, -1])
        preds = preds.cpu().detach().numpy()
        total_preds += list(preds)

    save_time = get_save_time()
    write_path = os.path.join(args.output_dir, f"submission_{save_time}_{args.model}.csv")
    os.makedirs(name=args.output_dir, exist_ok=True)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))
    logger.info("Successfully saved submission as %s", write_path)


# 모델 선택 
def get_model(args) -> nn.Module:
    
    try:
        model_name = args.model.lower()
        model = { # get은 사용자 입력
            "lstm": LSTM,
            "lstmattn": LSTMATTN,
            "bert": BERT,
            "lastquery": LastQuery,
            "saint": Saint,
            "tfixup": FixupEncoder,
            "gruattn": GRUATTN,
            "grusaint": GRUSaint
        }.get(model_name)(args) # [건우] model.py에서 각 모델의 init으로 args만 받기 때문
    except KeyError:
        logger.warn("No model name %s found", model_name)
    except Exception as e:
        logger.warn("Error while loading %s with args: %s", model_name,args)
        raise e
    return model


def compute_loss(preds: torch.Tensor, targets: torch.Tensor):
    """
    loss계산하고 parameter update
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(pred=preds, target=targets.float())

    # 마지막 시퀀드에 대한 값만 loss 계산
    loss = loss[:, -1]
    loss = torch.mean(loss)
    return loss


def update_params(loss: torch.Tensor,
                  model: nn.Module,
                  optimizer: torch.optim.Optimizer,
                  scheduler: torch.optim.lr_scheduler._LRScheduler,
                  args):
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    if args.scheduler == "linear_warmup":
        scheduler.step()
    optimizer.step()
    optimizer.zero_grad()


def save_checkpoint(state: dict, model_dir: str, model_filename: str) -> None:
    """ Saves checkpoint to a given directory. """
    save_path = os.path.join(model_dir, model_filename)
    logger.info("saving model as %s...", save_path)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(state, save_path)

# 모델 선택 + log저장 -> inference.py에서 사용
def load_model(args):
    model_path = os.path.join(args.model_dir, args.model_name)
    logger.info("Loading Model from: %s", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # load model state
    model.load_state_dict(load_state["state_dict"], strict=True) # strict=False하면 돌아가는데 어떻게 할까?
    logger.info("Successfully loaded model state from: %s", model_path)
    return model
