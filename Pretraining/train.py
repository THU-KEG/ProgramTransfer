import os
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import shutil
from tqdm import tqdm
import numpy as np
from Pretraining.utils import *
from Pretraining.model import RelationPT
from Pretraining.data import DataLoader
from transformers import *
from Pretraining.lr_scheduler import get_linear_schedule_with_warmup
from Pretraining.metric import *
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import torch.optim as optim
from IPython import embed


 

def evaluate(args, concept_inputs, relation_inputs, model, relation_eval_loader, concept_eval_loader, device, prefix = ''):
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
    # Eval!
    nb_eval_steps = 0
    func_metric = FunctionAcc(relation_eval_loader.vocab['function2id']['<END>'])
    pbar = ProgressBar(n_total=len(relation_eval_loader), desc="Evaluating")
    correct = 0
    tot = 0
    for step, batch in enumerate(relation_eval_loader):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        # print(batch[4].size())
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'concept_inputs': concept_inputs, 
                'relation_inputs': relation_inputs,
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2],
                'function_ids': batch[3],
                'relation_info': (batch[4], None),
                'concept_info': None
            }
            outputs = model(**inputs)
            pred_functions = outputs['pred_functions'].cpu().tolist()
            pred_relation = outputs['pred_relation']
            gt_relation = batch[5]
            gt_relation = gt_relation.squeeze(-1)
            # print(pred_relation.size(), gt_relation.size(), batch[3].size())
            correct += torch.sum(torch.eq(pred_relation, gt_relation).float())
            # print(correct)
            tot += len(pred_relation)
            gt_functions = batch[3].cpu().tolist()
            for pred, gt in zip(pred_functions, gt_functions):
                func_metric.update(pred, gt)
        nb_eval_steps += 1
        pbar(step)
    logging.info('')   
    acc = func_metric.result()
    logging.info('**** function results %s ****', prefix)
    info = 'acc: {}'.format(acc)
    logging.info(info)
    acc = correct.item() / tot
    logging.info('**** relation results %s ****', prefix)
    logging.info('acc: {}'.format(acc))



    nb_eval_steps = 0
    func_metric = FunctionAcc(concept_eval_loader.vocab['function2id']['<END>'])
    pbar = ProgressBar(n_total=len(concept_eval_loader), desc="Evaluating")
    correct = 0
    tot = 0
    for step, batch in enumerate(concept_eval_loader):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        # print(batch[4].size())
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'concept_inputs': concept_inputs, 
                'relation_inputs': relation_inputs,
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2],
                'function_ids': batch[3],
                'relation_info': None,
                'concept_info': (batch[4], None)
            }
            outputs = model(**inputs)
            pred_functions = outputs['pred_functions'].cpu().tolist()
            pred_relation = outputs['pred_concept']
            gt_relation = batch[5]
            gt_relation = gt_relation.squeeze(-1)
            # print(pred_relation.size(), gt_relation.size(), batch[3].size())
            correct += torch.sum(torch.eq(pred_relation, gt_relation).float())
            # print(correct)
            tot += len(pred_relation)
            gt_functions = batch[3].cpu().tolist()
            for pred, gt in zip(pred_functions, gt_functions):
                func_metric.update(pred, gt)
        nb_eval_steps += 1
        pbar(step)
    logging.info('')   
    acc = func_metric.result()
    logging.info('**** function results %s ****', prefix)
    info = 'acc: {}'.format(acc)
    logging.info(info)
    acc = correct.item() / tot
    logging.info('**** concept results %s ****', prefix)
    logging.info('acc: {}'.format(acc))

        
def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info("Create train_loader and val_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    relation_train_pt = os.path.join(args.input_dir, 'relation', 'train.pt')
    relation_val_pt = os.path.join(args.input_dir, 'relation', 'dev.pt')
    relation_train_loader = DataLoader(vocab_json, relation_train_pt, args.batch_size, training=True)
    relation_val_loader = DataLoader(vocab_json, relation_val_pt, args.batch_size)

    concept_train_pt = os.path.join(args.input_dir, 'concept', 'train.pt')
    concept_val_pt = os.path.join(args.input_dir, 'concept', 'dev.pt')
    concept_train_loader = DataLoader(vocab_json, concept_train_pt, args.batch_size, training=True)
    concept_val_loader = DataLoader(vocab_json, concept_val_pt, args.batch_size)

    with open(os.path.join(args.input_dir, 'relation', 'relation.pt'), 'rb') as f:
        input_ids = pickle.load(f)
        token_type_ids = pickle.load(f)
        attention_mask = pickle.load(f)
        input_ids = torch.LongTensor(input_ids).to(device)
        token_type_ids = torch.LongTensor(token_type_ids).to(device)
        attention_mask = torch.LongTensor(attention_mask).to(device)
    relation_inputs = {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask
    }


    with open(os.path.join(args.input_dir, 'concept', 'concept.pt'), 'rb') as f:
        input_ids = pickle.load(f)
        token_type_ids = pickle.load(f)
        attention_mask = pickle.load(f)
        input_ids = torch.LongTensor(input_ids).to(device)
        token_type_ids = torch.LongTensor(token_type_ids).to(device)
        attention_mask = torch.LongTensor(attention_mask).to(device)
    concept_inputs = {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask
    }

    vocab = relation_train_loader.vocab
    
    logging.info("Create model.........")
    config_class, model_class, tokenizer_class = (BertConfig, RelationPT, BertTokenizer)
    config = config_class.from_pretrained(args.model_name_or_path, num_labels = len(label_list))
    config.update({'vocab': vocab})
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case = False)
    model = model_class.from_pretrained(args.model_name_or_path, config = config)
    model = model.to(device)
    # logging.info(model)


    t_total = (len(relation_train_loader) + len(concept_train_loader)) // args.gradient_accumulation_steps * args.num_train_epochs    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    linear_param_optimizer = list(model.function_embeddings.named_parameters()) + list(model.function_classifier.named_parameters()) + list(model.function_decoder.named_parameters()) + list(model.relation_classifier.named_parameters()) + list(model.concept_classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},
        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(relation_train_loader.dataset))
        logging.info("  Num Epochs = %d", args.num_train_epochs)
        logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d", t_total)

    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(relation_train_loader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(relation_train_loader) // args.gradient_accumulation_steps)
        logging.info("  Continuing training from checkpoint, will skip to saved global_step")
        logging.info("  Continuing training from epoch %d", epochs_trained)
        logging.info("  Continuing training from global step %d", global_step)
        logging.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
    logging.info('Checking...')
    logging.info("===================Dev==================")
    # evaluate(args, concept_inputs, relation_inputs, model, relation_val_loader, concept_val_loader, device)
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    for _ in range(int(args.num_train_epochs)):
        logging.info('relation training begins')
        pbar = ProgressBar(n_total=len(relation_train_loader), desc='Training')
        for step, batch in enumerate(relation_train_loader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'concept_inputs': concept_inputs,
                'relation_inputs': relation_inputs,
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2],
                'function_ids': batch[3],
                'relation_info': (batch[4], batch[5]),
                'concept_info': None
            }
            outputs = model(**inputs)
            loss = args.func * outputs['function_loss'] + args.rel * outputs['relation_loss']
            loss.backward()
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1


        logging.info('concept training begins')
        pbar = ProgressBar(n_total=len(concept_train_loader), desc='Training')
        for step, batch in enumerate(concept_train_loader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'concept_inputs': concept_inputs,
                'relation_inputs': relation_inputs,
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2],
                'function_ids': batch[3],
                'relation_info': None,
                'concept_info': (batch[4], batch[5])
            }
            outputs = model(**inputs)
            loss = args.func * outputs['function_loss'] + args.con * outputs['concept_loss']
            loss.backward()
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

            # break
        # Save model checkpoint
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logging.info("Saving model checkpoint to %s", output_dir)
        tokenizer.save_vocabulary(output_dir)
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logging.info("Saving optimizer and scheduler states to %s", output_dir)
        logging.info("\n")
        if 'cuda' in str(device):
            torch.cuda.empty_cache()
        evaluate(args, concept_inputs, relation_inputs, model, relation_val_loader, concept_val_loader, device)

    return global_step, tr_loss / global_step


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)

    parser.add_argument('--save_dir', required=True, help='path to save checkpoints and logs')
    # parser.add_argument('--glove_pt', default='/data/csl/resources/word2vec/glove.840B.300d.py36.pt')
    parser.add_argument('--model_name_or_path', default = '/data/csl/resources/Bert/bert-base-cased')
    # parser.add_argument('--ckpt')

    # training parameters
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--learning_rate', default=3e-5, type = float)
    parser.add_argument('--crf_learning_rate', default=1e-3, type = float)
    parser.add_argument('--num_train_epochs', default=25, type = int)
    parser.add_argument('--save_steps', default=448, type = int)
    parser.add_argument('--logging_steps', default=448, type = int)
    parser.add_argument('--warmup_proportion', default=0.1, type = float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    # model hyperparameters
    parser.add_argument('--dim_word', default=300, type=int)
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--alpha', default = 1, type = float)
    parser.add_argument('--beta', default = 1e-1, type = float)
    parser.add_argument('--func', default = 1, type = float)
    parser.add_argument('--rel', default = 1, type = float)
    parser.add_argument('--con', default = 1, type = float)


    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, '{}.log'.format(time_)))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k+':'+str(v))

    seed_everything(666)

    train(args)


if __name__ == '__main__':
    main()
