from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import csv
import os
import logging
import argparse
import random
import h5py
from tqdm import tqdm, trange
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import math
from apex import amp
import json

from model.tokenization import BertTokenizer
from model.modeling import BertForMaskedLM, BertConfig
from model.optimization import BertAdam, BertAdam_FP16
from model.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from model.schedulers import LinearWarmUpScheduler

from apex.optimizers import FusedAdam
from apex.parallel import DistributedDataParallel as DDP

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class pretraining_dataset(Dataset):

    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        self.input_ids = np.asarray(f["input_ids"][:]).astype(np.int64)#[num_instances x max_seq_length])
        self.input_masks = np.asarray(f["input_mask"][:]).astype(np.int64) #[num_instances x max_seq_length]
        self.segment_ids = np.asarray(f["segment_ids"][:]).astype(np.int64) #[num_instances x max_seq_length]
        self.masked_lm_positions = np.asarray(f["masked_lm_positions"][:]).astype(np.int64) #[num_instances x max_pred_length]
        self.masked_lm_ids= np.asarray(f["masked_lm_ids"][:]).astype(np.int64) #[num_instances x max_pred_length]
        self.next_sentence_labels = np.asarray(f["next_sentence_labels"][:]).astype(np.int64) # [num_instances]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.input_ids)

    def __getitem__(self, index):
        
        input_ids= torch.from_numpy(self.input_ids[index]) # [max_seq_length]
        input_mask = torch.from_numpy(self.input_masks[index]) #[max_seq_length]
        segment_ids = torch.from_numpy(self.segment_ids[index])# [max_seq_length]
        masked_lm_positions = torch.from_numpy(self.masked_lm_positions[index]) #[max_pred_length]
        masked_lm_ids = torch.from_numpy(self.masked_lm_ids[index]) #[max_pred_length]
        next_sentence_labels = torch.from_numpy(np.asarray(self.next_sentence_labels[index])) #[1]
         
        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        if len((masked_lm_positions == 0).nonzero()) != 0:
          index = (masked_lm_positions == 0).nonzero()[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels]

def main():    

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain .hdf5 files  for the task.")

    parser.add_argument("--bert_model", default="bert-large-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        help="The BERT model config")
    parser.add_argument("--ckpt", 
                        default="",
                        type=str)
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_predictions_per_seq",
                        default=80,
                        type=int,
                        help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default=1000,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0.0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--log_freq',
                        type=float, default=500,
                        help='frequency of logging loss.')
    parser.add_argument('--checkpoint_activations',
                        default=False,
                        action='store_true',
                        help="Whether to use gradient checkpointing")
    parser.add_argument("--resume_from_checkpoint",
                        default=False,
                        action='store_true',
                        help="Whether to resume training from checkpoint.")
    parser.add_argument('--resume_step',
                        type=int,
                        default=-1,
                        help="Step to resume training from.")
    parser.add_argument('--num_steps_per_checkpoint',
                        type=int,
                        default=2000,
                        help="Number of update steps until a model checkpoint is saved to disk.")
    parser.add_argument('--dev_data_file',
                        type=str,
                        default="dev/dev.hdf5")
    parser.add_argument('--dev_batch_size',
                        type=int,
                        default=16)
    parser.add_argument("--save_total_limit", type=int, default=10)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    min_dev_loss = 1000000
    best_step = 0

    assert(torch.cuda.is_available())
    print(args.local_rank)
    if args.local_rank == -1:
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
                            args.gradient_accumulation_steps, args.train_batch_size))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps



    if not args.resume_from_checkpoint and os.path.exists(args.output_dir) and (os.listdir(args.output_dir) and os.listdir(args.output_dir)!=['logfile.txt']):
        logger.warning("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        # raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    if not args.resume_from_checkpoint:
        os.makedirs(args.output_dir, exist_ok=True)

    # Prepare model
    if args.config_file:
        config = BertConfig.from_json_file(args.config_file)

    if args.bert_model:
        model = BertForMaskedLM.from_pretrained(args.bert_model)
    else:
        model = BertForMaskedLM(config)

    print(args.ckpt)
    if args.ckpt:
        print("load from", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location='cpu')
        if model in ckpt:
            ckpt = ckpt['model']
        model.load_state_dict(ckpt, strict=False)

    pretrained_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), pretrained_model_file)

    if not args.resume_from_checkpoint:
        global_step = 0
    else:
        if args.resume_step == -1:
            model_names = [f for f in os.listdir(args.output_dir) if f.endswith(".pt")]
            args.resume_step = max([int(x.split('.pt')[0].split('_')[1].strip()) for x in model_names])
        
        global_step = args.resume_step

        checkpoint = torch.load(os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step)), map_location="cpu")
        model.load_state_dict(checkpoint['model'], strict=False)

        print("resume step from ", args.resume_step)

    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.fp16:
        optimizer = FusedAdam(optimizer_grouped_parameters,
                                    lr=args.learning_rate,
                                    bias_correction=False,
                                    weight_decay=0.01)

        if args.loss_scale == 0:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False, loss_scale="dynamic")
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False, loss_scale=args.loss_scale)

        scheduler = LinearWarmUpScheduler(optimizer, warmup=args.warmup_proportion, total_steps=args.max_steps)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                                lr=args.learning_rate,
                                warmup=args.warmup_proportion,
                                t_total=args.max_steps)
        
    if args.resume_from_checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
               
    if args.local_rank != -1:
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    files.sort()

    num_files = len(files)

    logger.info("***** Loading Dev Data *****")
    dev_data = pretraining_dataset(input_file=os.path.join(args.input_dir, args.dev_data_file), max_pred_length=args.max_predictions_per_seq)
    if args.local_rank == -1:
        dev_sampler = RandomSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.dev_batch_size * n_gpu, num_workers=4, pin_memory=True)
    else:
        dev_sampler = DistributedSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.dev_batch_size, num_workers=4, pin_memory=True)

    logger.info("***** Running training *****")
    logger.info("  Batch size = {}".format(args.train_batch_size))
    logger.info("  LR = {}".format(args.learning_rate))
    
    model.train()
    logger.info(" Training. . .")

    most_recent_ckpts_paths = []

    tr_loss = 0.0 # total added training loss
    average_loss = 0.0 # averaged loss every args.log_freq steps
    epoch = 0
    training_steps = 0
    while True:
        if not args.resume_from_checkpoint:
            random.shuffle(files)
            f_start_id = 0
        else:
            f_start_id = checkpoint['files'][0]
            files = checkpoint['files'][1:]
            args.resume_from_checkpoint = False
        for f_id in range(f_start_id, len(files)):
            data_file = files[f_id]
            logger.info("file no {} file {}".format(f_id, data_file))
            train_data = pretraining_dataset(input_file=data_file, max_pred_length=args.max_predictions_per_seq)

            if args.local_rank == -1:
                train_sampler = RandomSampler(train_data)
                train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size * n_gpu, num_workers=4, pin_memory=True)
            else:
                train_sampler = DistributedSampler(train_data)
                train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4, pin_memory=True)

            for step, batch in enumerate(tqdm(train_dataloader, desc="File Iteration")):
                model.train()
                training_steps += 1
                batch = [t.to(device) for t in batch]
                input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch#\
                loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, masked_lm_labels=masked_lm_labels,checkpoint_activations=args.checkpoint_activations)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                tr_loss += loss.item()
                average_loss += loss.item()

                if training_steps % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                if training_steps == 1 * args.gradient_accumulation_steps:
                    logger.info("Global Step:{} Average Loss = {} Step Loss = {} LR {}".format(global_step, average_loss, 
                                                                                loss.item(), optimizer.param_groups[0]['lr']))

                if training_steps % (args.log_freq * args.gradient_accumulation_steps) == 0:
                    logger.info("Global Step:{} Average Loss = {} Step Loss = {} LR {}".format(global_step,  average_loss / args.log_freq, 
                                                                                loss.item(), optimizer.param_groups[0]['lr']))
                    average_loss = 0

                if training_steps % (args.num_steps_per_checkpoint * args.gradient_accumulation_steps) == 0:
                    logger.info("Begin Eval")
                    model.eval()
                    with torch.no_grad():
                        dev_global_step = 0
                        dev_final_loss = 0.0
                        for dev_step, dev_batch in enumerate(tqdm(dev_dataloader, desc="Evaluating")):
                            batch = [t.to(device) for t in batch]
                            dev_input_ids, dev_segment_ids, dev_input_mask, dev_masked_lm_labels, dev_next_sentence_labels = batch
                            loss = model(input_ids=dev_input_ids, token_type_ids=dev_segment_ids, attention_mask=dev_input_mask, masked_lm_labels=dev_masked_lm_labels)
                            dev_final_loss += loss
                            dev_global_step += 1
                        dev_final_loss /= dev_global_step
                        if (torch.distributed.is_initialized()):
                            dev_final_loss /= torch.distributed.get_world_size()
                            torch.distributed.all_reduce(dev_final_loss)
                        logger.info("Dev Loss: {}".format(dev_final_loss.item()))
                        if dev_final_loss < min_dev_loss:
                            best_step = global_step
                            min_dev_loss = dev_final_loss
                            if (not torch.distributed.is_initialized() or (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0)):
                                logger.info("** ** * Saving best dev loss model ** ** * at step {}".format(best_step))
                                dev_model_to_save = model.module if hasattr(model, 'module') else model
                                output_save_file = os.path.join(args.output_dir, "best_ckpt.pt")
                                torch.save({'model' : dev_model_to_save.state_dict(),
                                            'optimizer' : optimizer.state_dict(),
                                            'files' : [f_id] + files}, output_save_file)

                    if (not torch.distributed.is_initialized() or (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0)):
                        # Save a trained model
                        logger.info("** ** * Saving fine - tuned model ** ** * ")
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        output_save_file = os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step))
                       
                        torch.save({'model' : model_to_save.state_dict(), 
                                'optimizer' : optimizer.state_dict(), 
                                'files' : [f_id] + files}, output_save_file)

                        most_recent_ckpts_paths.append(output_save_file)
                        if len(most_recent_ckpts_paths) > args.save_total_limit:
                            ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                            os.remove(ckpt_to_be_removed)

                    if global_step >= args.max_steps:
                        tr_loss = tr_loss * args.gradient_accumulation_steps / training_steps
                        if (torch.distributed.is_initialized()):
                            tr_loss /= torch.distributed.get_world_size()
                            print(tr_loss)
                            torch.distributed.all_reduce(torch.tensor(tr_loss).cuda())
                        logger.info("Total Steps:{} Final Loss = {}".format(training_steps, tr_loss))

                        with open(os.path.join(args.output_dir, "valid_results.txt"), "w") as f:
                            f.write("Min dev loss: {}\nBest step: {}\n".format(min_dev_loss, best_step))

                        return
            del train_dataloader
            del train_sampler
            del train_data       

            torch.cuda.empty_cache()
        epoch += 1

if __name__ == "__main__":
    main()
