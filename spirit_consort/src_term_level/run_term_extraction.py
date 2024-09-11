import json
import argparse
import os
import random
import logging
import tqdm

from shared.data_structures_copied import Dataset
from shared.const import task_ner_labels, get_labelmap
from shared.utils import set_seed, save_model, make_output_dir, generate_analysis_csv

from entity.utils import convert_dataset_to_samples, batchify, NpEncoder
from entity.models_copied import EntityModel

import torch
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup


# Set the CuBLAS workspace configuration for deterministic behavior
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
    )
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    # general arguments: Task, Directories, Train/Eval, and so on.
    parser.add_argument('--task', type=str, default=None, required=True,
                        help=f"Run one of the task in {list(task_ner_labels.keys())}")
    parser.add_argument('--pipeline_task', type=str, default=None, required=True,
                        help=f"Choose what kind of tasks to run: NER, Triplet, and RE")
    parser.add_argument('--do_train', action='store_true', 
                        help="Whether to run training")
    parser.add_argument('--use_train_extra', action='store_true', 
                        help="Whether to add additional training set")
    parser.add_argument('--do_eval', action='store_true', 
                        help="Whether to run evaluation while training")
    parser.add_argument('--do_predict_dev', action='store_true', 
                        help="Whether to run prediction on dev set")
    parser.add_argument('--do_predict_test', action='store_true', 
                        help="Whether to run prediction on test set")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--load_saved_model", action='store_true',
                        help="Whether to use fine-tuned model in entity output dir")
    
    # directory and file arguments
    parser.add_argument('--output_dir', type=str, default=None, required=True,
                        help="Output directory of the experiment outputs")
    parser.add_argument('--entity_output_dir', type=str, default=None,
                        help="Output directory of the entity prediction outputs")
    parser.add_argument('--dev_pred_filename', type=str, default="ent_pred_dev.json", 
                        help="Prediction filename for the dev set")
    parser.add_argument('--test_pred_filename', type=str, default="ent_pred_test.json", 
                        help="Prediction filename for the test set")

    # data-specific arguments:
    parser.add_argument('--data_dir', type=str, default="/data", required=True, 
                        help="Path to the preprocessed dataset")
    parser.add_argument('--train_shuffle', action='store_true',
                        help="Whether to train with randomly shuffled data")
    parser.add_argument('--remove_nested', action='store_true',
                        help="Whether to remove nested entities")
    parser.add_argument('--negative_sampling_train', type=float, default=1.0,
                        help="Set the proportion for neg sampling")
    parser.add_argument('--negative_sampling_dev', type=float, default=0.0,
                        help="Set the proportion for neg sampling. \
                        Only positive samples as default.")
    parser.add_argument('--negative_sampling_test', type=float, default=0.0,
                        help="Set the proportion for neg sampling. \
                        Only positive samples as default.")
    parser.add_argument('--num_segment_doc', type=int, default=30,
                        help="Number of bins for document segmentation") 
    
    # training-specific arguments:
    parser.add_argument('--context_window', type=int, required=True, default=None, 
                        help="Context window size W for the entity model")
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help="Maximum length of tokenized input sequence")
    parser.add_argument('--train_batch_size', type=int, default=32, 
                        help="Batch size during training")
    parser.add_argument('--eval_batch_size', type=int, default=32, 
                        help="Batch size during inference")
    parser.add_argument('--learning_rate', type=float, default=1e-5, 
                        help="Learning rate for the BERT encoder")
    parser.add_argument('--task_learning_rate', type=float, default=5e-4, 
                        help="Learning rate for task-specific parameters, i.e., classification head")
    parser.add_argument('--warmup_proportion', type=float, default=0.1, 
                        help="The ratio of the warmup steps to the total steps")
    parser.add_argument('--num_epoch', type=int, default=200, 
                        help="Number of the training epochs")
    parser.add_argument('--print_loss_step', type=int, default=30, 
                        help="How often logging the loss value during training")
    parser.add_argument('--eval_per_epoch', type=float, default=0.5, 
                        help="How often evaluating the trained model on dev set during training")
    parser.add_argument('--eval_start_epoch', type=int, default=20, 
                        help="Set the start epoch for eval")
    parser.add_argument('--max_patience', type=int, default=4)
    parser.add_argument("--bertadam", action="store_true", 
                        help="If bertadam, then set correct_bias = False")
    
    # model arguments:
    parser.add_argument('--model', type=str, default='bert-base-uncased', 
                        help="Base model name (a huggingface model)")
    parser.add_argument('--bert_model_dir', type=str, default=None, 
                        help="Base model directory")
    parser.add_argument('--max_span_length_entity', type=int, default=10, 
                        help="Entity Spans w/ length up to max_span_length are considered as candidates")
    parser.add_argument('--use_section_headers', type=str, default=None,
                        choices=[None, 'innermost', 'all', 'outermost', 'mapped'],
                        help="How to prefix section headers in front of input sents")
    parser.add_argument('--add_relative_position', action='store_true',
                        help="Whether to use relation position of each sentence")
    parser.add_argument('--add_special_tokens', action='store_true',
                        help="Whether to add special tokens of section headers")
    parser.add_argument('--use_dataset_name', action='store_true',
                        help="Whether to add dataset names for the input sentence")
    
    args = parser.parse_args()
    return args


def output_ner_predictions(model, batches, dataset, ner_id2label, output_file):
    """
    Save the prediction as a json file
    """
    ner_result = {}
    trg_result = {}
    tot_pred_ett = 0
    tot_pred_trg = 0

    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)
        pred_ner = output_dict['pred_ner']
        for sample, preds in zip(batches[i], pred_ner):
            off = sample['sent_start_in_doc'] - sample['sent_start']
            k = sample['doc_key'] + '-' + str(sample['sentence_ix'])
            ner_result[k] = []
            trg_result[k] = []
            for span, pred in zip(sample['spans'], preds):
                # span_id = '%s::%d::(%d,%d)'%(sample['doc_key'], sample['sentence_ix'], span[0]+off, span[1]+off)
                if pred == 0:
                    continue
                # ner_result[k].append([span[0]+off, span[1]+off, ner_id2label[pred]])
                if not ner_id2label[pred].isupper():
                    ner_result[k].append([span[0]+off, span[1]+off, ner_id2label[pred]])
                else:
                    trg_result[k].append([span[0]+off, span[1]+off, ner_id2label[pred]])
            tot_pred_ett += len(ner_result[k])
            tot_pred_trg += len(trg_result[k])

    logger.info('Total pred entities: %d'%tot_pred_ett)
    logger.info('Total pred triggers: %d'%tot_pred_trg)

    js = dataset.js
    for i, doc in enumerate(js):
        doc["predicted_ner"] = []
        for j in range(len(doc["sentences"])):
            k = doc['doc_key'] + '-' + str(j)
            if k in ner_result:
                doc["predicted_ner"].append(ner_result[k])
            else:
                logger.info('%s not in NER results!'%k)
                doc["predicted_ner"].append([])
        js[i] = doc

    logger.info('Output predictions to %s..'%(output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(doc, cls=NpEncoder) for doc in js))

def save_div(a, b):
    if b != 0:
        return a / b 
    else:
        return 0.0
    
def evaluate(model, batches, ent_gold, label_cnt_dict, vocab):
    """
    Evaluate the entity model for entity and trigger extraction
    """
    logger.info('Evaluating...')

    entity_pred_num, entity_gold_num, entity_correct_num = 0.0, ent_gold, 0.0

    inv_vocab = {v:k for k, v in vocab.items()}

    ner_result = {}
    for label, idx in vocab.items():
        ner_result[label] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, 
                        "gold": 0.0, "pred": 0.0, "correct": 0.0}
        ner_result[label]["gold"] = label_cnt_dict[label]

    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)
        batch_pred_ner = output_dict['pred_ner']
        for sample, pred_ner in zip(batches[i], batch_pred_ner):
            for gold, pred in zip(sample['spans_label'], pred_ner):
                if pred != 0:
                    ner_result[inv_vocab[pred]]["pred"] += 1
                    # if not inv_vocab[pred].isupper():
                    entity_pred_num += 1
                    # else:
                        # trigger_pred_num += 1

                    if pred == gold:
                        ner_result[inv_vocab[pred]]["correct"] += 1
                        # if not inv_vocab[pred].isupper():
                        entity_correct_num += 1
                        # else:
                            # trigger_correct_num += 1

    for label in ner_result:
        counts = ner_result[label]
        counts["precision"] = save_div(counts["correct"], counts["pred"])
        counts["recall"] = save_div(counts["correct"], counts["gold"])
        counts["f1"] = save_div(2*counts["precision"]*counts["recall"], counts["precision"]+counts["recall"])

    prec = save_div(entity_correct_num, entity_pred_num)
    rec = save_div(entity_correct_num, entity_gold_num)
    f1 = save_div(2*prec*rec, prec+rec)
    logger.info('Total NER >>> Prec: %.5f, Rec: %.5f, F1: %.5f'%(prec, rec, f1))
    return prec, rec, f1, ner_result


def main() -> None:
    args = get_args()
 
    args.train_data = os.path.join(args.data_dir, 'train.jsonl')
    # args.train_extra = os.path.join(args.data_dir, 'train_extra.jsonl')
    args.dev_data = os.path.join(args.data_dir, 'dev.jsonl')
    args.test_data = os.path.join(args.data_dir, 'test.jsonl')

    set_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    args.n_gpu = torch.cuda.device_count()

    # Assign specific version to generate output folder
    if args.do_train:
        args.entity_output_dir = make_output_dir(
            args.output_dir, task='ner', pipeline_task=args.pipeline_task
        )
        os.makedirs(args.entity_output_dir, exist_ok=True)

    elif args.load_saved_model and args.entity_output_dir:
        args.bert_model_dir = args.entity_output_dir
        args.entity_output_dir = make_output_dir(
            args.output_dir, task='ner', pipeline_task=args.pipeline_task
        )
        os.makedirs(args.entity_output_dir, exist_ok=True)        

    if args.do_train:
        logger.addHandler(
            logging.FileHandler(
                os.path.join(args.entity_output_dir, f"train.log"), 'w'
            )
        )
    else:
        logger.addHandler(
            logging.FileHandler(
                os.path.join(args.entity_output_dir, f"eval.log"), 'w'
            )
        )

    logger.info(f"args: {json.dumps(args.__dict__, indent=2, sort_keys=True)}")
    logger.info("device: {}, n_gpu: {}".format(device, args.n_gpu))

    ner_label2id, ner_id2label = get_labelmap(task_ner_labels[args.task])
    logger.info(f"NER labels: {ner_label2id}")

    num_entity_labels = len(task_ner_labels[args.task]) + 1

    # Dataset processing
    unique_section_headers = set()

    logger.info('## Generating Dev Samples.. ##')
    dev_data = Dataset(
        args.dev_data, num_segment_doc=args.num_segment_doc, dataset_name="CONSORT"
    )
    logger.info(f'## List of unique headers in Dev set: {dev_data.unique_section_headers}')

    if args.do_eval:
        dev_samples, dev_ner, dev_label_dict = convert_dataset_to_samples(
            args, 
            dev_data, 
            neg_sampling_ratio=args.negative_sampling_dev, # only with Positive dev samples to check upper bound
            ner_label2id=ner_label2id
        )
        dev_batches = batchify(dev_samples, args.eval_batch_size, args.model)

    if args.do_train:
        train_data = Dataset(
            args.train_data, num_segment_doc=args.num_segment_doc, dataset_name="CONSORT"
        )
        unique_section_headers.update(train_data.unique_section_headers)
        logger.info(f'## Number of unique headers in Train set: {len(unique_section_headers)}')
        logger.info(f'## List of unique headers in Train set: {train_data.unique_section_headers}')

        if args.use_train_extra:
            # Note that extra training set don't have section headers
            train_extra = Dataset(
                args.train_extra, num_segment_doc=args.num_segment_doc, dataset_name="LINH"
            )
            train_data.documents += train_extra.documents

        if args.do_predict_test:
            logger.info("## Now moving Dev data to Train data... ##")
            train_data.documents.extend(dev_data.documents)
            logger.info(f"## Length of Train data: {len(train_data)} ##")
            unique_section_headers.update(dev_data.unique_section_headers)
            logger.info(f'## Number of unique headers in Train+Dev set: {len(unique_section_headers)}')
        
        logger.info('## Generating Training Samples.. ##')
        train_samples, *_ = convert_dataset_to_samples(
            args, 
            train_data, 
            neg_sampling_ratio=args.negative_sampling_train, 
            is_training=True, 
            ner_label2id=ner_label2id
        )
        train_batches = batchify(train_samples, args.train_batch_size, args.model)
        
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        logger.info('# Vocab of original tokenizer: %d'%len(tokenizer))

        if args.use_dataset_name:
            tokenizer.add_tokens([
                '<CONSORT>', '<LINH>'
            ])
            logger.info('# Vocab after adding markers: %d'%len(tokenizer))

        model = EntityModel(
            args, num_entity_labels=num_entity_labels, tokenizer=tokenizer
        )

        param_optimizer = list(model.bert_model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                if 'bert' in n]},
            {'params': [p for n, p in param_optimizer
                if 'bert' not in n], 'lr': args.task_learning_rate}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, 
            lr=args.learning_rate, 
            correct_bias=not(args.bertadam)
        )
        t_total = len(train_batches) * args.num_epoch
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(t_total*args.warmup_proportion), t_total
        )
        
        tr_loss = 0
        tr_examples = 0
        global_step = 0
        eval_step = len(train_batches) // args.eval_per_epoch

        best_result = 0.0
        current_patience = 0
        do_exit = False

        for epoch in range(1, args.num_epoch+1):
            if do_exit:
                logger.info("===== Do EARLY STOPPING =====")
                break
            if args.train_shuffle:
                random.shuffle(train_batches)

            progress = tqdm.tqdm(
                total=len(train_batches), ncols=150, desc="Epoch: " + str(epoch)
            )
            for step, inputs in enumerate(train_batches):
                output_dict = model.run_batch(inputs, training=True)
                loss = output_dict['ner_loss']
                loss.backward()

                tr_loss += loss.item()
                tr_examples += len(inputs)
                global_step += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if global_step % args.print_loss_step == 0:
                    logger.info(
                        'Epoch=%d, iter=%d, loss=%.5f'%(
                            epoch, step, tr_loss / tr_examples
                        )
                    )
                    tr_loss = 0
                    tr_examples = 0

                if epoch > args.eval_start_epoch and args.do_eval and global_step % eval_step == 0:
                    p, r, f1, total = evaluate(
                        model, dev_batches, dev_ner, dev_label_dict, ner_label2id
                    )
                    if f1 > best_result:
                        best_result = f1
                        logger.info(
                            '!!! Best valid (epoch=%d): prec %.2f | rec %.2f | f1 %.2f' % (
                                epoch, p*100, r*100, f1*100
                            )
                        )
                        logger.info('Saving model to %s...'%(args.entity_output_dir))
                        save_model(model, global_step, "ner", args)
                        current_patience = 0
                    else:
                        if f1 != 0.0:
                            current_patience += 1
                            if current_patience >= args.max_patience:
                                do_exit = True
                progress.update(1)
            progress.close()

        if not args.do_eval:
            logger.info(
                '## Without Validation Set: Saving model to %s... ##'%(args.entity_output_dir)
            )
            save_model(model, global_step, "ner", args)

    if args.do_predict_dev:
        if not args.load_saved_model:
            args.bert_model_dir = args.entity_output_dir
        dev_pred_file = os.path.join(args.entity_output_dir, args.dev_pred_filename)

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        logger.info('# Vocab of original tokenizer: %d'%len(tokenizer))

        model = EntityModel(
            args, num_entity_labels=num_entity_labels, tokenizer=tokenizer, evaluation=True
        )
        
        if not args.do_eval:
            dev_data = Dataset(
                args.dev_data, 
                num_segment_doc=args.num_segment_doc, 
                dataset_name="CONSORT"
            )
            dev_samples, dev_ner, dev_label_dict = convert_dataset_to_samples(
                args, 
                dev_data, 
                neg_sampling_ratio=0.0, # only with Positive dev samples to check upper bound
                ner_label2id=ner_label2id
            )

        dev_batches = batchify(dev_samples, args.eval_batch_size, args.model)
        p, r, f1, total = evaluate(model, dev_batches, dev_ner, dev_label_dict, ner_label2id)
        with open(os.path.join(args.entity_output_dir, f"dev_result_by_labels.json"), 'w', encoding='utf-8') as f_out:
            f_out.write(json.dumps(total, indent=4))
        output_ner_predictions(model, dev_batches, dev_data, ner_id2label, output_file=dev_pred_file)

    if args.do_predict_test:
        if not args.load_saved_model:
            args.bert_model_dir = args.entity_output_dir

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        logger.info('# Vocab of original tokenizer: %d'%len(tokenizer))

        model = EntityModel(
            args, 
            num_entity_labels=num_entity_labels, 
            tokenizer=tokenizer, 
            evaluation=True
        )
        test_data = Dataset(
            args.test_data, 
            num_segment_doc=args.num_segment_doc, 
            dataset_name="CONSORT"
        )
        test_pred_file = os.path.join(
            args.entity_output_dir, args.test_pred_filename
        )
        test_samples, test_ner, test_label_dict = convert_dataset_to_samples(
            args, 
            test_data, 
            neg_sampling_ratio=args.negative_sampling_test, 
            ner_label2id=ner_label2id
        )

        test_batches = batchify(
            test_samples, args.eval_batch_size, args.model
        )
        p, r, f1, total = evaluate(
            model, test_batches, test_ner, test_label_dict, ner_label2id
        )
        with open(os.path.join(args.entity_output_dir, f"test_result_by_labels.json"), 'w', encoding='utf-8') as f_out:
            f_out.write(json.dumps(total, indent=4))
            
        output_ner_predictions(
            model, test_batches, test_data, ner_id2label, output_file=test_pred_file
        )



if __name__ == '__main__':
    main()
