import json
import re
import random
import os
import logging
from collections import Counter, defaultdict
import numpy as np
from transformers import AutoTokenizer
from shared.const import task_ner_labels

logger = logging.getLogger('root')

class StructuredHeaders(object):
    def __init__(self, input_file):
        self.header_dict = {}
        with open(input_file, 'r') as file:
            for line in file:
                elements = line.strip().split('|')
                # Ensure there are at least two elements to avoid index errors
                if len(elements) >= 2:
                    key = elements[0]
                    value = elements[1]
                    self.header_dict[key.lower()] = value.lower()
        print(f"## Length of Structured Header Dict >> {len(self.header_dict)} ##")

    def map_headers(self, header_name):
        return self.header_dict.get(header_name.strip().lower(), header_name)

def batchify(samples_all, batch_size, model_name):
    """
    Batchify samples with a batch size
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Exclude samples whose subtokens are more than 512
    samples = []
    for i in range(0, len(samples_all)):
        sub_tokens = []
        for token in samples_all[i]['tokens']:
            sub_tokens.extend(tokenizer.tokenize(token))
        if len(sub_tokens) <= 510:
            samples.append(samples_all[i])

    num_samples = len(samples)

    list_samples_batches = []
    
    # if a sentence is too long, make itself a batch to avoid GPU OOM
    to_single_batch = []
    for i in range(0, len(samples)):
        if len(samples[i]['tokens']) > 350:
            to_single_batch.append(i)
    
    for i in to_single_batch:
        logger.info('Single batch sample: %s-%d', samples[i]['doc_key'], samples[i]['sentence_ix'])
        list_samples_batches.append([samples[i]])
    samples = [sample for i, sample in enumerate(samples) if i not in to_single_batch]

    for i in range(0, len(samples), batch_size):
        list_samples_batches.append(samples[i:i+batch_size])

    assert (sum([len(batch) for batch in list_samples_batches]) == num_samples)

    return list_samples_batches

def overlap(s1, s2):
    if s2.start_sent >= s1.start_sent and s2.start_sent <= s1.end_sent:
        return True
    if s2.end_sent >= s1.start_sent and s2.end_sent <= s1.end_sent:
        return True
    return False

def convert_dataset_to_samples(
        args, 
        dataset, 
        neg_sampling_ratio, 
        is_training=False, 
        ner_label2id=None,
):
    """
    Extract sentences and gold entities from a dataset
    """
    max_len = 0
    num_ner = 0
    max_ner = 0
    label_cnt_dict = Counter()

    doc_len = []

    samples = []
    span_len_entity = {label: defaultdict(int) for label in task_ner_labels[args.task]}

    # Collect negative samples by looping the docs
    neg_samples = []
    neg_sample_cnt = 0
    if is_training and neg_sampling_ratio:
        for c, doc in enumerate(dataset):
            for i, sent in enumerate(doc):
                if not len(sent.ner):
                    # Step 1. Header filtering
                    # Include sentences that are from Methods, Design, or Abstract
                    pattern = re.compile(r'(method|design|title|abstract)', re.IGNORECASE)
                    # matches = pattern.findall(' '.join(sent.headers))
                    matches = pattern.findall(sent.headers[0])  # use outermost header only
                    if not matches:
                        sent.negative_sampling = False
                        continue
                    # Step 2. Filter out sentences without alphabet
                    joined_str = ''.join(sent.text)
                    if not bool(re.search('[a-zA-Z]', joined_str)):
                        sent.negative_sampling = False
                        continue

                    neg_samples.append(sent)

        # Indicate whether each neg sample to be included
        random.shuffle(neg_samples)
        for idx in range(len(neg_samples)):
            if idx <= len(neg_samples) * neg_sampling_ratio:
                neg_samples[idx].negative_sampling = True
                neg_sample_cnt += 1
            else:
                neg_samples[idx].negative_sampling = False

    if args.use_section_headers == 'mapped':
        Mapper = StructuredHeaders(
            os.path.join(args.data_dir, "structured_headings.txt")
        )

    # Generate samples with Neg samples
    for c, doc in enumerate(dataset):
        doc_len.append(len(doc))
        for i, sent in enumerate(doc):
            # Only with positive samples: either Train and Dev
            if not neg_sampling_ratio and not len(sent.ner):
                continue
            # Include the part of negative samples for Train
            elif is_training and neg_sampling_ratio:
                if not len(sent.ner) and not sent.negative_sampling:
                    continue
                    
            ner_not_nested = []
            ner_nested = []
            # If you want to remove nested entities
            if args.remove_nested:
                checker = [0]*len(sent.text)
                ner_sorted = sorted(sent.ner, key=lambda x: (x.span.start_sent, -(x.span.end_sent-x.span.start_sent)))
                for ner in ner_sorted:
                    if all(idx == 0 for idx in checker[ner.span.start_sent:ner.span.end_sent]):
                        checker[ner.span.start_sent:ner.span.end_sent] = [1]*(ner.span.end_sent-ner.span.start_sent+1)
                        ner_not_nested.append(ner)
                    else:
                        ner_nested.append(ner)
                sent.ner = ner_not_nested

            num_ner += len(sent.ner)

            sample = {
                'doc_key': doc._doc_key,
                'sentence_ix': sent.sentence_ix,
            }

            if args.context_window != 0 and len(sent.text) > args.context_window:
                logger.info('Long sentence: {} {}'.format(sample, len(sent.text)))
                # print('Exclude:', sample)
                # continue

            sample['tokens'] = sent.text
            sample['sent_length'] = len(sent.text)
            sent_start = 0
            sent_end = len(sample['tokens'])

            if args.use_section_headers is not None:
                if args.use_section_headers == 'innermost':
                    sample['tokens'] = ['Section-header:', sent.headers[-1]] + ['Sentence:'] + sample['tokens'] 
                elif args.use_section_headers == 'outermost':
                    sample['tokens'] = ['Section-header:', sent.headers[0]] + ['Sentence:'] + sample['tokens'] 
                elif args.use_section_headers == 'mapped':
                    sample['tokens'] = ['Section-header:', Mapper.map_headers(sent.headers[0])] + ['Sentence:'] + sample['tokens'] 
                elif args.use_section_headers == 'all':
                    sample['tokens'] = ['Section-header:', '|'.join(sent.headers)] + ['Sentence:'] + sample['tokens']
                sent_start = 3

            # if args.use_section_headers:
            #     sample['tokens'] = [f"<{sent.header_name}>"] + sample['tokens'] + [f"</{sent.header_name}>"]
            #     sent_start = 1

            max_len = max(max_len, len(sent.text))
            max_ner = max(max_ner, len(sent.ner))

            if args.context_window > 0:
                add_left = (args.context_window-len(sample['tokens'])) // 2
                add_right = (args.context_window-len(sample['tokens'])) - add_left
                # add left context
                j = i - 1
                while j >= 0 and add_left > 0:
                    context_to_add = doc[j].text[-add_left:]
                    add_left -= len(context_to_add)
                    j -= 1
                    # if j < 0 or add_left <= 0:
                    #     sample['tokens'] = context_to_add + ["[SEP]"] + sample['tokens']
                    # else:
                    sample['tokens'] = context_to_add + sample['tokens']
                    sent_start += len(context_to_add)
                    sent_end += len(context_to_add)
                # add right context
                j = i + 1
                # add_flag = True
                while j < len(doc) and add_right > 0:
                    context_to_add = doc[j].text[:add_right]
                    # if add_flag:
                        # sample['tokens'] = sample['tokens'] + ["[SEP]"] + context_to_add
                        # add_flag = False
                    # else:
                    sample['tokens'] = sample['tokens'] + context_to_add
                    add_right -= len(context_to_add)
                    j += 1

            if args.use_dataset_name:
                sample['tokens'] = [f'Extract term in <{dataset.dataset_name}> from given text sequence:'] + ['[SEP]'] + sample['tokens']
                sent_start += 2

            sample['sent_start'] = sent_start
            sample['sent_end'] = sent_end
            sample['sent_start_in_doc'] = sent.sentence_start
            
            sent_ner = {}
            for ner in sent.ner:               
                sent_ner[ner.span.span_sent] = ner.label
                label_cnt_dict[ner.label] += 1.0
                span_len = ner.span.end_sent - ner.span.start_sent + 1
                span_len_entity[ner.label][span_len] += 1

            span2id = {}
            sample['spans'] = []
            sample['spans_label'] = []
            sample['sent_header_idx'] = []
            sample['sent_pos_in_doc'] = []

            for j in range(len(sent.text)):
                for k in range(j, min(len(sent.text), j + args.max_span_length_entity)):
                    sample['spans'].append((j+sent_start, k+sent_start, k-j+1))
                    span2id[(j, k)] = len(sample['spans'])-1
                    if (j, k) in sent_ner and (k-j+1) <= args.max_span_length_entity:
                        sample['spans_label'].append(ner_label2id[sent_ner[(j, k)]])
                    else:
                        sample['spans_label'].append(0)
                    sample['sent_header_idx'].append(sent_start-1) # Only valid when using special tokens
                    sample['sent_pos_in_doc'].append(doc.normalized_sentence_positions[i])

            samples.append(sample)
            
    avg_length = sum([len(sample['tokens']) for sample in samples]) / len(samples)
    max_length = max([len(sample['tokens']) for sample in samples])

    for label, cnt in span_len_entity.items():
        cnt = dict(sorted(
            [(token_len, v) for token_len, v in cnt.items()], key=lambda x: x[0]
        ))
        span_len_entity[label] = cnt

    doc_len = sorted(doc_len)

    logger.info('Extracted %d samples from %d documents, with %d NER labels, %.3f avg input length, %d max length'%(len(samples), len(dataset), num_ner, avg_length, max_length))
    logger.info('Max Length: %d, Max NER: %d'%(max_len, max_ner))
    logger.info(f'Span Length of Entities >>> {span_len_entity}')
    logger.info(f'# Negative Samples >>> {neg_sample_cnt} out of {len(neg_samples)}')
    logger.info(f'Min Length: {doc_len[0]}, Medium Length: {doc_len[len(doc_len) // 2]}, Max Length: {doc_len[-1]}')

    return samples, num_ner, label_cnt_dict


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def get_train_fold(data, fold):
    print('Getting train fold %d...'%fold)
    l = int(len(data) * 0.1 * fold)
    r = int(len(data) * 0.1 * (fold+1))
    new_js = []
    new_docs = []
    for i in range(len(data)):
        if i < l or i >= r:
            new_js.append(data.js[i])
            new_docs.append(data.documents[i])
    print('# documents: %d --> %d'%(len(data), len(new_docs)))
    data.js = new_js
    data.documents = new_docs
    return data

def get_test_fold(data, fold):
    print('Getting test fold %d...'%fold)
    l = int(len(data) * 0.1 * fold)
    r = int(len(data) * 0.1 * (fold+1))
    new_js = []
    new_docs = []
    for i in range(len(data)):
        if i >= l and i < r:
            new_js.append(data.js[i])
            new_docs.append(data.documents[i])
    print('# documents: %d --> %d'%(len(data), len(new_docs)))
    data.js = new_js
    data.documents = new_docs
    return data
