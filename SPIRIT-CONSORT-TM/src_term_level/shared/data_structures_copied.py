"""
This code is based on DYGIE++'s codebase
"""
import json
import copy
import re
import os
from collections import Counter
import numpy as np
# import pandas as pd
from shared.const import task_ner_labels

def fields_to_batches(d, keys_to_ignore=[]):
    keys = [key for key in d.keys() if key not in keys_to_ignore]
    lengths = [len(d[k]) for k in keys]
    assert len(set(lengths)) == 1, print(d['doc_key'], lengths)
    length = lengths[0]
    res = [{k: d[k][i] for k in keys} for i in range(length)]
    return res

def get_sentence_of_span(span, sentence_starts, doc_tokens):
    """
    Return the index of the sentence that the span is part of.
    """
    # Inclusive sentence ends
    sentence_ends = [x - 1 for x in sentence_starts[1:]] + [doc_tokens - 1]
    in_between = [span[0] >= start and span[1] <= end
                  for start, end in zip(sentence_starts, sentence_ends)]
    assert sum(in_between) == 1
    the_sentence = in_between.index(True)
    return the_sentence

def find_overlapped_spans(span1, span2):
    if span1[1] >= span2[0] and span1[0] <= span2[1]:
        return True
    return False

class Dataset:
    def __init__(
            self, 
            json_file, 
            pred_file=None, 
            doc_range=None, 
            num_segment_doc=0,
            dataset_name=None
    ):
        self.js = self._read(json_file, pred_file)
        if doc_range is not None:
            self.js = self.js[doc_range[0]:doc_range[1]]
        self.documents = [Document(js, num_segment_doc) for js in self.js]
        
        self.unique_section_headers = set()
        for doc in self.documents:
            self.unique_section_headers.update(set(
                [header for headers in doc.sentence_headers for header in headers]
            ))

        self.dataset_name = dataset_name

    def update_from_js(self, js):
        self.js = js
        self.documents = [Document(js) for js in self.js]

    def _read(self, json_file, pred_file=None):
        gold_docs = [json.loads(line) for line in open(json_file)]
        if pred_file is None:
            return gold_docs

        pred_docs = [json.loads(line) for line in open(pred_file)]
        merged_docs = []
        for gold, pred in zip(gold_docs, pred_docs):
            assert gold["doc_key"] == pred["doc_key"]
            assert gold["sentences"] == pred["sentences"]
            merged = copy.deepcopy(gold)
            for k, v in pred.items():
                if "predicted" in k:
                    merged[k] = v
            merged_docs.append(merged)

        return merged_docs

    def __getitem__(self, ix):
        return self.documents[ix]

    def __len__(self):
        return len(self.documents)


class Document:
    def __init__(self, js, num_segment_doc):
        self._doc_key = js["doc_key"]
        entries = fields_to_batches(js, keys_to_ignore=[
            "doc_key", "clusters", "predicted_clusters", "section_starts"
        ])
        sentence_lengths = [len(entry["sentences"]) for entry in entries]
        sentence_starts = np.cumsum(sentence_lengths)
        sentence_starts = np.roll(sentence_starts, 1)
        sentence_starts[0] = 0
        self.sentence_starts = sentence_starts

        pattern1 = r'(?:\d+\.)+'
        # pattern1 = r'^\d+\.\s'
        # pattern2 = r'[^a-zA-Z\s]'

        self.sentence_headers = []
        for entry in entries:
            headers = []
            for header in entry['section_headers']:
                header = header.lower().lstrip('#').strip()
                header = re.sub(pattern1, '', header)
                header = re.sub(r'\d', '', header)
                header = header.split(':')[0].strip()
                header = re.sub(r'(_\d+)', '', header)
                header = re.sub(r'\s*\d+', '', header)
                header = 'appendix' if header.startswith('appendix') else header
                header = header.rstrip('.').replace('_', '')
                header = 'no-header' if not header else header
                headers.append(header.strip())
            self.sentence_headers.append(headers)

        self.sentences = [Sentence(entry, sentence_start, sentence_ix, headers, len(sentence_lengths))
                          for sentence_ix, (entry, sentence_start, headers)
                          in enumerate(zip(entries, sentence_starts, self.sentence_headers))]
        sentence_indices = [i for i in range(len(self.sentences))]

        if num_segment_doc:
            normalized_sentence_positions = np.linspace(sentence_indices[0], sentence_indices[-1], num_segment_doc)
            self.normalized_sentence_positions = np.digitize(
                np.arange(sentence_indices[0], sentence_indices[-1] + 1), normalized_sentence_positions
            )
            # print(self.normalized_sentence_positions)
            assert len(self.sentences) == len(self.normalized_sentence_positions)

    def __repr__(self):
        return "\n".join([str(i) + ": " + " ".join(sent.text) for i, sent in enumerate(self.sentences)])

    def __getitem__(self, ix):
        return self.sentences[ix]

    def __len__(self):
        return len(self.sentences)

    def print_plaintext(self):
        for sent in self:
            print(" ".join(sent.text))


    def find_cluster(self, entity, predicted=True):
        """
        Search through erence clusters and return the one containing the query entity, if it's
        part of a cluster. If we don't find a match, return None.
        """
        clusters = self.predicted_clusters if predicted else self.clusters
        for clust in clusters:
            for entry in clust:
                if entry.span == entity.span:
                    return clust

        return None

    @property
    def n_tokens(self):
        return sum([len(sent) for sent in self.sentences])


class Sentence:
    def __init__(self, entry, sentence_start, sentence_ix, headers, num_sents):
        self.sentence_start = sentence_start
        self.text = entry["sentences"]
        self.sentence_ix = sentence_ix
        self.normalized_sentence_ix = sentence_ix / num_sents
        self.headers = headers

        # Gold
        if "ner_flavor" in entry:
            self.ner = [NER(this_ner, self.text, sentence_start, flavor=this_flavor)
                        for this_ner, this_flavor in zip(entry["ner"], entry["ner_flavor"])]
        elif "ner" in entry:
            self.ner = [NER(this_ner, self.text, sentence_start)
                        for this_ner in entry["ner"]]
        if "triggers" in entry:
            self.triggers = [NER(this_trg, self.text, sentence_start) 
                             for this_trg in entry["triggers"]]
        if "relations" in entry:
            self.relations = [Relation(this_relation, self.text, sentence_start) for
                              this_relation in entry["relations"]]
        if "triplets" in entry:
            self.triplets = [Triplet(this_triplet, self.text, sentence_start) for 
                                          this_triplet in entry["triplets"]]
        # if "events" in entry:
            # self.events = Events(entry["events"], self.text, sentence_start)

        # Predicted
        if "predicted_ner" in entry:
            self.predicted_ner = [NER(this_ner, self.text, sentence_start, flavor=None) for
                                  this_ner in entry["predicted_ner"]]
        if "predicted_triggers" in entry:
            self.predicted_triggers = [NER(this_trg, self.text, sentence_start, flavor=None) for
                                  this_trg in entry["predicted_triggers"]]
        if "predicted_relations" in entry:
            self.predicted_relations = [Relation(this_relation, self.text, sentence_start) for
                                        this_relation in entry["predicted_relations"]]
        if "predicted_triplets" in entry:
            self.predicted_triplets = [Triplet(this_pair, self.text, sentence_start) for 
                                          this_pair in entry["predicted_triplets"]]
        # if "predicted_events" in entry:
            # self.predicted_events = Events(entry["predicted_events"], self.text, sentence_start)

        # Top spans
        if "top_spans" in entry:
            self.top_spans = [NER(this_ner, self.text, sentence_start, flavor=None) for
                                this_ner in entry["top_spans"]]

    def __repr__(self):
        the_text = " ".join(self.text)
        the_lengths = np.array([len(x) for x in self.text])
        tok_ixs = ""
        for i, offset in enumerate(the_lengths):
            true_offset = offset if i < 10 else offset - 1
            tok_ixs += str(i)
            tok_ixs += " " * true_offset

        return the_text + "\n" + tok_ixs

    def __len__(self):
        return len(self.text)

    def get_flavor(self, argument):
        the_ner = [x for x in self.ner if x.span == argument.span]
        if len(the_ner) > 1:
            print("Weird")
        if the_ner:
            the_flavor = the_ner[0].flavor
        else:
            the_flavor = None
        return the_flavor


class Span:
    def __init__(self, start, end, text, sentence_start):
        self.start_doc = start
        self.end_doc = end
        self.span_doc = (self.start_doc, self.end_doc)
        self.start_sent = start - sentence_start
        self.end_sent = end - sentence_start
        self.span_sent = (self.start_sent, self.end_sent)
        self.text = text[self.start_sent:self.end_sent + 1]

    def __repr__(self):
        return str((self.start_sent, self.end_sent, self.text))

    def __eq__(self, other):
        return (self.span_doc == other.span_doc and
                self.span_sent == other.span_sent and
                self.text == other.text)

    def __hash__(self):
        tup = self.span_doc + self.span_sent + (" ".join(self.text),)
        return hash(tup)


class Token:
    def __init__(self, ix, text, sentence_start):
        self.ix_doc = ix
        self.ix_sent = ix - sentence_start
        self.text = text[self.ix_sent]

    def __repr__(self):
        return str((self.ix_sent, self.text))

class Argument:
    def __init__(self, span, role, event_type):
        self.span = span
        self.role = role
        self.event_type = event_type

    def __repr__(self):
        return self.span.__repr__()[:-1] + ", " + self.event_type + ", " + self.role + ")"

    def __eq__(self, other):
        return (self.span == other.span and
                self.role == other.role and
                self.event_type == other.event_type)

    def __hash__(self):
        return self.span.__hash__() + hash((self.role, self.event_type))


class NER:
    def __init__(self, ner, text, sentence_start, flavor=None):
        self.span = Span(ner[0], ner[1], text, sentence_start)
        self.label = ner[2]
        self.flavor = flavor

    def __repr__(self):
        return self.span.__repr__() + ": " + self.label

    def __eq__(self, other):
        return (self.span == other.span and
                self.label == other.label and
                self.flavor == other.flavor)
    
class Triplet:
    def __init__(self, triplet, text, sentence_start):
        start1, end1 = triplet[0], triplet[1]
        start2, end2 = triplet[2], triplet[3]
        start_trg, end_trg = triplet[4], triplet[5]
        span1 = Span(start1, end1, text, sentence_start)
        span2 = Span(start2, end2, text, sentence_start)
        span_trg = Span(start_trg, end_trg, text, sentence_start)
        self.triplet = (span1, span2, span_trg)

        if len(triplet) > 6:
            self.label = triplet[6]

    def __repr__(self):
        return self.triplet[0].__repr__() + ";" + self.triplet[1].__repr__() + ";" + self.triplet[2].__repr__()

    def __eq__(self, other):
        return (self.triplet == other.triplet) and (self.label == other.label)


class Relation:
    def __init__(self, relation, text, sentence_start):
        start1, end1 = relation[0], relation[1]
        start2, end2 = relation[2], relation[3]
        label = relation[4]
        span1 = Span(start1, end1, text, sentence_start)
        span2 = Span(start2, end2, text, sentence_start)
        self.pair = (span1, span2)
        self.flipped_pair = (span2, span1)
        self.label = label
        if len(relation) == 5:
            self.certainty = ""  # Placeholder
        else:
            self.certainty = relation[5]

    def __repr__(self):
        return self.pair[0].__repr__() + ", " + self.pair[1].__repr__() + ": " + self.label

    def __eq__(self, other):
        return (self.pair == other.pair) and (self.label == other.label)
    
    def flipped_match(self, other, mode='strict'):
        if mode == 'strict':
            return (self.flipped_pair == other.pair) and (self.label == other.label)
        if mode == 'relaxed':
            return self.find_overlap(other, mode='flipped')
        
    def find_overlap(self, other, mode='ordered'):
        if mode == 'ordered':
            if find_overlapped_spans(self.pair[0].span_sent, other.pair[0].span_sent) and \
            find_overlapped_spans(self.pair[1].span_sent, other.pair[1].span_sent):
                return True            
        else:
            if find_overlapped_spans(self.pair[1].span_sent, other.pair[0].span_sent) and \
            find_overlapped_spans(self.pair[0].span_sent, other.pair[1].span_sent):
                return True
        return False


####################################################################################################

# Code to do evaluation of predictions for a loaded dataset.

def safe_div(num, denom):
    if denom > 0:
        return round(num/denom, 4)
    else:
        return 0

def compute_f1(predicted, gold, matched):
    # F1 score.
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    result = dict(
        precision=precision, 
        recall=recall, 
        f1=f1,
        n_gold=gold,
        n_pred=predicted,
        n_correct=matched
    )
    return result

def evaluate_sent(sent, counts, ner_result_by_class, errors, confusion, label2idx):

    correct_ner = set()
    correct_ner_partial = set()

    # # Remove multi-label cases (temporary)
    # multilabel_spans = set()
    # seen_spans = set()
    # for ner in sent.ner:
    #     if ner.span.span_sent not in seen_spans:
    #         seen_spans.add(ner.span.span_sent)
    #     else:
    #         multilabel_spans.add(ner.span.span_sent)
    # sent_ner_filtered = [ner for ner in sent.ner if ner.span.span_sent not in multilabel_spans]
    # sent.ner = sent_ner_filtered

    for ner in sent.ner:
        ner_result_by_class[ner.label]['gold'] += 1.0
    counts["ner_gold"] += len(sent.ner)

    # NER score for entities
    gold_ners = {
        (ner.span.span_sent): ner.label for ner in sent.ner
    }

    # Choose longer mention if nested mention has the same predicted label
    predicted_ner_sorted = sorted(
        [ner for ner in sent.predicted_ner],
        key=lambda x: x.span.span_sent[1]-x.span.span_sent[0],
        reverse=True
    )

    predicted_ner_processed = []
    checker = [[] for _ in range(len(sent.text))]
    for ner in predicted_ner_sorted:
        dup_flag = False
        for lst in checker[ner.span.span_sent[0]:ner.span.span_sent[1]]:
            if ner.label in lst:
                dup_flag = True
                break
        if not dup_flag:
            predicted_ner_processed.append(ner)
            for lst in checker[ner.span.span_sent[0]:ner.span.span_sent[1]+1]:
                lst.append(ner.label)

    sent.predicted_ner = predicted_ner_processed

    for ner in sent.predicted_ner:
        ner_result_by_class[ner.label]['pred'] += 1.0
        ner_result_by_class[ner.label]['pred_relaxed'] += 1.0
    counts["ner_predicted"] += len(sent.predicted_ner)
    counts["ner_relaxed_predicted"] += len(sent.predicted_ner)

    pred_ners = {
        (ner.span.span_sent): ner.label for ner in sent.predicted_ner
    }

    # Evaluate
    partial_match_ner = []
    for prediction in sent.predicted_ner:

        flag_ner_em, flag_ner_pm = False, False

        # Strict Matching
        for actual in sent.ner:
            if prediction == actual:
                counts["ner_matched"] += 1
                ner_result_by_class[prediction.label]['correct'] += 1.0
                correct_ner.add(prediction.span)
                flag_ner_em = True
                errors['tp-ner-em'].append((
                    prediction.span.start_doc, prediction.span.end_doc, prediction.label
                ))
                confusion[label2idx[prediction.label]][label2idx[actual.label]] += 1
                break

        # Partial Matching
        # Remove duplicated prediction -> 1:1 mapping for every gold standard
        # But consider nested gold cases whose concepts are same
        is_duplicated = False
        for actual in sent.ner:
            if (actual.label == prediction.label) and \
            find_overlapped_spans(actual.span.span_sent, prediction.span.span_sent):
                if actual not in partial_match_ner:
                    partial_match_ner.append(actual)
                    counts["ner_partial_matched"] += 1
                    ner_result_by_class[prediction.label]['correct_relaxed'] += 1.0
                    correct_ner_partial.add(prediction.span)  # Add span of prediction for RE evaluation
                    is_duplicated = False
                    flag_ner_pm = True
                    if not flag_ner_em:
                        errors['tp-ner-relaxed'].append((
                            actual.span.start_doc, actual.span.end_doc, actual.label
                        ))
                    break
                else:
                    is_duplicated = True
                    correct_ner_partial.add(prediction.span)
        if is_duplicated:
            counts["ner_relaxed_predicted"] -= 1
            ner_result_by_class[prediction.label]['pred_relaxed'] -= 1.0

        if not flag_ner_em:
            if not flag_ner_pm:
                errors['fp-ner-relaxed'].append((
                    prediction.span.start_doc, prediction.span.end_doc, prediction.label
                ))
            else:
                errors['fp-ner-em'].append((
                    prediction.span.start_doc, prediction.span.end_doc, prediction.label
                ))
            if prediction.span.span_sent in gold_ners:
                confusion[label2idx[prediction.label]][label2idx[gold_ners[prediction.span.span_sent]]] += 1
            else:
                confusion[label2idx[prediction.label]][label2idx["NULL"]] += 1

    for actual in sent.ner:
        decomposed = (
            actual.span.start_doc, actual.span.end_doc, actual.label
        )
        if decomposed not in errors['tp-ner-em']:
            if decomposed not in errors['tp-ner-relaxed']:
                errors['fn-ner-relaxed'].append(decomposed)
            else:
                errors['fn-ner-em'].append(decomposed)
            if not actual.span.span_sent in pred_ners:
                confusion[label2idx["NULL"]][label2idx[actual.label]] += 1

    return counts, ner_result_by_class, errors, confusion
    

def evaluate_predictions(dataset, output_dir, task, dataset_name):

    counts = Counter()
    ner_labels = task_ner_labels[dataset_name]

    ner_result_by_class = {}
    for label in ner_labels:
        ner_result_by_class[label] = {
            "gold": 0.0, 
            "pred": 0.0, 
            "pred_relaxed": 0.0, 
            "correct": 0.0,
            "correct_relaxed": 0.0
        }

    # Initialize confusion matrix
    ner_labels.append("NULL")
    label2idx = {label: i for i, label in enumerate(ner_labels)}    
    confusion = np.zeros((len(ner_labels), len(ner_labels)), dtype=np.int32)

    errors_doc = {}
    for doc in dataset:
        errors = {
            'tp-ner-em':[],
            'fp-ner-em':[],
            'fn-ner-em':[],
            'tp-ner-relaxed':[],
            'fp-ner-relaxed':[],
            'fn-ner-relaxed':[]
        }

        for sent in doc:
            counts, ner_result_by_class, errors, confusion = evaluate_sent(
                sent, counts, ner_result_by_class, errors, confusion, label2idx
            )       
        errors_doc[doc._doc_key] = errors

    scores_ner = compute_f1(
        counts["ner_predicted"], counts["ner_gold"], counts["ner_matched"]
    )
    scores_ner_soft = compute_f1(
        counts["ner_relaxed_predicted"], counts["ner_gold"], counts["ner_partial_matched"]
    )

    for label in ner_result_by_class:
        counts_label = ner_result_by_class[label]
        counts_label["precision"] = safe_div(counts_label["correct"], counts_label["pred"])
        counts_label["recall"] = safe_div(counts_label["correct"], counts_label["gold"])
        counts_label["f1"] = safe_div(2*counts_label["precision"]*counts_label["recall"], \
                                      counts_label["precision"]+counts_label["recall"])       
        
        counts_label["precision_relaxed"] = safe_div(counts_label["correct_relaxed"], counts_label["pred_relaxed"])
        counts_label["recall_relaxed"] = safe_div(counts_label["correct_relaxed"], counts_label["gold"])
        counts_label["f1_relaxed"] = safe_div(2*counts_label["precision_relaxed"]*counts_label["recall_relaxed"], \
                                      counts_label["precision_relaxed"]+counts_label["recall_relaxed"])  
        
    with open(os.path.join(output_dir, f"{task}_ner_result_by_class_e2e.json"), 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps(ner_result_by_class, indent=4))
        
    print("NER Result by class is saved!!!")

    print_predictions_entity(
        result=errors_doc,
        ner_label_result=ner_result_by_class,
        confusion=confusion,
        ner_labels=ner_labels,
        gold_file=dataset,
        output_file=f'{output_dir}/ner_predictions_sorted.txt'
    )

    result = dict(
        ner=scores_ner, 
        ner_soft=scores_ner_soft
    )

    return result


def print_predictions_entity(
    result, ner_label_result, confusion, ner_labels, gold_file, output_file
):
    
    with open(output_file, "w") as f:     
        header = ["NER-ENTITY_TYPE", \
                  "Prec", "Rec", "F1", \
                  "Prec-relaxed", "Rec-relaxed", "F1-relaxed", \
                  "Gold", "Pred", "Correct", "Pred-relaxed", 'Correct-relaxed']
        f.write("\t".join(header) + "\n")
        f.write("="*89 + "\n")
        
        # Record scores of NER by label
        for k, v in ner_label_result.items():
            record = [
                k, v['precision'], v['recall'], v['f1'], 
                v['precision_relaxed'], v['recall_relaxed'], v['f1_relaxed'], 
                int(v['gold']), int(v['pred']), int(v['correct']), 
                int(v['pred_relaxed']), int(v['correct_relaxed'])
            ]
            f.write("\t".join([str(r) for r in record]) + "\n")      
        f.write('\n\n')
        
        # Function to save confusion matrix to a text file
        ner_labels = [l.split('_')[0] for l in ner_labels]
        f.write('\t' + '\t'.join(ner_labels) + '\n')
        for i in range(len(ner_labels)):
            f.write(ner_labels[i] + '\t' + '\t'.join(map(str, confusion[i])) + '\n')
        f.write('\n\n')
        
        # list TP/FP/FNs
        for doc in gold_file:
            text = []
            abstract = []
            for idx, sent in enumerate(doc):
                text.extend(sent.text)
                if idx == 0:
                    title = ' '.join(sent.text)
                else:
                    abstract.extend(sent.text)

            f.write('|'.join([doc._doc_key, 't', title]))
            f.write('\n')
            abstract = ' '.join(abstract)
            f.write('|'.join([doc._doc_key, 'a', abstract]))
            f.write('\n\n')

            errors = result[doc._doc_key]
        
            ner_keys = ['fp-ner-em', 'fp-ner-relaxed', \
                        'fn-ner-em', 'fn-ner-relaxed', \
                        'tp-ner-em', 'tp-ner-relaxed']               
                
            # for idx, k in enumerate(ner_keys):
            #     for t in errors[k]:
            #         t = [
            #             str(t[0]), str(t[1]), ' '.join(text[t[0]:t[1]+1]), t[2]
            #         ]
            #         f.write("\t".join([k.upper(), doc._doc_key] + t))
            #         f.write("\n")
            #     if idx % 2 == 1:
            #         f.write("\n")

            errors_full = []
            for error_type, error_list in errors.items():
                for sample in error_list:
                    errors_full.append(
                        [error_type] + list(sample)
                    )
            errors_full = sorted(errors_full, key=lambda x: x[1])
            for idx, t in enumerate(errors_full):
                t = [
                    t[0].upper(),
                    doc._doc_key,
                    str(t[1]), 
                    str(t[2]), 
                    ' '.join(text[t[1]:t[2]+1]), t[3]
                ]
                f.write("\t".join(t))    
                f.write("\n")       

            f.write("\n")
                    
#                 for k in ner_keys_relaxed:
#                     for t in doc_result[k]:
#                         t = [str(t[0]), str(t[1]), row.text[t[0]:t[1]], t[2]]
#                         f.write("\t".join([k.upper(), row.pmid]+t))
#                         f.write("\n")
#                     f.write("\n")

                # for k in nel_keys:
                #     for t in doc_result[k]:
                #         if t[3]:
                #             cui = ';'.join([label2ref[t[2]], t[3]])
                #         else:
                #             cui = ''
                #         t = [str(t[0]), str(t[1]), row.text[t[0]:t[1]], t[2], cui]
                #         f.write("\t".join([k.upper(), row.pmid]+t))
                #         f.write("\n")
                #     f.write("\n")
                    
            
                
def analyze_relation_coverage(dataset):
    
    def overlap(s1, s2):
        if s2.start_sent >= s1.start_sent and s2.start_sent <= s1.end_sent:
            return True
        if s2.end_sent >= s1.start_sent and s2.end_sent <= s1.end_sent:
            return True
        return False

    nrel_gold = 0
    nrel_pred_cover = 0
    nrel_top_cover = 0

    npair_pred = 0
    npair_top = 0

    nrel_overlap = 0

    for d in dataset:
        for s in d:
            pred = set([ner.span for ner in s.predicted_ner])
            top = set([ner.span for ner in s.top_spans])
            npair_pred += len(s.predicted_ner) * (len(s.predicted_ner) - 1)
            npair_top += len(s.top_spans) * (len(s.top_spans) - 1)
            for r in s.relations:
                nrel_gold += 1
                if (r.pair[0] in pred) and (r.pair[1] in pred):
                    nrel_pred_cover += 1
                if (r.pair[0] in top) and (r.pair[1] in top):
                    nrel_top_cover += 1
                
                if overlap(r.pair[0], r.pair[1]):
                    nrel_overlap += 1

    print('Coverage by predicted entities: %.3f (%d / %d), #candidates: %d'%(nrel_pred_cover/nrel_gold*100.0, nrel_pred_cover, nrel_gold, npair_pred))
    print('Coverage by top 0.4 spans: %.3f (%d / %d), #candidates: %d'%(nrel_top_cover/nrel_gold*100.0, nrel_top_cover, nrel_gold, npair_top))
    print('Overlap: %.3f (%d / %d)'%(nrel_overlap / nrel_gold * 100.0, nrel_overlap, nrel_gold))
