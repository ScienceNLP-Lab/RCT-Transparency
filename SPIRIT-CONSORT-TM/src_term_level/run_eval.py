import argparse
import os
from shared.data_structures_copied import Dataset, evaluate_predictions
from shared.utils import generate_analysis_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', type=str, default=None, required=True)
    parser.add_argument('--output_dir', type=str, default=None, required=True)
    parser.add_argument('--task', type=str, default=None, required=True)
    parser.add_argument('--dataset_name', type=str, default=None, required=True)
    args = parser.parse_args()

    data = Dataset(args.prediction_file)
    eval_result = evaluate_predictions(
        data, args.output_dir, task=args.task, dataset_name=args.dataset_name
    )

    args.csv_dir = os.path.join(args.output_dir, "analysis.csv")
    generate_analysis_csv(data, args.csv_dir)

    print('Evaluation result %s'%(args.prediction_file))

    print('NER - P: %f, R: %f, F1: %f'%(
        eval_result['ner']['precision'], 
        eval_result['ner']['recall'], 
        eval_result['ner']['f1']        
    ))

    print('NER - Pred: %f, Gold: %f, Correct: %f'%(
        eval_result['ner']['n_pred'], 
        eval_result['ner']['n_gold'], 
        eval_result['ner']['n_correct']
    ))
    
    print('NER Relaxed - P: %f, R: %f, F1: %f'%(
        eval_result['ner_soft']['precision'],
        eval_result['ner_soft']['recall'],
        eval_result['ner_soft']['f1']
    ))
    print('NER Soft - Pred: %f, Gold: %f, Correct: %f'%(
        eval_result['ner_soft']['n_pred'], 
        eval_result['ner_soft']['n_gold'], 
        eval_result['ner_soft']['n_correct']
    ))