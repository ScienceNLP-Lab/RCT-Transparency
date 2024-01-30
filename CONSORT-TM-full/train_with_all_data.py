import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from transformers import AdamW, get_linear_schedule_with_warmup
from models.bert_model import BERT
from data import colloate_fn, colloate_fn_contextual, data_load, contextual_load, adjust_id
from utils import EarlyStopping, LRScheduler
import torch
from torch.utils.data import DataLoader
import tqdm
import time, os, json
from argparse import ArgumentParser
import pickle

def train_bert(config):
	base_folder = config.section + "_" + config.mode + "_" + config.target[1:-1] + "_header=" + config.section_header + "_rltv=" + str(config.rltv) + "_section_emb=" + str(config.section_emb) + "_section_avg_sep=" + config.section_avg_sep + \
	"_augmentation_mode=" + str(config.augmentation_mode) + "_header_information_contextual_" + str(config.header_information_contextual)
	folders = pickle.load(open( "folders_5.p", "rb" ) )
	
	if config.save:
		if not os.path.exists(base_folder):
			os.mkdir(base_folder)
		timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
		log_home_dir = os.path.join(base_folder, timestamp)
		os.mkdir(log_home_dir)		

	# for folder_no in range(5):
	# 	train_pmids = folders[0][folder_no]
	# 	test_pmids = folders[1][folder_no]

	if config.mode == "contextual":

		all_dataset, list_name = contextual_load(config.train_file, config)

		# test_dataset, list_name = contextual_load(config.train_file, config)

		num_labels = len(list_name[0])


		if config.save:
			timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
			log_dir = os.path.join(log_home_dir, timestamp)
			os.mkdir(log_dir)
			log_file = os.path.join(log_dir, 'log.txt')
			best_model = os.path.join(log_dir, 'best.mdl')
			with open(log_file, 'w', encoding='utf-8') as w:
				w.write(str(config) + '\n')

	else:
		train_dataset, list_name = data_load(config.train_file, train_pmids, config, folder_no, current_mode = "train")
		test_dataset, list_name = data_load(config.train_file, test_pmids, config, folder_no, current_mode = "test")
		num_labels = len(list_name[0])

		train = train_dataset
		test = test_dataset

		max_sen_num = max(inst.sent_id for inst in train)
		config.sen_num = max_sen_num
		# if articles in test set is longer than articles in train set
		test = adjust_id(test, max_sen_num)

		if config.save:
			timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
			log_dir = os.path.join(log_home_dir, timestamp)
			os.mkdir(log_dir)
			log_file = os.path.join(log_dir, 'log.txt')
			best_model = os.path.join(log_dir, 'best.mdl')
			with open(log_file, 'w', encoding='utf-8') as w:
				w.write(str(config) + '\n')



	use_gpu = config.use_gpu
	if use_gpu and config.gpu_device >= 0:
		torch.cuda.set_device(config.gpu_device)


	model = BERT(config, num_labels)
	model.load_bert(config.bert_model_name)
	batch_num = len(all_dataset) // config.batch_size
	total_steps = batch_num * config.max_epoch
	# test_batch_num = len(test) // config.eval_batch_size + (len(test) % config.eval_batch_size != 0)

	if use_gpu:
		model.cuda()
	param_groups = [
		{
			'params': [p for n, p in model.named_parameters() if n.startswith('bert')],
			'lr': config.bert_learning_rate, 'weight_decay': config.bert_weight_decay
		},
		{
			'params': [p for n, p in model.named_parameters() if not n.startswith('bert')],
			'lr': config.learning_rate, 'weight_decay': config.weight_decay
		},
	]

	optimizer = AdamW(params=param_groups, eps=1e-8)
	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=0,
		num_training_steps=total_steps,
	)

	Loss = torch.nn.BCELoss()
	best_loss, best_epoch = 100, 0

	if config.early_stopping:
		print('INFO: Initializing early stopping')
		early_stopping = EarlyStopping()
	if config.lr_scheduler:
		print('INFO: Initializing learning rate scheduler')
		lr_scheduler = LRScheduler(optimizer)

	if config.target == "[CLS]":
		if config.mode == "single":
			for epoch in range(config.max_epoch):
				running_loss = 0.0
				progress = tqdm.tqdm(total=batch_num, ncols=75, desc='Train {}'.format(epoch))
				optimizer.zero_grad()
				for batch_idx, batch in enumerate(DataLoader(train, batch_size=config.batch_size, shuffle=True, collate_fn=colloate_fn)):
					optimizer.zero_grad()
					model.train()
					prediction = model(batch)
					loss = Loss(prediction, batch.labels)

					loss.backward()
					torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
					optimizer.step()
					scheduler.step()
					running_loss += loss.item()
					progress.update(1)

				progress.close()
				print('INFO: Training loss_fn is ', round(running_loss/len(train), 4))

				if running_loss/len(train) < best_loss:
					best_loss = running_loss/len(train)
					best_epoch = epoch

					if config.save:
						# sid, report, eval_running_loss, target_result, valid_result, auc = evaluate(model, Loss, test, config, test_batch_num, epoch, 'TEST', list_name, log_dir)
						result = json.dumps({'epoch': epoch, 'train_loss': running_loss/len(train), 'report_test': report})
						torch.save(dict(model=model.state_dict(), config=config), best_model)
						with open(log_file, 'a', encoding='utf-8') as w:
							w.write(result + '\n')
						print('INFO: Log file: ', log_file)
				else:
					if config.save:
						result = json.dumps({'epoch': epoch, 'train_loss': best_loss})
						with open(log_file, 'a', encoding='utf-8') as w:
							w.write(result + '\n')
						print('INFO: Log file: ', log_file)
				model.train()
		elif config.mode == "contextual":
			for epoch in range(config.max_epoch):
				running_loss = 0.0
				progress = tqdm.tqdm(total=batch_num, ncols=75, desc='Train {}'.format(epoch))
				optimizer.zero_grad()
				for batch_idx, batch in enumerate(DataLoader(all_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=colloate_fn_contextual)):
					print(batch_idx)
					optimizer.zero_grad()
					model.train()
					prediction = model(batch)
					loss = Loss(prediction, batch.labels)

					loss.backward()
					torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
					optimizer.step()
					scheduler.step()
					running_loss += loss.item()
					progress.update(1)

				progress.close()
				print('INFO: Training loss_fn is ', round(running_loss/len(all_dataset), 4))

				if running_loss/len(all_dataset) < best_loss:
					best_loss = running_loss/len(all_dataset)
					best_epoch = epoch

					if config.save:
						# sid, report, eval_running_loss, target_result, valid_result, auc = evaluate(model, Loss, test, config, test_batch_num, epoch, 'TEST', list_name, log_dir)
						result = json.dumps({'epoch': epoch, 'train_loss': running_loss/len(all_dataset)})
						torch.save(dict(model=model.state_dict(), config=config), best_model)
						with open(log_file, 'a', encoding='utf-8') as w:
							w.write(result + '\n')
						print('INFO: Log file: ', log_file)
				else:
					if config.save:
						result = json.dumps({'epoch': epoch, 'train_loss': best_loss})
						with open(log_file, 'a', encoding='utf-8') as w:
							w.write(result + '\n')
						print('INFO: Log file: ', log_file)
				model.train()

	if config.target == "[SEP]":
		for epoch in range(config.max_epoch):
			running_loss = 0.0
			progress = tqdm.tqdm(total=batch_num, ncols=75, desc='Train {}'.format(epoch))
			optimizer.zero_grad()
			for batch_idx, batch in enumerate(DataLoader(
				train, batch_size=config.batch_size,
				shuffle=True, collate_fn=colloate_fn_contextual)):
				optimizer.zero_grad()
				model.train()
				prediction = model(batch)
				labels = torch.unsqueeze(batch.labels, 1)
				labels = labels.repeat(1, 388, 1)
				loss = Loss(prediction, labels)
				loss_mask = torch.unsqueeze(batch.loss_mask, 2)
				loss = loss * loss_mask
				loss = torch.mean(torch.sum(loss, 1))

				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				optimizer.step()
				scheduler.step()
				running_loss += loss.item()
				progress.update(1)

			progress.close()
			print('INFO: Training loss_fn is ', round(running_loss/len(train), 4))

			if running_loss/len(train) < best_loss:
				best_loss = running_loss/len(train)
				best_epoch = epoch

				if config.save:
					sid, report, eval_running_loss, target_result, valid_result, auc = evaluate(model, Loss, test, config, test_batch_num, epoch, 'TEST', list_name, log_dir)
					result = json.dumps({'epoch': epoch, 'train_loss': running_loss/len(train), 'report_test': report})
					torch.save(dict(model=model.state_dict(), config=config), best_model)
					with open(log_file, 'a', encoding='utf-8') as w:
						w.write(result + '\n')
					print('INFO: Log file: ', log_file)
			else:
				if config.save:
					result = json.dumps({'epoch': epoch, 'train_loss': best_loss})
					with open(log_file, 'a', encoding='utf-8') as w:
						w.write(result + '\n')
					print('INFO: Log file: ', log_file)
			model.train()
	if config.save:
		best = json.dumps({'best epoch': best_epoch})
		with open(log_file, 'a', encoding='utf-8') as w:
			w.write(best + '\n')

if __name__ == '__main__':
	# hyperparameters
    parser = ArgumentParser()

    parser.add_argument('--train_file', type=str,
                        help='path to the training file')   

    parser.add_argument('--test_file', type=str,
                        help='path to the test file')   

    parser.add_argument('--use_gpu', type=bool, default=1,
                        help='if or not use GPU, choose from True or False')   

    parser.add_argument('--gpu_device', type=int, default=0, help='number of GPU devices')

    parser.add_argument('--target', type=str, 
                        help='target token, choose either [CLS] or [SEP]')
    
    parser.add_argument('--mode', type=str, 
                        help='input mode, choose either contextual or single')

    parser.add_argument('--save', type=int, default=1,
                        help='create log file or not, choose either 1 or 0')
    
    parser.add_argument('--section_header', type=str, 
                        help='section header to involve, choose from none, both, inner or outer')

    parser.add_argument('--bert_model_name', type=str, 
                        help='pretrained language model name, choose from huggingface')

    parser.add_argument('--bert_dropout', type=float, default=0.1,
                        help='dropout rate')
    
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size')
    
    parser.add_argument('--eval_batch_size', type=int, default=4,
                        help='batch size')

    parser.add_argument('--max_epoch', type=int, default=20,
                        help='number of epoch')

    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay rate of linear neural network') 
   
    parser.add_argument('--bert_weight_decay', type=float, default=0,
                        help='weight decay rate of the bert model') 
    
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate of linear neural network') 

    parser.add_argument('--bert_learning_rate', type=float, default=1e-5,
                        help='learning rate of bert finetuning')   

    parser.add_argument('--early_stopping', type=int, default=0,
                        help='early stopping')   

    parser.add_argument('--lr_scheduler', type=int, default=0,
                        help='learning rate scheduler') 

    parser.add_argument('--test_on_specific_section', type=str, help='test a model performance on which specific section? choose from Methods, Results, Discussion or whole')

    parser.add_argument('--header_information_contextual', type=int, 
                        help='0 - not involve any header information in contextual setting; 1 - involve header information for the current sentence only in contextual \
                        setting; 2 - involve header information for the preceding, current and trailing sentences in contextual setting')

    parser.add_argument('--augmentation_mode', type=int, help='0 - no augmentation; 1 - (generative by GPT-4) concatenate all augmentation outcomes with the \
                        original samples in the training set;  2 - (generative by GPT-4) split augmentation outcomes into different folders in the training set; \
                        3 - (rephrasing by GPT-4) add the rephrased data to the training set; \
                        4 - (EDA) add the data augmented by EDA method; \
                        5 - (UMLS-EDA) add the data augmented by UMLS-EDA method')  

    parser.add_argument('--augmentation_file', type=str, help='the path to the augmentation file')

    parser.add_argument('--sent_dim', type=int, 
                        help='dimension of relative/absolute sentence position embedding')
    
    parser.add_argument('--rltv', type=int, 
                        help='use relative/absolute/none sentence position (1/0/-1)')

    parser.add_argument('--section_emb', type=int,
                        help='whether the model would add section headers as a separate feature')
    
    parser.add_argument('--position_emb', type=int,
                        help='whether the model would add position embedding')
    
    parser.add_argument('--section_dim', type=int,
                        help='the number of dimensions of section headers')

    parser.add_argument('--rltv_bins_num', type=int, 
    					help='number of bins for relative postion embedding') 

    parser.add_argument('--section_avg_sep', type=str, 
    					help='whether the model would add section headers as a separate feature - use seperate method (sep) or average (avg)') 

    parser.add_argument('--section', type=str, 
    					help='select the section-specific data from whole/Methods/Results/Discussion and train the model')

    parser.add_argument('--checkpoint', type=str, 
    					help='location of the checkpoint file')
    
    parser.add_argument('--folder_no', type=int, 
    					help='select data from a specific folder (choose from 1/2/3/4/5)')


    config = parser.parse_args()
    print("config", config)
    train_bert(config)
