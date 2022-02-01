import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import load_dataset
import torch.nn.functional as F
import jiwer
import time
import argparse    
import pickle
import numpy as np


def load_data(args):   
    if args.dataset == "ag_news":
        dataset = load_dataset("ag_news")
        num_labels = 4
    elif args.dataset == "yelp":
        dataset = load_dataset("yelp_polarity")
        num_labels = 2
    elif args.dataset == "mnli":
        dataset = load_dataset("glue", "mnli")
        num_labels = 3

    dataset = dataset.shuffle(seed=0)

    return dataset, num_labels

def main(args):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset 
    dataset, num_labels = load_data(args)

    # load the fine-tuned classification model
    model_checkpoint = f"../Classifier//result/gpt2_{args.dataset}_finetune.pth"
    text_classification_model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=num_labels).to(device)
    text_classification_model.load_state_dict(torch.load(model_checkpoint, map_location=device))

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.model_max_length = 512
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    text_classification_model.config.pad_token_id = text_classification_model.config.eos_token_id

    # tokenize the data
    if args.dataset == "mnli":
        text_key = args.attack_target
        testset_key = "validation_matched"
        preprocess_function = lambda examples: tokenizer(
            examples['premise'], examples['hypothesis'], max_length=256, truncation=True)
    else:
        text_key = 'text'
        testset_key = 'test'
        preprocess_function = lambda examples: tokenizer(examples['text'], max_length=256, truncation=True)

    tokenized_dataset = dataset.map(preprocess_function, batched=True) 

    # Find the embeddings of all tokens
    with torch.no_grad():
        embeddings = text_classification_model.get_input_embeddings()(torch.arange(0, tokenizer.vocab_size).long().to(device))


    # Word Error Rate
    def wer(x, y):
        x = " ".join(["%d" % i for i in x])
        y = " ".join(["%d" % i for i in y])

        return jiwer.wer(x, y)

    # Block sparsity loss    
    def L2L1(x):
        L2 = x.norm(p=2,dim=1)
        return L2.mean()



    def gen_adv(sentence_number, lr = 0.15, list_w_sim =[10, 8, 5, 2]):

        for w_sim_coef in list_w_sim:
            
            label = tokenized_dataset[testset_key][sentence_number]['label']
            input_ids = tokenized_dataset[testset_key][sentence_number]['input_ids']
            w = [embeddings[i].cpu().numpy() for i in input_ids]
            w = torch.tensor(w,requires_grad=False).to(device)

            all_w = [ torch.clone(w.detach()) ]


            w_a = torch.tensor(w,requires_grad=True).to(device)
            optimizer = torch.optim.Adam([w_a],lr=lr)
            
            pred_project = text_classification_model(inputs_embeds=w_a.unsqueeze(0)).logits

            itr = 0
            while torch.argmax(pred_project)==label:
                itr+=1

                pred = text_classification_model(inputs_embeds=w_a.unsqueeze(0)).logits

                adv_loss = -F.cross_entropy(pred, label * torch.ones(1).long().to(device))       
            
                total_loss = adv_loss + w_sim_coef * L2L1(w_a-all_w[0]) 

                optimizer.zero_grad()
                total_loss.backward()

                if args.dataset == "mnli":
                    premise_length = len(tokenizer.encode(tokenized_dataset['validation_matched'][sentence_number]['premise']))
                    if args.attack_target == "premise":
                        fix_idx = torch.from_numpy(np.arange(premise_length, len(input_ids))).to(device)
                    else:
                        fix_idx = torch.from_numpy(np.arange(0, premise_length)).to(device)
                    w_a.grad.index_fill_(0, fix_idx, 0)


                optimizer.step()
                
                print(f'itr: {itr} \t w_sim_coef: {w_sim_coef} \t lr: {lr} \t loss: {adv_loss.data.item(), total_loss.data.item()}')

                cosine = -1 * torch.matmul(w_a,embeddings.transpose(1, 0))/torch.unsqueeze(w_a.norm(2,1), 1)/torch.unsqueeze(embeddings.norm(2,1), 0)
                index_prime = torch.argmin(cosine,dim=1)
                w_prime = embeddings[index_prime]
                
                
                print(tokenizer.decode(index_prime))

                pred_project = text_classification_model(inputs_embeds=w_prime.unsqueeze(0)).logits
                
                
                if torch.equal(w_prime,w)==False:
                    skip=False
                    for w_ in all_w:
                        if torch.equal(w_prime,w_):
                            skip=True
                    if skip==False:
                        print('*****************')
                        w_a.data=w_prime
                        all_w.append(torch.clone(w_prime.detach()))

                if itr> 500:
                    break
            if torch.argmax(pred_project)!=label:
                break
            
        if itr==0: 
            return tokenized_dataset[testset_key][sentence_number][text_key], tokenized_dataset[testset_key][sentence_number][text_key], torch.argmax(pred_project), label, lr, w_sim_coef, itr, 0
        else:    
            error_rate = wer(index_prime,input_ids)   
            if args.dataset == "mnli":
                if args.attack_target == "premise":
                    return tokenizer.decode(index_prime[:premise_length]), tokenized_dataset[testset_key][sentence_number][text_key], torch.argmax(pred_project), label, lr, w_sim_coef, itr, error_rate
                else:
                    return tokenizer.decode(index_prime[premise_length:]), tokenized_dataset[testset_key][sentence_number][text_key], torch.argmax(pred_project), label, lr, w_sim_coef, itr, error_rate
            else:
                return tokenizer.decode(index_prime), tokenized_dataset[testset_key][sentence_number][text_key], torch.argmax(pred_project), label, lr, w_sim_coef, itr, error_rate

    
    attack_dict = {}

    time_begin = time.time()
    for idx in range(args.start_index, args.start_index+args.num_samples):
        sentence_adv, sentence_org, label_adv, label_org, lr,  w_sim_coef, itr, error_rate = gen_adv(idx)
        
        if label_adv == label_org:
            sentence_adv, sentence_org, label_adv, label_org, lr, w_sim_coef, itr, error_rate = gen_adv(idx, lr=0.3, list_w_sim=[10, 7, 5, 3, 1])
            if label_adv == label_org:
                print(f'Failed to generate Adv attack on {idx}!')
                attack_result = 'failed'
            else:
                print(f'Successed to generate Adv attack on {idx}!')
                attack_result = 'success'
        else:
            print(f'Successed to generate Adv attack on {idx}!')
            attack_result = 'success'
        
        attack_dict[idx]=[attack_result, sentence_adv, sentence_org, label_adv.item(), label_org, lr, w_sim_coef, itr, error_rate]

    print(f'finished attack for {args.num_samples} samples in {time.time()-time_begin} seconds!')
    print(time.time()-time_begin)
        
    os.makedirs(args.result_folder, exist_ok=True)
    with open(f'{args.result_folder}/{args.start_index}_{args.start_index+args.num_samples}_{args.dataset}.pkl', 'wb') as f:
        pickle.dump(attack_dict, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Block-Sparse Attack.")

    # Bookkeeping
    parser.add_argument("--result_folder", default="result", type=str,
        help="folder for loading trained models")

    # Data
    parser.add_argument("--dataset", default="ag_news", type=str,
        choices=["ag_news", "yelp", "mnli"],
        help="classification dataset to use")
    parser.add_argument("--num_samples", default=100, type=int,
        help="number of samples to attack")

    # Attack setting
    parser.add_argument("--start_index", default=0, type=int,
        help="starting sample index")
    parser.add_argument("--attack_target", default="premise", type=str,
        choices=["premise", "hypothesis"],
        help="attack either the premise or hypothesis for MNLI")

    args = parser.parse_args()

    main(args)
