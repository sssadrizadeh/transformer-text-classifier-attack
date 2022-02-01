import pickle
import numpy as np
import jiwer
import tensorflow as tf
import tensorflow_hub as hub
import argparse    

# Word Error Rate
def wer(x, y):
    x = " ".join(["%d" % i for i in x])
    y = " ".join(["%d" % i for i in y])
    print(x,y)

    return jiwer.wer(x, y)


# Universal Sentence Encoder
class USE:
    def __init__(self):
        self.encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def compute_sim(self, clean_texts, adv_texts):
        clean_embeddings = self.encoder(clean_texts)
        adv_embeddings = self.encoder(adv_texts)
        cosine_sim = tf.reduce_mean(tf.reduce_sum(clean_embeddings * adv_embeddings, axis=1))

        return float(cosine_sim.numpy())

def main(args):

    with (open(f'{args.result_folder}/{args.start_index}_{args.start_index+args.num_samples}_{args.dataset}.pkl', "rb")) as f:
        d = (pickle.load(f))

    use = USE()

    def eval(d,similarity = [],bad = 0,wer_ = 0,fail=0):
        for idx in range(args.start_index, args.start_index+args.num_samples):
            if d[idx][0]=='success':
                s = use.compute_sim([d[idx][2]], [d[idx][1]])
                if s<0.3:
                    # this sentence failed due to low similarity
                    bad+=1
                else:
                    wer_+=d[idx][-1]
                    similarity.append(s)
            else:
                fail+=1
        return similarity, bad, wer_, fail
        
    similarity, bad, wer_, fail = eval(d)

    print("\n\n")
    print("*********** RESULTS ***********")
    print("\n")

    print(f'Average semantic similarity in successful attacks: \t {sum(similarity) / len(similarity)}')
    print(f'Token error rate in successful attacks: \t \t {wer_/len(similarity)}')
    print(f'Number of failed attacks due to low similarity: \t {bad}')
    print(f'Total number of failed attacks: \t \t \t {fail+bad}')
    print(f'Success attack rate: \t \t \t \t \t {1-(fail+bad)/len(d)}')
    print("\n")

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
    # parser.add_argument("--attack_target", default="premise", type=str,
    #     choices=["premise", "hypothesis"],
    #     help="attack either the premise or hypothesis for MNLI")

    args = parser.parse_args()

    main(args)