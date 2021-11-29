from summarizer import Summarizer
import torch
import json
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from tqdm import tqdm

def main():
    # Set random seeds
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    with open('parsed_report.json', encoding='utf-8') as f:
        body_dict = json.load(f)
    
    
    raw_sentence_list = list(body_dict.values())
    sentence_list = [sentence for sentence in raw_sentence_list[:100] if len(sentence)>1] #TODO only segmenting first 100 sentences. Extend to entire document
    body = ' '.join(sentence_list)

    # Instantiate model
    model = Summarizer()
    model.model.model.to(device) # automatically defaults to GPU, but good to check
    print(f' Using device {model.model.model.device.type} \n')

    #Summarize text
    summary = model(body, ratio=0.2)
    print('Summary of input text is \n',summary, '\n')
    summary_embeddings = torch.tensor(model.run_embeddings(body, ratio=0.2))
    # print(f' Shape of extractive summary embeddings is {summary_embeddings.shape} \n') # no_sentences x 1024


    # Encode individual sentences

    sentence_embeddings = []
    for sentence in tqdm(sentence_list):
        sentence_embeddings.extend(model.run_embeddings(sentence, num_sentences=1, min_length=1)) # min length defaults to 40 chars to summarize text, but sentences can be shorter
    
    sentence_embeddings_array  = np.array(sentence_embeddings)

    number_of_clusters = 5

    clustering = KMeans(n_clusters=number_of_clusters, random_state=0).fit(sentence_embeddings_array)

    cluster_dict = {}
    for sentence, label in zip(sentence_list, clustering.labels_):
        if label not in cluster_dict.keys():
            cluster_dict[label] = [sentence]
        else:
            cluster_dict[label].append(sentence)
    
    print(f'The text has the following clusters \n {cluster_dict} \n')

    sentence_centroids = {}    
    centroids_dict = {i:np.infty for i in range(len(clustering.cluster_centers_))}

    for centroid in clustering.cluster_centers_:
        for i in range(len(sentence_embeddings_array)):
            index = clustering.cluster_centers_.tolist().index(list(centroid))
            if np.sqrt(np.sum((sentence_embeddings_array[i] - np.array(centroid))**2)) < centroids_dict[index]:
                centroids_dict[index] = np.sqrt(np.sum((sentence_embeddings_array[i] - np.array(centroid))**2))
                sentence_centroids[index] = sentence_list[i]
    
    print('The sentences closest to the cluster centroids are \n')
    print(sentence_centroids)

    two_dim = PCA(random_state=0, n_components=2).fit_transform(sentence_embeddings_array)
    centroids_two_dim = PCA(random_state=0, n_components=2).fit_transform(clustering.cluster_centers_)

    PC_dict ={} # keys are cluster labels (0,1,2,etc), values are PC values
    for i in range(len(two_dim)):
        if clustering.labels_[i] not in PC_dict.keys():
            PC_dict[clustering.labels_[i]] = [two_dim[i]]
        else:
            PC_dict[clustering.labels_[i]].append(two_dim[i])
    
    color = iter(cm.rainbow(np.linspace(0, 1, len(PC_dict)+1)))

    for i in range(len(PC_dict)):
        c = next(color)
        plt.scatter(np.array(PC_dict[i])[:,0], np.array(PC_dict[i])[:,1], color=c)

    c = next(color)
    plt.scatter(np.array(centroids_two_dim)[:,0],np.array(centroids_two_dim[:,1]), color=c)
    plt.xlabel('1st Principal Component')
    plt.ylabel('2nd Principal Component')
    plt.savefig('sentence_clusters.png')

if __name__ == '__main__':
    main()
