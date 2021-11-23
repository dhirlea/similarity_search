from summarizer import Summarizer
import torch
import spacy
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def main():
    # Set random seeds
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    body = "The Chrysler Building, the famous art deco New York skyscraper, will be sold for a small fraction of its previous sales price.\
        The deal, first reported by The Real Deal, was for $150 million, according to a source familiar with the deal.\
        Mubadala, an Abu Dhabi investment fund, purchased 90% of the building for $800 million in 2008.Real estate firm Tishman Speyer had owned the other 10%.\
        The buyer is RFR Holding, a New York real estate company.\
        Officials with Tishman and RFR did not immediately respond to a request for comments.\
        It's unclear when the deal will close.\
        The building sold fairly quickly after being publicly placed on the market only two months ago.\
        The sale was handled by CBRE Group.\
        The incentive to sell the building at such a huge loss was due to the soaring rent the owners pay to Cooper Union, a New York college, for the land under the building.\
        The rent is rising from $7.75 million last year to $32.5 million this year to $41 million in 2028.\
        Meantime, rents in the building itself are not rising nearly that fast.\
        While the building is an iconic landmark in the New York skyline, it is ompeting against newer office towers with large floor-to-ceiling windows and all the modern amenities.\
        Still the building is among the best known in the city, even to people who have never been to New York.\
        It is famous for its triangle-shaped, vaulted windows worked into the stylized crown, along with its distinctive eagle gargoyles near the top.\
        It has been featured prominently in many films, including Men in Black 3, Spider-Man, Armageddon, Two Weeks Notice and Independence Day.\
        The previous sale took place just before the 2008 financial meltdown led to a plunge in real estate prices.\
        Still there have been a number of high profile skyscrapers purchased for top dollar in recent years, including the Waldorf Astoria hotel, which Chinese firm Anbang Insurance purchased in 2016 for nearly $2 billion, and the Willis Tower in Chicago, which was formerly known as Sears Tower, once the world's tallest.\
        Blackstone Group (BX) bought it for $1.3 billion 2015.\
        The Chrysler Building was the headquarters of the American automaker until 1953, but it was named for and owned by Chrysler chief Walter Chrysler, not the company itself.\
        Walter Chrysler had set out to build the tallest building in the world, a competition at that time with another Manhattan skyscraper under construction at 40 Wall Street at the south end of Manhattan. He kept secret the plans for the spire that would grace the top of the building, building it inside the structure and out of view of the public until 40 Wall Street was complete.\
        Once the competitor could rise no higher, the spire of the Chrysler building was raised into view, giving it the title.\
    "

    nlp = spacy.load('en_core_web_sm')

    doc = nlp(body)
    sentence_list = []
    for sentence in doc.sents:
        sentence_list.append(sentence.text.strip())

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
    for sentence in sentence_list:
        sentence_embeddings.extend(model.run_embeddings(sentence, num_sentences=1, min_length=5)) # min length defaults to 40 chars to summarize text, but sentences can be shorter
    
    sentence_embeddings_array  = np.array(sentence_embeddings)

    clustering = KMeans(n_clusters=3, random_state=0).fit(sentence_embeddings_array)

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


    cluster1 = []
    cluster2 = []
    cluster3 = []
    for i in range(len(two_dim)):
        if clustering.labels_[i] == 0:
            cluster1.append(two_dim[i])
        elif clustering.labels_[i] == 1:
            cluster2.append(two_dim[i])
        else:
            cluster3.append(two_dim[i])

    plt.scatter(np.array(cluster1)[:,0], np.array(cluster1)[:,1], c='c')
    plt.scatter(np.array(cluster2)[:,0], np.array(cluster2)[:,1], c='g')
    plt.scatter(np.array(cluster3)[:,0], np.array(cluster3)[:,1], c='b')
    plt.scatter(np.array(centroids_two_dim)[:,0],np.array(centroids_two_dim[:,1]), c='r')
    plt.xlabel('1st Principal Component')
    plt.ylabel('2nd Principal Component')
    plt.savefig('sentence_clusters.png')

if __name__ == '__main__':
    main()
