import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances


def plot_history(data, label, ax = None):
    if ax==None:
        f, ax = plt.subplots(1,1)
    ax.plot(data.epoch, data.history[label], "--*", label="training set")
    ax.plot(data.epoch, data.history["val_"+label], "--.", label="validation set")
    
    #ax.set_title(label+" scores")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("binary crossentropy" if label=="loss" else "accuracy")
    ax.legend()
    ax.grid()
    return f, ax
    

def get_predictions(model, X_test):
    return (model.predict(X_test)>0.5)*1


def conf_matrix(y_test, preds):
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, preds)
    ).plot()
    
    
def get_k_nearest_from(word, word2int, int2word, vectors, k=10, low_memory=False):
    idx = word2int[word]
    if low_memory==False:
        distances = euclidean_distances(vectors)
        dist_rank = np.argsort(distances[idx])
        
    else:
        word_vec = vectors[idx]
        distances = euclidean_distances(vectors, word_vec.reshape(1,-1)).ravel()
        dist_rank = np.argsort(distances)
        
    return list(map(lambda i: int2word[i], dist_rank[1:k+1]))
    