import os

def get_documents(path):

    docs = []; classes=[]
    for file in os.listdir(path):
        try:
            with open(path + file, "r") as f:
                docs.append(
                    f.readlines()
                )
            classes.append(file.split(".")[-1])
        except:
            continue
    return docs,classes


def write_documents(docs, path, orig_path):
    os.makedirs(path, exist_ok=True)
    for i, file in enumerate(os.listdir(orig_path)):
        try:
            with open(path + file, "w") as f:
                f.writelines(docs[i])
        except:
            continue
            
def get_documents_v2(path):
    docs = []; classes = []
    for direc in os.listdir(path):
        for file in os.listdir(path + direc):
            try:
                with open(path + direc + "/" + file, "r") as f:
                    sentences = f.read()
                    docs.append(sentences)
                    classes.append(direc)
            except:
                continue
    return docs, classes


def write_documents_v2(docs, path, orig_path):
    os.makedirs(path, exist_ok=True)
    i=0
    for direc in os.listdir(orig_path):
        os.makedirs(path+"/"+direc, exist_ok=True)
        
        for file in os.listdir(orig_path + "/" + direc):
            try:
                with open(path + direc + "/" + file, "w") as f:
                    f.write(docs[i])
                i+=1
            except:
                continue