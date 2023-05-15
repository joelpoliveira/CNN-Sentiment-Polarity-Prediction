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
