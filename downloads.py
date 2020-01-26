import os
import zipfile,wget


def download_data():
    print("Downloading word embedding...")
    downloaded_glove1 = wget.download("http://nlp.stanford.edu/data/{}".format('glove.6B.zip'))
    downloaded_snli = wget.download("https://nlp.stanford.edu/projects/snli/snli_1.0.zip")
    zip = zipfile.ZipFile(downloaded_snli)
    zip.extractall(path="./Data")

    if not os.path.exists("./Data"):
        os.mkdir("./Data")
    print("Extracting...")
    zip = zipfile.ZipFile(downloaded_glove1)
    zip.extractall(path="./Data")
    print("done")


if __name__ == '__main__':
    download_data()