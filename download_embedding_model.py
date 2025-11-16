import os
from transformers import AutoTokenizer, AutoModel

def download_model():
    # Download model and tokenizer
    if not os.path.exists("./models"):
        os.mkdir("./models")
        embedding_model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
        embedding_tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
        embedding_model.save_pretrained("./models/embedding_model")
        embedding_tokenizer.save_pretrained("./models/embedding_tokenizer")

        print("Successfully downloaded model !")
    else:
        print("Model has already existed !")

if __name__ == "__main__":
    download_model()