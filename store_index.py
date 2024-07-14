from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

    # Now do stuff
if 'my_index' not in pc.list_indexes().names():
        
    
    pc.create_index(
            name='my_index', 
            dimension=1536, 
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-west-2'
        )
    )

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
#pinecone.init(api_key=PINECONE_API_KEY,
#              environment=PINECONE_API_ENV)


index_name="med-bot"

#Creating Embeddings for Each of The Text Chunks & storing
docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)

