
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.text_splitter import TokenTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import DirectoryLoader
import jieba as jb

# 参考链接
# https://blog.csdn.net/weixin_42608414/article/details/129493302


#中文分词处理
files=['state_of_the_policy.txt']
 
for file in files:
    #读取data文件夹中的中文文档
    my_file=f"./data/{file}"
    with open(my_file,"r",encoding='utf-8') as f:  
        data = f.read()
    
    #对中文文档进行分词处理
    cut_data = " ".join([w for w in list(jb.cut(data))])
    #分词处理后的文档保存到data文件夹中的cut子文件夹中
    cut_file=f"./data/cut/cut_{file}"
    with open(cut_file, 'w') as f:   
        f.write(cut_data)
        f.close()


#加载文档
loader = DirectoryLoader('./data/cut',glob='**/*.txt')
docs = loader.load()

#文档切块
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
doc_texts = text_splitter.split_documents(docs)

#调用openai Embeddings
import os

os.environ["OPENAI_API_KEY"] = "your-openai_api_key"
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
#向量化
vectordb = Chroma.from_documents(doc_texts, embeddings, persist_directory="./data/cut")
vectordb.persist()
#创建聊天机器人对象chain
chain = ChatVectorDBChain.from_llm(OpenAI(temperature=0, model_name="gpt-3.5-turbo"), vectordb, return_source_documents=True)
————————————————
版权声明：本文为CSDN博主「-派神-」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_42608414/article/details/129493302
# Load Data
from langchain.document_loaders import TextLoader
loader = TextLoader("state_of_the_policy.txt")

# create your index
from langchain.indexes import VectorstoreIndexCreator
index = VectorstoreIndexCreator().from_loaders([loader])

# Query Your Index
query = "20大报告谁主讲的?"
answer = index.query(query)

print(answer)


