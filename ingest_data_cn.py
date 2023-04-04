
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import DirectoryLoader
import jieba as jb
import chardet


# 参考链接
# https://blog.csdn.net/weixin_42608414/article/details/129493302


#中文分词处理
files=['state_of_the_policy.txt']
 
for file in files:
    #读取data文件夹中的中文文档
    my_file=f"./data/{file}"
    with open(my_file,"r",encoding='utf-8') as f:  
        data = f.read()
        print("open" + f.encoding)
    
    #对中文文档进行分词处理
    cut_data = " ".join([w for w in list(jb.cut(data))])
    #分词处理后的文档保存到data文件夹中的cut子文件夹中
    cut_file=f"./data/cut/cut_{file}"
    with open(cut_file, 'w', encoding="utf-8") as f:   
        f.write(cut_data)
        print("cut" + f.encoding)
        f.close()

#加载文档
loader = DirectoryLoader('./data/cut',glob='**/*.txt')
docs = loader.load()

#文档切块
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
doc_texts = text_splitter.split_documents(docs)

#调用openai Embeddings
import os
os.environ["OPENAI_API_KEY"] = "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"
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
#向量化
vectordb = Chroma.from_documents(doc_texts, embeddings, persist_directory="./data/cut")
vectordb.persist()
#创建聊天机器人对象chain
chain = ChatVectorDBChain.from_llm(OpenAI(temperature=0, model_name="gpt-3.5-turbo"), vectordb, return_source_documents=True)

#交互函数
def get_answer(question):
  chat_history = []
  result = chain({"question": question, "chat_history": chat_history});
  return result["answer"]

# Query Your Index
question = "20大报告谁主讲的?"
print(get_answer(question))


