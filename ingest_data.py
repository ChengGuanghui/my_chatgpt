
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.text_splitter import TokenTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import DirectoryLoader
import jieba as jb

# 参考链接
# https://blog.csdn.net/weixin_42608414/article/details/129493302

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


