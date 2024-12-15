from rank_bm25 import BM25Okapi
import re
import jieba

# 读取文件内容并逐行建立文档列表
def load_documents(file_path):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 清洗文本，去除多余空白符
            text = line.strip()
            if text:
                documents.append(text)
    return documents

# 分词器函数
def tokenize(text):
    # 使用nltk的word_tokenize进行中文分词
    return [x for x in jieba.cut(text, cut_all=True)]
    # return word_tokenize(text)

# 建立倒排索引
def build_inverted_index(documents):
    # 对文档列表的每个文档进行分词
    tokenized_docs = [tokenize(doc) for doc in documents]
    
    # 使用BM25算法处理分词后的文档
    bm25 = BM25Okapi(tokenized_docs)
    return bm25

# 检索函数
def search(query, bm25, documents, top_n=3):
    tokenized_query = tokenize(query)
    # 计算每个文档的BM25得分
    scores = bm25.get_scores(tokenized_query)
    
    # 获取得分最高的文档
    top_docs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_n]
    results = [(documents[idx], score) for idx, score in top_docs]
    return results

# 主函数
# file_path = '/home/liziang/tx/demo-chatbot-develop/data/names.txt'
# documents = load_documents(file_path)
# bm25 = build_inverted_index(documents)

# 测试检索
# query = "缕金百蝶穿花大红洋缎窄褃袄"
# results = search(query, bm25, documents)
# for doc, score in results:
#     print(f"Document: {doc}", f"Score: {score}")
# text = '大红皮球红缔地女衫'
# token_res = tokenize(text)
# print(token_res)
