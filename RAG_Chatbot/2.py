####################################################################################
#  source code from: ITRI_EOSL R5_llama3 Test @rachelliu
#
#  reference from github:  https://github.com/R300-AI/ITRI_Ollama_RAG.git
#
#
#  pip install Ollama  langchain langchain_openai rich
#  Ollama run llama3, phi3, llama2, mistral
#  FY112_GPT310--------code
#  ssh rachelliu@140.96.98.113
#  conda activate FY112_GPT310
#  cd 2024_llama3
#  python R500_llama3_RAG_NTTMES_inf.py
#  python R500_llama3_RAG_NTTMES_inf.py --gpt phi3
####################################################################################

import argparse
import datetime
import os
import random
import string
import time
from datetime import datetime
from datetime import timedelta

import numpy as np
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import speech_recognition as sr

def setinfo(c_logPath='r500.txt',type='info',text=''):
    """
       寫入日誌，格式化為 [時間戳]|[類型]|[文字內容]
       :param c_logPath: 日誌文件路徑
       :param type: 記錄類型 (e.g., info, Q, A)
       :param text: 記錄的內容
   """
    tt2 = str(datetime.now().strftime("%Y%m%d_%H:%M:%S.%f")) #取到秒鐘
    #print("{:20s}|{:>16s}|{}".format(tt2,str(type).upper(),text))
    with open(c_logPath, 'a+', encoding='utf-8') as f:
        f.write("{:20s}|{:>16s}|{}\n".format(tt2, str(type).upper(), text))

def genguidtxt():
    """
        生成基於時間與隨機字母和數字的唯一文件名。
        格式: yyyy-mm-dd-HH-MM-隨機字符串.txt
    """
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d-%H-%M")
    random_number = random.randint(100000, 999999)
    # 生成四位數字
    digits = ''.join(random.choices(string.digits, k=4))

    # 生成兩位字母
    letters = ''.join(random.choices(string.ascii_letters, k=2))

    # 將數字和字母混合打亂
    random_str = ''.join(random.sample(digits + letters, 6))

    # 组合時間+亂數
    result = f"{time_str}-{random_str}.txt"
    #print(result)
    return result
# 初始化Ollama模型
# 創建文件鏈，將llm和提示模板結合
def prompt_template():
    # 設定提示模板，將系統和使用者的提示組合
    prompt = ChatPromptTemplate.from_messages([
        # ('system', 'Answer the user\'s questions in Traditional Chinese (including proper noun, name(if it\'s a Chinese name)), based on the context provided below:\n\n{context} '),
        ('system',
         '你是一位心理諮商助理，善於傾聽使用者的問題，並提供關懷與建議:\n\n{context}'),
        ('user', 'Question: {input} #zh-tw'),
    ])
    return prompt

def load_trained_db(DB_FAISS_PATH="./NTTPDF_db"):
    """
    加載FAISS向量資料庫，處理加載錯誤。
    """
    # embeddings = OllamaEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    try:
        vectordb = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True)
        # 將向量資料庫設為檢索器
        retriever = vectordb.as_retriever()
    except FileNotFoundError:
        print(f"{DB_FAISS_PATH} 的數據路徑錯誤")
        return 0,'Err001'
    except Exception as e:
        print(f"{e}讀取db錯誤")
        return 0,f'Err002-{e}'
    return 1,retriever
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("請開始說話...")
        try:
            audio = recognizer.listen(source, timeout=10)  # 偵聽語音，超時10秒
            text = recognizer.recognize_google(audio, language='zh-TW')  # 將語音轉為文字
            print(f"您說的是: {text}")
            return text
        except sr.UnknownValueError:
            print("無法辨識語音")
            return "e"
        except sr.RequestError as e:
            print(f"無法請求結果; {e}")
        return "e"
def main(config):
    prompt=prompt_template()
    llm = Ollama(model=config.gpt)
    document_chain = create_stuff_documents_chain(llm, prompt)
    f,retriever=load_trained_db(DB_FAISS_PATH=config.m)
    if(config.islog):
        if(not os.path.exists(config.logpath)): os.mkdir(config.logpath)
        txtfile = genguidtxt()
        c_logPath = os.path.join(config.logpath, txtfile)
        setinfo(c_logPath=c_logPath,text='initialized...config={0}'.format(config))

    # 創建檢索鏈，將檢索器和文件鏈結合
    if(f==1):
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        context = ['Assuming you are an expert in EXC-MES system operations and factory communications']
        # input_text = input('>>> 請提問... (輸入 -1 表示結束)\n')
        while True:
            input_text = speech_to_text()
            if input_text and input_text != "e":  # 如果成功辨識（非空字串）
                break
            print("無法辨識，請再試一次。")
        index = 1
        while input_text.lower() != '-1':
            t1=time.time()
            response = retrieval_chain.invoke({
                'input': input_text,
                'context': context
            })
            ans = response['answer']
            doc = response['context']
            ###################################
            #把response給print出來~
            ###################################
            print(ans)

            if(config.isref): print(doc)
            t2=time.time()
            td = timedelta(seconds=np.round(t2-t1,2))
            print(f'第{index}次訪問結束: 耗時 hh:mm:ss:fff ={td}')
            if(config.islog):
                setinfo(c_logPath=c_logPath,type='Q',text=f'{index:03}-{input_text}')
                setinfo(c_logPath=c_logPath,type='A',text=f'{index:03}-{ans}')
                if(config.isref):
                    setinfo(c_logPath=c_logPath,type='doc',text=f'{index:03}-{doc}')
                setinfo(c_logPath=c_logPath,type='time',text=f'{index:03}-{td}')

            while True:
                input_text = speech_to_text()
                if input_text and input_text != "e":  # 如果成功辨識（非空字串）
                    break
                print("無法辨識，請再試一次。")
            index +=1
if __name__=='__main__':
    """
        根據命令行參數初始化模型與向量資料庫，啟動用戶交互模式。
    """
    parser=argparse.ArgumentParser()
    parser.add_argument('--m', type=str, default="D:/rag/train_output_FAISS")
    parser.add_argument('--gpt', type=str, default='llama3.1', choices=['llama3','phi3','mistral', 'llama2'])
    parser.add_argument('--isref', type=int, default=0, choices=[0,1])
    parser.add_argument('--islog', type=int, default=1, choices=[0,1])
    parser.add_argument('--logpath', type=str, default=r"./nttlog")
    config = parser.parse_args()
    main(config)
