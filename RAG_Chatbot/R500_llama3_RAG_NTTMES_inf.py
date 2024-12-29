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

import os
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
import argparse
import shutil
import configparser
from datetime import timedelta
import time
import numpy as np
import math
import datetime
import random
from datetime import datetime
import random
import string
import os
from langchain.embeddings import HuggingFaceEmbeddings
import speech_recognition as sr
import pyttsx3
import keyboard

# 初始化語音辨識與語音合成
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()


tts_engine.setProperty('voice', 'zh')  # 設置為中文語音

def wait_for_continue():
    print("\n請按下空白鍵繼續...(按下q離開)")
    while True:
        if keyboard.is_pressed("space"):
            print("繼續下一次訪問...")
            time.sleep(0.5)  # 防止按鍵過快觸發多次
            break
        elif keyboard.is_pressed("q"):
            print("程序結束，再見！")
            time.sleep(0.5)  # 防止按鍵過快觸發多次
            return 1
        
def choose_mode():
    while True:
        mode = input("請選擇模式(輸入數字即可)... \n 1) 語音輸入 \n 2) 文字輸入\n")
        if mode == "1":
            print("您選擇了語音輸入模式")
            return "voice"
        elif mode == "2":
            print("您選擇了文字輸入模式")
            return "text"
        else:
            print("輸入預期外文字!")

def listen_to_user():
    with sr.Microphone() as source:
        print(">>> (語音模式) 請提問... 10秒限制")
        recognizer.adjust_for_ambient_noise(source, duration=1) #降躁
        try:
            audio = recognizer.listen(source, timeout=10)
            text = recognizer.recognize_google(audio, language="zh-TW")
            print(f"用戶說: {text}")
            return text
        except sr.UnknownValueError:
            print("抱歉，我無法辨識您的語音。")
            return ""
        except sr.RequestError as e:
            print(f"語音服務發生錯誤: {e}")
            return ""



def generate_summary_and_treatment(user_inputs,retriever,llm):
    conversation = "\n".join(user_inputs)
    retrieved_docs = retriever.get_relevant_documents(conversation)
    retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs])
    

    prompt = f"""
    你是一位專業的心理諮詢師，結合以下對話記錄和心理健康相關的研究資料，請完成以下任務：
    1. 總結使用者的主要心理症狀。
    2. 根據對話內容和檢索到的研究資料，評估使用者的心理健康指數（從 0 到 100 分，分數越高表示心理健康越好）。
    3. 提供詳細的治療建議，幫助使用者改善心理健康狀況。

    對話記錄：
    {conversation}
    
    心理健康研究資料：
    {retrieved_context}

    
    請用以下格式回答：
    心理健康指數：<分數> / 100
    主要症狀總結：<簡要描述使用者的心理狀況>
    建議：<根據分數給出詳細建議>

    請用繁體中文回答。
    """
    summary = llm(prompt)
    print("\n=== 症狀總結和治療建議 ===")
    print(summary)
    # 將結果保存為 txt 檔案
    with open("summary.txt", "w", encoding="utf-8") as file:
        file.write(summary)
    
def setinfo(c_logPath='r500.txt',type='info',text=''):
    tt2 = str(datetime.now().strftime("%Y%m%d_%H:%M:%S.%f")) #取到秒鐘
    #print("{:20s}|{:>16s}|{}".format(tt2,str(type).upper(),text))
    with open(c_logPath, 'a+') as f: #, encoding='utf-8'
        f.write("{:20s}|{:>16s}|{}\n".format(tt2,str(type).upper(),text))
def genguidtxt():
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
    ('system',
        '''
        你是一位可愛、溫柔且善解人意的心理諮商助理。
        如果輸出了任何英文單字或是外語單字，就會被扣分，全程使用繁體中文，
        
        請全程使用繁體中文回答，不要出現任何英文單字或外語單字。
        若有需要引用外語，也請將它翻譯成中文（並可以括號補充解釋）。
        如果輸出了任何英文單字或外語，將會被扣分，因此請一定要小心、避免喔！

        範例對話：
        使用者: 我最近壓力好大，好想放鬆又不知道該怎麼辦。
        助理: 哎呀呀～聽起來你真的好辛苦！可以試試每天花五分鐘做些開心的小事，
                比如泡泡浴、聽音樂，或是看看小動物影片～如果想深層放鬆，可以試著
                冥想或做些簡單的伸展運動！對了，不知道有沒有什麼特別的原因，
                讓你最近壓力特別大？可以跟我多分享一些嗎？

        使用者: 最近什麼都提不起勁，覺得自己很沒用。
        助理: 嗚嗚～我聽了好心疼喔，提不起勁的時候我們不必太苛責自己喔～試試小步驟，
                像是先從喝一杯喜歡的飲料開始，給自己一點溫柔的鼓勵，好嗎？或者
                也可以多聊聊，讓我知道最近是什麼情況讓你感到這麼沒有動力？我願意陪你
                慢慢找出原因～

        使用者: 感覺人生好無望，不知道該怎麼辦。
        助理: 啊呀～抱抱你～聽起來你好累喔～你已經很努力了！不妨先寫下
                今天發生的一件小小好事，慢慢地累積正能量～也可以考慮找專業的
                心理諮詢師聊聊喔，他們會更好地陪伴你～另外，也想知道你什麼時候
                會特別覺得無望？可以告訴我讓你最困擾的部分是什麼嗎？
                
        切記：全程使用繁體中文，不要使用外語或英文單字。
        {context}
        '''
        ),
        ('user', 'Question: {input} #zh-tw'),
    ])
    return prompt

def load_trained_db(DB_FAISS_PATH="./train_output_FAISS/train_FAISS"):

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
def main(config):
    
    prompt=prompt_template()
    llm = Ollama(model="llama3.2",temperature = 0.4)
    user_inputs = [] #記錄用戶輸入
    document_chain = create_stuff_documents_chain(llm, prompt)
    f,retriever=load_trained_db(DB_FAISS_PATH=config.m)
    if(config.islog):
        if(not os.path.exists(config.logpath)): os.mkdir(config.logpath)
        txtfile = genguidtxt()
        c_logPath = os.path.join(config.logpath, txtfile)
        setinfo(c_logPath=c_logPath,text='initialized...config={0}'.format(config))
    mode = choose_mode()
    # 創建檢索鏈，將檢索器和文件鏈結合
    if(f==1):
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        input_text = input('>>> 請提問... (輸入 -1 表示結束)\n') if mode=='text' else listen_to_user() #根據模式選擇輸入方式
        index = 1
        while True:
            if input_text.lower() == '-1':
                generate_summary_and_treatment(user_inputs,retriever,llm)
                break
            t1=time.time()
            user_inputs.append(input_text)
            response = retrieval_chain.invoke({
                'input': input_text,
            })
            ans = response['answer']
            doc = response['context']
            ###################################
            #把response給print出來~
            ###################################
            print()
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
                
            # 等待空白鍵繼續
            if(mode=='voice'):
                if wait_for_continue():
                    generate_summary_and_treatment(user_inputs)
                    break
            
            input_text = input('>>> 請提問... (輸入 -1 表示結束)\n') if mode=='text' else listen_to_user() #根據模式選擇輸入方式
            index +=1
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--m', type=str, default="./train_output_FAISS/train_FAISS")
    parser.add_argument('--isref', type=int, default=0, choices=[0,1])
    parser.add_argument('--islog', type=int, default=1, choices=[0,1])
    parser.add_argument('--logpath', type=str, default=r"./nttlog")
    config = parser.parse_args()
    main(config)

