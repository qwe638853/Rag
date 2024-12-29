# ITRI_Ollama_RAG
NTT_local_MES_demo_lite

# 1.install ollama ![image](https://github.com/R300-AI/ITRI_Ollama_RAG/blob/main/image/llama.png)
1 . install your [ollama](https://ollama.com/download)   based on your operating system.   
2 . install llm. llama2 (3.8GB), llama3 (4.7GB), phi3(2.4GB).  
```
     ollama run llama2
```
```
     ollama run llama3
```
```
     ollama run phi3
```

3 . check ollama is installated and runing
     
```
     http://localhost:11434
```
![image](https://github.com/R300-AI/ITRI_Ollama_RAG/blob/main/image/localhost.png)
```
     ollama list
```
![image](https://github.com/R300-AI/ITRI_Ollama_RAG/blob/main/image/ollamalist.png)
# 2.install virtual environment and package

1 . create conda evn
```
     conda create --name myenv python=3.10
```
2 . pip install requirement 
    follow the setup [file](https://github.com/R300-AI/ITRI_Ollama_RAG/blob/main/requirments.yml).  

# 3.add embedded model file and folder, download the code
1 . please contact ITRI's fellow.  
2 . copy the folder and *.py file.  
├── db                                            # embbeded folder  
│     ├── index.faiss                               # Facebook AI Similarity Search (FAISS) file  
│     ├── index.pkl                                 # pickle serialization file  
├── R500_llama3_RAG_NTTMES_inf.py                 # inference python file  
├── ...                                           # other files  
└── README.md  

# 4. run the inference model
```
     conda activate myenv
```
```
    python R500_llama3_RAG_NTTMES_inf.py.py
```
-testing results as follows: 
![image](https://github.com/R300-AI/ITRI_Ollama_RAG/blob/main/image/testresult.png)
