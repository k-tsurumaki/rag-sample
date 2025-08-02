import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.llms import HuggingFacePipeline
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from transformers import pipeline
import warnings

# 警告を無視
warnings.filterwarnings("ignore")

def load_documents():
    """サンプルドキュメントを読み込む"""
    with open("sample_documents.txt", "r", encoding="utf-8") as f:
        content = f.read()
    
    # ドキュメントをセクションに分割
    sections = content.split("\n\n")
    documents = []
    
    for i, section in enumerate(sections):
        if section.strip():
            doc = Document(
                page_content=section.strip(),
                metadata={"source": "sample_documents.txt", "section": i}
            )
            documents.append(doc)
    
    return documents

def create_vector_store(documents):
    """ベクトルストアを作成"""
    # テキストスプリッターを初期化
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    
    # ドキュメントを分割
    splits = text_splitter.split_documents(documents)
    
    # 埋め込みモデルを初期化（HuggingFaceの無料モデルを使用）
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Chromaベクトルストアを作成
    persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return vectorstore

def create_qa_chain(vectorstore):
    """QAチェーンを作成（Ollamaを使用）"""
    try:
        # Ollamaローカルモデルを初期化
        llm = OllamaLLM(
            model="mistral",  # または "llama3.2:1b", "codellama", "mistral"など
            base_url="http://localhost:11434",  # Ollamaのデフォルトポート
            temperature=0.3,
            num_predict=512,
        )
        
        print("Ollamaモデルを初期化しました")
        
        # RetrievalQAチェーンを作成
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            verbose=True
        )
        
        return qa_chain
        
    except Exception as e:
        print(f"Ollamaモデルの初期化に失敗しました: {e}")
        print("代替としてHuggingFaceモデルを使用します...")
        
        # フォールバック: HuggingFace pipelineを使用
        hf_pipeline = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-medium",
            max_length=512,
            temperature=0.7,
            do_sample=True,
            device=-1  # CPUを使用
        )
        
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        
        # RetrievalQAチェーンを作成
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        return qa_chain

def full_rag_demo():
    """完全なRAGデモ（Ollamaを使用した回答生成）"""
    print("=== 完全なRAGデモ（Ollama + 検索 + 回答生成） ===")
    
    try:
        # ドキュメントを読み込み
        documents = load_documents()
        print(f"読み込んだドキュメント数: {len(documents)}")
        
        # ベクトルストアを作成
        vectorstore = create_vector_store(documents)
        print("ベクトルストアを作成しました")
        
        # QAチェーンを作成
        qa_chain = create_qa_chain(vectorstore)
        print("QAチェーンを作成しました")
        
        # 質問例
        queries = [
            "Pythonの機械学習について教えて",
            "Webフレームワークにはどんなものがありますか？",
            "データサイエンスで使われるライブラリは？"
        ]
        
        for query in queries:
            print(f"\n質問: {query}")
            print("=" * 80)
            
            try:
                # QAチェーンで回答を生成
                result = qa_chain.invoke({"query": query})
                
                print("回答:")
                print(result["result"])
                
                print("\n参考文書:")
                for i, doc in enumerate(result["source_documents"], 1):
                    print(f"[文書 {i}] {doc.page_content[:150]}...")
                
            except Exception as e:
                print(f"回答生成中にエラーが発生しました: {e}")
                
    except Exception as e:
        print(f"完全なRAGデモでエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

def simple_rag_demo():
    """シンプルなRAGデモ（LLMなしバージョン）"""
    print("=== シンプルなRAGデモ（検索のみ） ===")
    
    # ドキュメントを読み込み
    documents = load_documents()
    print(f"読み込んだドキュメント数: {len(documents)}")
    
    # ベクトルストアを作成
    vectorstore = create_vector_store(documents)
    print("ベクトルストアを作成しました")
    
    # 検索テスト
    queries = [
        "Pythonの機械学習について教えて",
        "Webフレームワークにはどんなものがありますか？",
        "データサイエンスで使われるライブラリは？"
    ]
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    for query in queries:
        print(f"\n質問: {query}")
        print("-" * 50)
        
        # 関連ドキュメントを検索
        docs = retriever.get_relevant_documents(query)
        
        for i, doc in enumerate(docs, 1):
            print(f"関連文書 {i}:")
            print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
            print()

def main():
    # 環境変数を読み込み
    load_dotenv()
    
    print("LangChainを使ったRAGサンプル")
    print("=" * 40)
    
    try:
        # Ollamaが利用可能かをチェック
        print("使用可能なデモモードを選択してください:")
        print("1. シンプルデモ（検索のみ）")
        print("2. 完全なRAGデモ（Ollama + 回答生成）")
        
        mode = input("選択してください (1/2): ").strip()
        
        if mode == "2":
            print("\nOllamaを使用した完全なRAGデモを実行します...")
            print("注意: Ollamaが起動していることを確認してください（ollama serve）")
            full_rag_demo()
        else:
            print("\nシンプルなRAGデモを実行します...")
            simple_rag_demo()
        
        print("\n=== RAGサンプルが正常に実行されました ===")
        if mode != "2":
            print("完全なRAGデモを試すには、以下の手順でOllamaを設定してください：")
            print("1. Ollamaをインストール: https://ollama.com/")
            print("2. モデルをダウンロード: ollama pull llama3.2:3b")
            print("3. Ollamaサーバーを起動: ollama serve")
        
    except KeyboardInterrupt:
        print("\n処理が中断されました。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
