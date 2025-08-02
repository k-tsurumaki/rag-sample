import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
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

def load_prompt_template(template_type="detailed"):
    """プロンプトテンプレートをファイルから読み込む"""
    
    try:
        # prompt_templates.txtファイルを読み込み
        with open("prompt_templates.txt", "r", encoding="utf-8") as f:
            content = f.read()
        
        # セクションごとに分割してテンプレートを抽出
        templates = {}
        
        # BASICテンプレートを抽出
        if "## [BASIC]" in content:
            basic_start = content.find("## [BASIC]") + len("## [BASIC]")
            basic_end = content.find("## [DETAILED]")
            if basic_end == -1:
                basic_end = content.find("## [TECHNICAL]")
            if basic_end == -1:
                basic_end = len(content)
            templates["basic"] = content[basic_start:basic_end].strip()
        
        # DETAILEDテンプレートを抽出
        if "## [DETAILED]" in content:
            detailed_start = content.find("## [DETAILED]") + len("## [DETAILED]")
            detailed_end = content.find("## [TECHNICAL]")
            if detailed_end == -1:
                detailed_end = len(content)
            templates["detailed"] = content[detailed_start:detailed_end].strip()
        
        # TECHNICALテンプレートを抽出
        if "## [TECHNICAL]" in content:
            technical_start = content.find("## [TECHNICAL]") + len("## [TECHNICAL]")
            templates["technical"] = content[technical_start:].strip()
        
        print(f"プロンプトテンプレートをファイルから読み込みました: {list(templates.keys())}")
        
        # 指定されたテンプレートを取得
        template = templates.get(template_type)
        if not template:
            print(f"警告: '{template_type}'テンプレートが見つかりません。detailedを使用します。")
            template = templates.get("detailed", templates.get("basic", ""))
        
        if not template:
            raise FileNotFoundError("有効なテンプレートが見つかりません")
            
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
    except FileNotFoundError:
        print("prompt_templates.txtが見つかりません。デフォルトテンプレートを使用します。")
        
        # フォールバック: デフォルトテンプレート
        fallback_templates = {
            "basic": """
以下の文書を参考に質問に答えてください:

{context}

質問: {question}

回答:
""",
            "detailed": """
あなたは専門的な質問応答アシスタントです。以下のルールに従って回答してください：

1. 提供された文書の内容のみを基に回答する
2. 文書に記載されていない情報は推測や創作しない
3. 回答は正確で分かりやすく、日本語で行う
4. 不明な点がある場合は「提供された文書では確認できません」と明記する
5. 可能な限り具体例を含めて説明する

以下の文書を参考に、質問に対して正確で詳細な回答を提供してください。

参考文書:
{context}

質問: {question}

回答: 提供された文書に基づいて回答します。
""",
            "technical": """
あなたは技術文書を専門とするAIアナリストです。以下の要件に従って回答してください：

**技術回答の要件:**
1. 技術的な正確性を最優先とする
2. 具体的なライブラリ名、フレームワーク名を含める
3. メリット・デメリットを明確に説明する
4. 関連技術との比較や位置づけを説明する
5. 文書にない情報は「文書では言及されていません」と明記する

**技術文書:**
{context}

**技術的質問:** {question}

**技術的回答:**
文書の内容を基に、技術的観点から詳細に回答します。
"""
        }
        
        template = fallback_templates.get(template_type, fallback_templates["detailed"])
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    except Exception as e:
        print(f"テンプレート読み込み中にエラーが発生しました: {e}")
        # 最小限のフォールバックテンプレート
        basic_template = """
以下の文書を参考に質問に答えてください:

{context}

質問: {question}

回答:
"""
        return PromptTemplate(
            template=basic_template,
            input_variables=["context", "question"]
        )

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

def create_qa_chain(vectorstore, prompt_type="detailed"):
    """QAチェーンを作成（Ollamaを使用、カスタムプロンプト対応）"""
    try:
        # カスタムプロンプトテンプレートを読み込み
        prompt_template = load_prompt_template(prompt_type)
        
        # Ollamaローカルモデルを初期化
        llm = OllamaLLM(
            model="mistral",  # または "llama3.2:1b", "codellama", "mistral"など
            base_url="http://localhost:11434",  # Ollamaのデフォルトポート
            temperature=0.3,
            num_predict=512,
        )
        
        print("Ollamaモデルを初期化しました")
        print(f"プロンプトテンプレート '{prompt_type}' を適用しました")
        
        # RetrievalQAチェーンを作成（カスタムプロンプト使用）
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            verbose=True,
            chain_type_kwargs={
                "prompt": prompt_template,
                "document_variable_name": "context"
            }
        )
        
        return qa_chain
        
    except Exception as e:
        print(f"Ollamaモデルの初期化に失敗しました: {e}")
        print("代替としてHuggingFaceモデルを使用します...")
        
        # フォールバック: HuggingFace pipelineを使用
        prompt_template = load_prompt_template(prompt_type)
        
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
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt_template,
                "document_variable_name": "context"
            }
        )
        
        return qa_chain

def show_template_preview():
    """プロンプトテンプレートのプレビューを表示"""
    print("=== プロンプトテンプレートプレビュー ===")
    
    for template_type in ["basic", "detailed", "technical"]:
        print(f"\n--- {template_type.upper()} テンプレート ---")
        try:
            template = load_prompt_template(template_type)
            # サンプルデータでテンプレートを表示
            sample_context = "Pythonは高水準プログラミング言語です..."
            sample_question = "Pythonについて教えて"
            
            formatted = template.format(context=sample_context, question=sample_question)
            print(formatted[:300] + "..." if len(formatted) > 300 else formatted)
        except Exception as e:
            print(f"エラー: {e}")
        print("-" * 50)

def full_rag_demo():
    """完全なRAGデモ（Ollamaを使用した回答生成）"""
    print("=== 完全なRAGデモ（Ollama + 検索 + 回答生成） ===")
    
    try:
        # プロンプトテンプレートの選択
        print("\nプロンプトテンプレートを選択してください:")
        print("1. basic - 基本的なプロンプト")
        print("2. detailed - 詳細なルール付きプロンプト（推奨）")
        print("3. technical - 技術文書特化プロンプト")
        
        prompt_choice = input("選択してください (1/2/3) [デフォルト: 2]: ").strip()
        
        prompt_types = {"1": "basic", "2": "detailed", "3": "technical"}
        prompt_type = prompt_types.get(prompt_choice, "detailed")
        
        print(f"'{prompt_type}' プロンプトテンプレートを使用します\n")
        
        # ドキュメントを読み込み
        documents = load_documents()
        print(f"読み込んだドキュメント数: {len(documents)}")
        
        # ベクトルストアを作成
        vectorstore = create_vector_store(documents)
        print("ベクトルストアを作成しました")
        
        # QAチェーンを作成
        qa_chain = create_qa_chain(vectorstore, prompt_type)
        print("QAチェーンを作成しました")
        
        # 質問例
        queries = [
            "Pythonの機械学習について教えて",
            "DevOpsでPythonはどのように活用されていますか？",
            "東京で一番おいしいラーメン屋さんは？"
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
                
                print("\n" + "="*80)
                
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
        # 利用可能なモードを表示
        print("使用可能なデモモードを選択してください:")
        print("1. シンプルデモ（検索のみ）")
        print("2. 完全なRAGデモ（Ollama + 回答生成）")
        print("3. プロンプトテンプレートプレビュー")
        
        mode = input("選択してください (1/2/3): ").strip()
        
        if mode == "3":
            show_template_preview()
        elif mode == "2":
            print("\nOllamaを使用した完全なRAGデモを実行します...")
            print("注意: Ollamaが起動していることを確認してください（ollama serve）")
            full_rag_demo()
        else:
            print("\nシンプルなRAGデモを実行します...")
            simple_rag_demo()
        
        print("\n=== RAGサンプルが正常に実行されました ===")
        if mode == "1":
            print("完全なRAGデモを試すには、以下の手順でOllamaを設定してください：")
            print("1. Ollamaをインストール: https://ollama.com/")
            print("2. モデルをダウンロード: ollama pull mistral")
            print("3. Ollamaサーバーを起動: ollama serve")
        elif mode == "3":
            print("プロンプトテンプレートはprompt_templates.txtファイルで編集できます。")
        
    except KeyboardInterrupt:
        print("\n処理が中断されました。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
