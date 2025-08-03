import os
import warnings

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
                metadata={"source": "sample_documents.txt", "section": i},
            )
            documents.append(doc)

    return documents


def load_prompt_template():
    """プロンプトテンプレートをファイルから読み込む"""
    # prompt_templates.txtファイルを読み込み
    with open("prompt_templates.txt", "r", encoding="utf-8") as f:
        template_content = f.read().strip()

    print("プロンプトテンプレートをファイルから読み込みました")

    if not template_content:
        raise ValueError("テンプレートファイルが空です")

    return PromptTemplate(
        template=template_content, input_variables=["context", "question"]
    )


def create_vector_store(documents):
    """ベクトルストアを作成"""
    # テキストスプリッターを初期化
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, length_function=len
    )

    # ドキュメントを分割
    splits = text_splitter.split_documents(documents)

    # 埋め込みモデルを初期化（HuggingFaceの無料モデルを使用）
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    # Chromaベクトルストアを作成
    persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=embeddings, persist_directory=persist_directory
    )

    return vectorstore


def create_qa_chain(vectorstore):
    """QAチェーンを作成（Ollamaを使用、カスタムプロンプト対応）"""
    # カスタムプロンプトテンプレートを読み込み
    prompt_template = load_prompt_template()

    # Ollamaローカルモデルを初期化
    llm = OllamaLLM(
        model="mistral",  # または "llama3.2:1b", "codellama", "mistral"など
        base_url="http://localhost:11434",  # Ollamaのデフォルトポート
        temperature=0.3,
        num_predict=512,
    )

    print("Ollamaモデルを初期化しました")
    print("プロンプトテンプレートを適用しました")

    # RetrievalQAチェーンを作成（カスタムプロンプト使用）
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        verbose=True,
        chain_type_kwargs={
            "prompt": prompt_template,
            "document_variable_name": "context", # 検索された文書がテンプレートの{content}部分に挿入される
        },
    )
    return qa_chain


def show_template_preview():
    """使用中のプロンプトテンプレートを表示"""
    print("\n--- 使用中のプロンプトテンプレート ---")
    try:
        template = load_prompt_template()
        # サンプルデータでテンプレートを表示
        sample_context = "Pythonは高水準プログラミング言語です..."
        sample_question = "Pythonについて教えて"

        formatted = template.format(context=sample_context, question=sample_question)
        print(formatted[:300] + "..." if len(formatted) > 300 else formatted)
    except Exception as e:
        print(f"テンプレートの読み込みに失敗しました: {e}")
    print("-" * 40)


def full_rag_demo():
    """RAG（検索拡張生成）のフルデモ"""
    print("\n=== RAG（検索拡張生成）デモ ===")

    # ドキュメントを読み込み
    print("1. ドキュメントを読み込んでいます...")
    documents = load_documents()

    # ベクターストアを作成
    print("2. ベクターストアを作成しています...")
    vectorstore = create_vector_store(documents)

    # QAチェーンを作成
    print("3. QAチェーンを作成しています...")
    qa_chain = create_qa_chain(vectorstore)

    # テンプレートプレビューを表示
    show_template_preview()

    print("\n質問を入力してください（終了するには 'quit' と入力）:")

    while True:
        question = input("\n質問: ")
        if question.lower() == "quit":
            break

        print("\n回答を生成中...")
        result = qa_chain.invoke({"query": question})

        print("\n【回答】")
        print(result["result"])

        # ソース文書を表示
        # if "source_documents" in result and result["source_documents"]:
        #     print("\n【参照文書】")
        #     for i, doc in enumerate(result["source_documents"], 1):
        #         print(f"{i}. {doc.page_content[:200]}...")


def main():
    # 環境変数を読み込み
    load_dotenv()

    print("LangChainを使ったRAGサンプル")
    print("=" * 40)

    try:
        # 利用可能なモードを表示
        print("使用可能なデモモードを選択してください:")
        print("1. 完全なRAGデモ（Ollama + 回答生成）")
        print("2. プロンプトテンプレートプレビュー")

        mode = input("選択してください (1/2): ").strip()

        if mode == "2":
            show_template_preview()
        else:
            print("\nOllamaを使用した完全なRAGデモを実行します...")
            print("注意: Ollamaが起動していることを確認してください（ollama serve）")
            full_rag_demo()

        print("\n=== RAGサンプルが正常に実行されました ===")
        if mode != "2":
            print("Ollamaの設定:")
            print("1. Ollamaをインストール: https://ollama.com/")
            print("2. モデルをダウンロード: ollama pull mistral")
            print("3. Ollamaサーバーを起動: ollama serve")
        else:
            print(
                "プロンプトテンプレートはprompt_templates.txtファイルで編集できます。"
            )

    except KeyboardInterrupt:
        print("\n処理が中断されました。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
