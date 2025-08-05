#!/usr/bin/env python3
"""
StreamlineFramework ハイブリッドRAGシステム
- 通常のRAG回答生成
- キーワード抽出
- LangExtractによるソース位置特定
"""

import argparse
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import google.generativeai as genai
import langextract as lx
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 環境変数を読み込み
load_dotenv()

# 警告を抑制
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class SourceLocation:
    """ソース位置情報"""

    text: str  # 引用されたテキスト
    start_char: int  # 開始文字位置
    end_char: int  # 終了文字位置
    keyword: str  # 対応するキーワード
    category: str  # 抽出カテゴリ


@dataclass
class HybridRAGResult:
    """ハイブリッドRAGの結果"""

    question: str  # 元の質問
    answer: str  # RAG回答
    extracted_keywords: List[str]  # 抽出されたキーワード
    source_locations: List[SourceLocation]  # ソース位置情報
    metadata: Dict[str, Any]  # メタデータ

    def to_json(self) -> str:
        """JSON文字列に変換"""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


class PromptLoader:
    """プロンプトテンプレートローダー"""

    def __init__(self, prompt_file: str = "hybrid_rag_prompts.txt"):
        self.prompt_file = prompt_file
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        """プロンプトファイルから各プロンプトを読み込み"""
        try:
            with open(self.prompt_file, "r", encoding="utf-8") as f:
                content = f.read()

            prompts = {}
            current_prompt = None
            current_content = []

            for line in content.split("\n"):
                if line.startswith("## ") and line.endswith("_PROMPT"):
                    if current_prompt:
                        prompts[current_prompt] = "\n".join(current_content).strip()
                    current_prompt = line[3:]  # "## " を除去
                    current_content = []
                elif current_prompt and not line.startswith("#"):
                    current_content.append(line)

            # 最後のプロンプトを追加
            if current_prompt:
                prompts[current_prompt] = "\n".join(current_content).strip()

            return prompts

        except FileNotFoundError:
            raise FileNotFoundError(
                f"プロンプトファイル '{self.prompt_file}' が見つかりません"
            )
        except Exception as e:
            raise Exception(f"プロンプトファイルの読み込みエラー: {e}")

    def get_prompt(self, prompt_name: str) -> str:
        """指定されたプロンプトを取得"""
        if prompt_name not in self.prompts:
            raise KeyError(f"プロンプト '{prompt_name}' が見つかりません")
        return self.prompts[prompt_name]


class DocumentLoader:
    """参考資料ローダー（LangChain対応）"""

    def __init__(self, document_file: str = "sample_documents.txt"):
        self.document_file = document_file

    def load_documents(self) -> str:
        """参考資料を文字列として読み込み（従来互換）"""
        try:
            with open(self.document_file, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"参考資料ファイル '{self.document_file}' が見つかりません"
            )
        except Exception as e:
            raise Exception(f"参考資料の読み込みエラー: {e}")

    def load_documents_as_langchain(self) -> List[Document]:
        """LangChain Document形式で読み込み"""
        try:
            with open(self.document_file, "r", encoding="utf-8") as f:
                content = f.read()

            # テキストをセクションに分割
            sections = content.split("\n\n")
            documents = []

            for i, section in enumerate(sections):
                if section.strip():
                    doc = Document(
                        page_content=section.strip(),
                        metadata={
                            "source": self.document_file,
                            "section": i,
                            "section_type": "framework_documentation",
                        },
                    )
                    documents.append(doc)

            return documents

        except FileNotFoundError:
            raise FileNotFoundError(
                f"参考資料ファイル '{self.document_file}' が見つかりません"
            )
        except Exception as e:
            raise Exception(f"参考資料の読み込みエラー: {e}")


class GeminiClient:
    """Gemini API クライアント（LangChain対応）"""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY環境変数が設定されていません")

        self.model_name = model_name

        # 直接のGemini APIクライアント（LangExtract用）
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)

        # LangChain用のLLM（Geminiの直接実装）
        self.langchain_llm = self._create_langchain_llm()

    def _create_langchain_llm(self):
        """LangChain用のLLMを作成（Gemini直接実装）"""
        try:
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=0.1,
                max_output_tokens=2048,
            )
        except ImportError:
            # langchain_google_genaiが利用できない場合のフォールバック
            return None

    def generate_response(self, prompt: str) -> str:
        """Geminiで応答を生成（直接API）"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=2048,
                ),
            )
            return response.text
        except Exception as e:
            raise Exception(f"Gemini API エラー: {e}")

    def get_langchain_llm(self):
        """LangChain用のLLMを取得"""
        if self.langchain_llm is None:
            # フォールバック実装: 簡単なラッパー
            class GeminiWrapper:
                def __init__(self, gemini_client):
                    self.gemini_client = gemini_client

                def __call__(self, prompt):
                    return self.gemini_client.generate_response(prompt)

                def invoke(self, prompt):
                    if hasattr(prompt, "to_string"):
                        prompt_str = prompt.to_string()
                    else:
                        prompt_str = str(prompt)
                    return self.gemini_client.generate_response(prompt_str)

            return GeminiWrapper(self)
        return self.langchain_llm


class LangExtractClient:
    """LangExtract クライアント"""

    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client
        self.examples = self._create_examples()

    def _create_examples(self):
        """LangExtract用の例を作成"""
        return [
            lx.data.ExampleData(
                text="StreamlineFrameworkの@Controllerアノテーションは、Webコントローラクラスを定義するために使用されます。パフォーマンスは従来比30%向上しています。",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="annotation_name",
                        extraction_text="@Controller",
                        attributes={
                            "category": "アノテーション",
                            "purpose": "Webコントローラ定義",
                        },
                    ),
                    lx.data.Extraction(
                        extraction_class="performance_metric",
                        extraction_text="30%向上",
                        attributes={
                            "category": "性能指標",
                            "metric_type": "パフォーマンス改善",
                        },
                    ),
                ],
            )
        ]

    def extract_keyword_sources(
        self, context: str, keywords: List[str], prompt: str
    ) -> List[SourceLocation]:
        """キーワードのソース位置を抽出"""
        try:
            print(
                f"LangExtractでキーワードソース抽出開始: {len(keywords)}個のキーワード"
            )

            # キーワードを含めたプロンプトを作成
            full_prompt = f"{prompt}\n\n対象キーワード: {', '.join(keywords)}"

            result = lx.extract(
                text_or_documents=context,
                prompt_description=full_prompt,
                examples=self.examples,
                extraction_passes=1,
                max_workers=1,
                max_char_buffer=800,
                model_id=self.gemini_client.model_name,
                api_key=self.gemini_client.api_key,
                temperature=0.1,
            )

            source_locations = []
            for extraction in result.extractions:
                source_location = SourceLocation(
                    text=extraction.extraction_text,
                    start_char=getattr(extraction, "start_char", 0),
                    end_char=getattr(
                        extraction, "end_char", len(extraction.extraction_text)
                    ),
                    keyword=self._find_matching_keyword(
                        extraction.extraction_text, keywords
                    ),
                    category=extraction.extraction_class,
                )
                source_locations.append(source_location)

            print(f"LangExtract抽出完了: {len(source_locations)}件のソース位置")
            return source_locations

        except Exception as e:
            raise Exception(f"LangExtract抽出エラー: {e}")

    def _find_matching_keyword(self, extracted_text: str, keywords: List[str]) -> str:
        """抽出されたテキストに最も関連するキーワードを特定"""
        for keyword in keywords:
            if (
                keyword.lower() in extracted_text.lower()
                or extracted_text.lower() in keyword.lower()
            ):
                return keyword
        return "関連キーワード不明"


class LangChainRAGEngine:
    """LangChainベースのRAGエンジン"""

    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client
        self.vectorstore = None
        self.qa_chain = None

    def setup_vectorstore(self, documents: List[Document]):
        """ベクトルストアを設定"""
        try:
            # テキストスプリッターでドキュメントを分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )

            splits = text_splitter.split_documents(documents)

            # 埋め込みモデルを初期化
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )

            # Chromaベクトルストアを作成
            persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_directory,
            )

            print(f"✓ ベクトルストア作成完了: {len(splits)}個のチャンク")

        except Exception as e:
            raise Exception(f"ベクトルストア設定エラー: {e}")

    def setup_qa_chain(self, prompt_template: str):
        """QAチェーンを設定"""
        try:
            if not self.vectorstore:
                raise ValueError("ベクトルストアが設定されていません")

            # プロンプトテンプレートを作成
            prompt = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

            # LangChain LLMを取得
            llm = self.gemini_client.get_langchain_llm()

            # RetrievalQAチェーンを作成
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": 3}  # 上位3つの関連文書を取得
                ),
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": prompt,
                    "document_variable_name": "context",
                },
            )

            print("✓ QAチェーン設定完了")

        except Exception as e:
            raise Exception(f"QAチェーン設定エラー: {e}")

    def query(self, question: str) -> Dict[str, Any]:
        """質問に対してRAG検索を実行"""
        try:
            if not self.qa_chain:
                raise ValueError("QAチェーンが設定されていません")

            result = self.qa_chain.invoke({"query": question})

            return {
                "answer": result["result"],
                "source_documents": result["source_documents"],
                "retrieved_context": "\n\n".join(
                    [doc.page_content for doc in result["source_documents"]]
                ),
            }

        except Exception as e:
            raise Exception(f"RAG検索エラー: {e}")


class HybridRAGSystem:
    """ハイブリッドRAGシステムメインクラス（LangChain対応）"""

    def __init__(self):
        self.prompt_loader = PromptLoader()
        self.document_loader = DocumentLoader()
        self.gemini_client = GeminiClient()
        self.langextract_client = LangExtractClient(self.gemini_client)

        # LangChainベースのRAGエンジンを初期化
        self.rag_engine = LangChainRAGEngine(self.gemini_client)

        # 参考資料を事前読み込み（両形式）
        self.context = (
            self.document_loader.load_documents()
        )  # 従来形式（LangExtract用）
        self.documents = (
            self.document_loader.load_documents_as_langchain()
        )  # LangChain形式

        # LangChainベースのRAGを設定
        self._setup_langchain_rag()

        print("✓ ハイブリッドRAGシステム初期化完了")
        print(f"✓ Geminiモデル: {self.gemini_client.model_name}")
        print(f"✓ 参考資料読み込み: {len(self.context)}文字")
        print(f"✓ LangChain文書数: {len(self.documents)}個")

    def _setup_langchain_rag(self):
        """LangChainベースのRAGを設定"""
        try:
            # ベクトルストアを設定
            self.rag_engine.setup_vectorstore(self.documents)

            # RAG用のプロンプトテンプレートを取得
            rag_prompt_template = self.prompt_loader.get_prompt("INITIAL_ANSWER_PROMPT")

            # QAチェーンを設定
            self.rag_engine.setup_qa_chain(rag_prompt_template)

        except Exception as e:
            print(f"⚠️ LangChain RAG設定でエラー: {e}")
            print("フォールバック: 従来の方式を使用します")
            self.rag_engine = None

    def process_question(self, question: str) -> HybridRAGResult:
        """質問を処理してハイブリッドRAG結果を生成"""
        print("\n=== 質問処理開始 ===")
        print(f"質問: {question}")

        # Step 1: RAG回答生成
        print("\n1. RAG回答生成中...")
        answer = self._generate_rag_answer(question)
        print(f"✓ 回答生成完了: {len(answer)}文字")

        # Step 2: キーワード抽出
        print("\n2. キーワード抽出中...")
        keywords = self._extract_keywords(answer)
        print(f"✓ キーワード抽出完了: {keywords}")

        # Step 3: ソース位置特定
        print("\n3. ソース位置特定中...")
        source_locations = self._extract_source_locations(keywords)
        print(f"✓ ソース位置特定完了: {len(source_locations)}件")

        # 結果をまとめる
        result = HybridRAGResult(
            question=question,
            answer=answer,
            extracted_keywords=keywords,
            source_locations=source_locations,
            metadata={
                "model": self.gemini_client.model_name,
                "context_length": len(self.context),
                "answer_length": len(answer),
                "keyword_count": len(keywords),
                "source_count": len(source_locations),
            },
        )

        print("\n=== 処理完了 ===")
        return result

    def _generate_rag_answer(self, question: str) -> str:
        """RAG回答を生成（LangChain使用）"""
        if self.rag_engine:
            # LangChainベースのRAGを使用
            try:
                rag_result = self.rag_engine.query(question)
                return rag_result["answer"]
            except Exception as e:
                print(f"⚠️ LangChain RAGでエラー: {e}")
                print("フォールバック: 従来方式を使用")

        # フォールバック: 従来の方式
        prompt_template = self.prompt_loader.get_prompt("INITIAL_ANSWER_PROMPT")
        prompt = prompt_template.format(context=self.context, question=question)
        return self.gemini_client.generate_response(prompt)

    def _extract_keywords(self, answer: str) -> List[str]:
        """回答からキーワードを抽出"""
        prompt_template = self.prompt_loader.get_prompt("KEYWORD_EXTRACTION_PROMPT")
        prompt = prompt_template.format(answer_text=answer)

        response = self.gemini_client.generate_response(prompt)

        # キーワードをパース（カンマ区切りを想定）
        keywords = [kw.strip() for kw in response.split(",") if kw.strip()]
        return keywords[:10]  # 最大10個のキーワードに制限

    def _extract_source_locations(self, keywords: List[str]) -> List[SourceLocation]:
        """キーワードのソース位置を特定"""
        if not keywords:
            return []

        prompt = self.prompt_loader.get_prompt("LANGEXTRACT_SOURCE_PROMPT")
        return self.langextract_client.extract_keyword_sources(
            self.context, keywords, prompt
        )


def main():
    """メイン関数 - コマンドライン処理"""
    parser = argparse.ArgumentParser(
        description="StreamlineFramework ハイブリッドRAGシステム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python enhanced_rag.py                                            # 対話モード（デフォルト）
  python enhanced_rag.py -q "@Controllerアノテーションの使い方を教えてください"  # 単発質問
  python enhanced_rag.py -q "StreamlineORMの特徴は何ですか？"             # 単発質問
  python enhanced_rag.py --interactive                             # 明示的に対話モード
  python enhanced_rag.py -q "質問" --output result.json             # 結果をJSONで保存
        """,
    )

    parser.add_argument(
        "-q", "--question", type=str, help="処理する質問（指定しない場合は対話モード）"
    )

    parser.add_argument(
        "--interactive", action="store_true", help="対話モードで実行（デフォルト）"
    )

    parser.add_argument(
        "--output", type=str, help="結果をJSONファイルに出力（ファイルパス指定）"
    )

    args = parser.parse_args()

    try:
        # システム初期化
        rag_system = HybridRAGSystem()

        # 引数で質問が指定されている場合
        if args.question:
            # 単発質問モード
            result = rag_system.process_question(args.question)
            print_result(result)

            if args.output:
                save_result(result, args.output)

        # 対話モードが明示的に指定されているか、引数がない場合
        elif args.interactive or not any([args.question, args.output]):
            # 対話モード
            print("\n=== StreamlineFramework ハイブリッドRAG 対話モード ===")
            print("質問を入力してください（'exit'で終了）:")

            while True:
                question = input("\n質問 > ").strip()
                if question.lower() in ["exit", "quit", "終了"]:
                    print("システムを終了します。")
                    break

                if not question:
                    continue

                try:
                    result = rag_system.process_question(question)
                    print_result(result)

                    if args.output:
                        save_result(result, args.output)

                except Exception as e:
                    print(f"❌ エラーが発生しました: {e}")
                    sys.exit(1)

        else:
            parser.print_help()
            sys.exit(1)

    except Exception as e:
        print(f"❌ システムエラー: {e}")
        sys.exit(1)


def print_result(result: HybridRAGResult):
    """結果を整形して表示"""
    print(f"\n{'=' * 60}")
    print(f"【質問】{result.question}")
    print(f"{'=' * 60}")

    print("\n【回答】")
    print(result.answer)

    print(f"\n【抽出キーワード】({len(result.extracted_keywords)}個)")
    for i, keyword in enumerate(result.extracted_keywords, 1):
        print(f"  {i}. {keyword}")

    print(f"\n【ソース位置情報】({len(result.source_locations)}件)")
    for i, source in enumerate(result.source_locations, 1):
        print(f"  {i}. キーワード: {source.keyword}")
        print(f"     カテゴリ: {source.category}")
        print(f'     テキスト: "{source.text}"')
        print(f"     位置: {source.start_char}-{source.end_char}文字目")
        print()

    print("【メタデータ】")
    for key, value in result.metadata.items():
        print(f"  {key}: {value}")


def save_result(result: HybridRAGResult, output_file: str):
    """結果をJSONファイルに保存"""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result.to_json())
        print(f"\n✓ 結果を保存しました: {output_file}")
    except Exception as e:
        print(f"❌ ファイル保存エラー: {e}")


if __name__ == "__main__":
    main()
