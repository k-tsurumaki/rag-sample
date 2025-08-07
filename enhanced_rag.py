#!/usr/bin/env python3
"""
StreamlineFramework ハイブリッドRAGシステム
- 通常のRAG回答生成
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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 環境変数を読み込み
load_dotenv()

# 警告を抑制
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class SystemConfig:
    """システム設定"""

    # ファイルパス
    prompt_file: str = "hybrid_rag_prompts.txt"
    document_file: str = "sample_documents.txt"
    chroma_persist_directory: str = "./chroma_db"

    # Ollama設定
    ollama_model: str = "mistral"
    ollama_base_url: str = "http://localhost:11434"
    ollama_temperature: float = 0.3
    ollama_num_predict: int = 512

    # Gemini設定
    gemini_model: str = "gemini-2.5-flash"
    gemini_temperature: float = 0.1
    gemini_top_p: float = 0.8
    gemini_top_k: int = 40
    gemini_max_output_tokens: int = 2048

    # LangExtract設定
    langextract_extraction_passes: int = 1
    langextract_max_workers: int = 1
    langextract_max_char_buffer: int = 800

    # RAG設定
    text_splitter_chunk_size: int = 2000
    text_splitter_chunk_overlap: int = 400
    retriever_k: int = 3
    retriever_fetch_k: int = 6
    retriever_lambda_mult: float = 0.7

    # 埋め込みモデル設定
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # デバッグ設定
    debug: bool = False

    @classmethod
    def from_env(cls) -> "SystemConfig":
        """環境変数から設定を生成"""
        return cls(
            chroma_persist_directory=os.getenv(
                "CHROMA_PERSIST_DIRECTORY", cls.chroma_persist_directory
            ),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_base_url),
            ollama_model=os.getenv("OLLAMA_MODEL", cls.ollama_model),
            gemini_model=os.getenv("GEMINI_MODEL", cls.gemini_model),
            debug=os.getenv("DEBUG", "false").lower() == "true",
        )


@dataclass
class SourceLocation:
    """ソース位置情報"""

    start_char: int  # 開始文字位置
    end_char: int  # 終了文字位置
    keyword: str  # LangExtractで抽出されたテキスト
    category: str  # 抽出カテゴリ
    section: int  # セクション番号


@dataclass
class HybridRAGResult:
    """ハイブリッドRAGの結果"""

    question: str  # 元の質問
    answer: str  # RAG回答
    extracted_keywords: List[
        str
    ]  # LangExtractで抽出されたテキスト（後方互換性のため保持）
    source_locations: List[SourceLocation]  # ソース位置情報
    metadata: Dict[str, Any]  # メタデータ

    def to_json(self) -> str:
        """JSON文字列に変換"""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


class PromptLoader:
    """プロンプトテンプレートローダー"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        """プロンプトファイルから各プロンプトを読み込み"""
        try:
            with open(self.config.prompt_file, "r", encoding="utf-8") as f:
                content = f.read()

            return self._parse_prompts(content)

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"プロンプトファイル '{self.config.prompt_file}' が見つかりません"
            ) from e
        except Exception as e:
            raise Exception(f"プロンプトファイルの読み込みエラー: {e}") from e

    def _parse_prompts(self, content: str) -> Dict[str, str]:
        """プロンプト内容を解析"""
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

    def get_prompt(self, prompt_name: str) -> str:
        """指定されたプロンプトを取得"""
        if prompt_name not in self.prompts:
            available_prompts = list(self.prompts.keys())
            raise KeyError(
                f"プロンプト '{prompt_name}' が見つかりません。"
                f"利用可能なプロンプト: {available_prompts}"
            )
        return self.prompts[prompt_name]


class DocumentLoader:
    """参考資料ローダー（LangChain対応）"""

    def __init__(self, config: SystemConfig):
        self.config = config

    def load_documents(self) -> str:
        """参考資料を文字列として読み込み（従来互換）"""
        try:
            with open(self.config.document_file, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"参考資料ファイル '{self.config.document_file}' が見つかりません"
            ) from e
        except Exception as e:
            raise Exception(f"参考資料の読み込みエラー: {e}") from e

    def load_documents_as_langchain(self) -> List[Document]:
        """LangChain Document形式で読み込み"""
        try:
            content = self.load_documents()
            return self._create_langchain_documents(content)
        except Exception as e:
            raise Exception(f"LangChain文書作成エラー: {e}") from e

    def _create_langchain_documents(self, content: str) -> List[Document]:
        """テキストをLangChain Documentに変換"""
        # 空行2つ以上で区切られたセクションを取得
        sections = content.split("\n\n")
        documents = []

        current_section = 0
        accumulated_content = ""

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # セクションタイトル（短い行）かコンテンツかを判定
            if (
                len(section) < 100
                and not section.endswith(".")
                and not section.endswith("。")
            ):
                # 新しいセクションタイトルの場合、前のセクションがあれば保存
                if accumulated_content:
                    doc = Document(
                        page_content=accumulated_content.strip(),
                        metadata={
                            "source": self.config.document_file,
                            "section": current_section,
                            "section_type": "framework_documentation",
                        },
                    )
                    documents.append(doc)

                # 新しいセクションを開始
                current_section += 1
                accumulated_content = section + "\n\n"
            else:
                # コンテンツを蓄積
                accumulated_content += section + "\n\n"

        # 最後のセクションを追加
        if accumulated_content:
            doc = Document(
                page_content=accumulated_content.strip(),
                metadata={
                    "source": self.config.document_file,
                    "section": current_section,
                    "section_type": "framework_documentation",
                },
            )
            documents.append(doc)

        print(f"作成された文書数: {len(documents)}")
        print("\n=== 全参考資料の内容 ===")
        for i, doc in enumerate(documents):
            print(f"\n--- 文書 {i} (セクション {doc.metadata['section']}) ---")
            print(f"長さ: {len(doc.page_content)} 文字")
            print(f"内容:\n{doc.page_content}")
            print("-" * 80)

        return documents


class GeminiClient:
    """Gemini API クライアント（LangExtract専用）"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.api_key = self._get_api_key()
        self.model = self._initialize_model()

    def _get_api_key(self) -> str:
        """API キーを取得"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY環境変数が設定されていません。"
                "https://aistudio.google.com/app/apikey からAPIキーを取得してください。"
            )
        return api_key

    def _initialize_model(self) -> genai.GenerativeModel:
        """Gemini モデルを初期化"""
        genai.configure(api_key=self.api_key)
        return genai.GenerativeModel(self.config.gemini_model)

    @property
    def model_name(self) -> str:
        """モデル名を取得"""
        return self.config.gemini_model

    def generate_response(self, prompt: str) -> str:
        """Geminiで応答を生成（LangExtract用のみ）"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.gemini_temperature,
                    top_p=self.config.gemini_top_p,
                    top_k=self.config.gemini_top_k,
                    max_output_tokens=self.config.gemini_max_output_tokens,
                ),
            )
            return response.text
        except Exception as e:
            raise Exception(f"Gemini API エラー: {e}") from e


class OllamaClient:
    """Ollama クライアント（RAG用）"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.llm = self._initialize_llm()
        print(f"✓ Ollamaモデル初期化: {self.config.ollama_model}")

    def _initialize_llm(self) -> OllamaLLM:
        """Ollama LLMを初期化"""
        return OllamaLLM(
            model=self.config.ollama_model,
            base_url=self.config.ollama_base_url,
            temperature=self.config.ollama_temperature,
            num_predict=self.config.ollama_num_predict,
        )

    @property
    def model_name(self) -> str:
        """モデル名を取得"""
        return self.config.ollama_model

    def generate_response(self, prompt: str) -> str:
        """Ollamaで応答を生成"""
        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            raise Exception(f"Ollama API エラー: {e}") from e

    def get_langchain_llm(self) -> OllamaLLM:
        """LangChain用のLLMを取得"""
        return self.llm


class LangExtractClient:
    """LangExtract クライアント"""

    def __init__(self, config: SystemConfig, gemini_client: GeminiClient):
        self.config = config
        self.gemini_client = gemini_client
        self.examples = self._create_examples()

    def _create_examples(self) -> List[lx.data.ExampleData]:
        """LangExtract用の例を作成（RAG参考資料から重要フレーズを抽出）"""
        return [
            lx.data.ExampleData(
                text="[セクション 1]\nStreamlineFrameworkは、エンタープライズ向けのJavaフレームワークです。依存性注入（DI）、アスペクト指向プログラミング（AOP）、Model-View-Controller（MVC）アーキテクチャを統合的にサポートしています。\n\n[セクション 2]\n@Controller、@Service、@Repository、@Componentアノテーションによる自動コンポーネント検出機能を提供し、大規模なWebアプリケーション開発を効率化します。",
                extractions=self._create_example_extractions_1(),
            ),
            lx.data.ExampleData(
                text="[セクション 3]\nデータアクセス層では、StreamlineORMが独自のO/Rマッピング機能を提供します。@Entity、@Table、@Column、@Id、@GeneratedValueアノテーションを使用してエンティティクラスを定義できます。\n\n[セクション 4]\nStreamlineORMは従来のORMツールと比較して30%のパフォーマンス向上を実現しており、大規模なデータベース操作においても高い処理速度を維持します。",
                extractions=self._create_example_extractions_2(),
            ),
        ]

    def _create_example_extractions_1(self) -> List[lx.data.Extraction]:
        """セクション1の例抽出を作成（RAG参考資料からの重要フレーズ抽出）"""
        return [
            lx.data.Extraction(
                extraction_class="feature_name",
                extraction_text="エンタープライズ向けのJavaフレームワーク",
                char_interval=lx.data.CharInterval(start_pos=35, end_pos=58),
                attributes={
                    "category": "フレームワーク特徴",
                    "importance": "high",
                    "context": "StreamlineFrameworkの基本概要",
                    "section": "1",
                },
            ),
            lx.data.Extraction(
                extraction_class="technical_term",
                extraction_text="依存性注入（DI）",
                char_interval=lx.data.CharInterval(start_pos=62, end_pos=73),
                attributes={
                    "category": "技術用語",
                    "importance": "high",
                    "context": "アーキテクチャの特徴",
                    "section": "1",
                },
            ),
            lx.data.Extraction(
                extraction_class="technical_term",
                extraction_text="アスペクト指向プログラミング（AOP）",
                char_interval=lx.data.CharInterval(start_pos=75, end_pos=97),
                attributes={
                    "category": "技術用語",
                    "importance": "high",
                    "context": "プログラミングパラダイム",
                    "section": "1",
                },
            ),
            lx.data.Extraction(
                extraction_class="architecture_pattern",
                extraction_text="Model-View-Controller（MVC）",
                char_interval=lx.data.CharInterval(start_pos=99, end_pos=122),
                attributes={
                    "category": "アーキテクチャパターン",
                    "importance": "high",
                    "context": "アーキテクチャ設計",
                    "section": "1",
                },
            ),
            lx.data.Extraction(
                extraction_class="annotation_name",
                extraction_text="@Controller",
                char_interval=lx.data.CharInterval(start_pos=154, end_pos=165),
                attributes={
                    "category": "アノテーション",
                    "purpose": "Webコントローラ定義",
                    "importance": "high",
                    "section": "2",
                },
            ),
            lx.data.Extraction(
                extraction_class="annotation_name",
                extraction_text="@Service",
                char_interval=lx.data.CharInterval(start_pos=167, end_pos=175),
                attributes={
                    "category": "アノテーション",
                    "purpose": "サービス層定義",
                    "importance": "high",
                    "section": "2",
                },
            ),
            lx.data.Extraction(
                extraction_class="annotation_name",
                extraction_text="@Repository",
                char_interval=lx.data.CharInterval(start_pos=177, end_pos=188),
                attributes={
                    "category": "アノテーション",
                    "purpose": "データアクセス層定義",
                    "importance": "high",
                    "section": "2",
                },
            ),
            lx.data.Extraction(
                extraction_class="annotation_name",
                extraction_text="@Component",
                char_interval=lx.data.CharInterval(start_pos=190, end_pos=200),
                attributes={
                    "category": "アノテーション",
                    "purpose": "汎用コンポーネント定義",
                    "importance": "high",
                    "section": "2",
                },
            ),
            lx.data.Extraction(
                extraction_class="feature_name",
                extraction_text="自動コンポーネント検出機能",
                char_interval=lx.data.CharInterval(start_pos=211, end_pos=224),
                attributes={
                    "category": "機能名",
                    "importance": "medium",
                    "context": "アノテーション機能",
                    "section": "2",
                },
            ),
        ]

    def _create_example_extractions_2(self) -> List[lx.data.Extraction]:
        """セクション2の例抽出を作成（RAG参考資料からの重要フレーズ抽出）"""
        return [
            lx.data.Extraction(
                extraction_class="feature_name",
                extraction_text="StreamlineORM",
                char_interval=lx.data.CharInterval(start_pos=23, end_pos=36),
                attributes={
                    "category": "フレームワークコンポーネント",
                    "component_type": "O/Rマッピング",
                    "importance": "high",
                    "section": "3",
                },
            ),
            lx.data.Extraction(
                extraction_class="feature_name",
                extraction_text="O/Rマッピング機能",
                char_interval=lx.data.CharInterval(start_pos=39, end_pos=50),
                attributes={
                    "category": "機能名",
                    "importance": "medium",
                    "context": "データアクセス層",
                    "section": "3",
                },
            ),
            lx.data.Extraction(
                extraction_class="annotation_name",
                extraction_text="@Entity",
                char_interval=lx.data.CharInterval(start_pos=56, end_pos=63),
                attributes={
                    "category": "ORMアノテーション",
                    "purpose": "エンティティクラス定義",
                    "importance": "high",
                    "section": "3",
                },
            ),
            lx.data.Extraction(
                extraction_class="annotation_name",
                extraction_text="@Table",
                char_interval=lx.data.CharInterval(start_pos=65, end_pos=71),
                attributes={
                    "category": "ORMアノテーション",
                    "purpose": "テーブルマッピング",
                    "importance": "medium",
                    "section": "3",
                },
            ),
            lx.data.Extraction(
                extraction_class="annotation_name",
                extraction_text="@Column",
                char_interval=lx.data.CharInterval(start_pos=73, end_pos=80),
                attributes={
                    "category": "ORMアノテーション",
                    "purpose": "カラムマッピング",
                    "importance": "medium",
                    "section": "3",
                },
            ),
            lx.data.Extraction(
                extraction_class="annotation_name",
                extraction_text="@Id",
                char_interval=lx.data.CharInterval(start_pos=82, end_pos=85),
                attributes={
                    "category": "ORMアノテーション",
                    "purpose": "主キー指定",
                    "importance": "high",
                    "section": "3",
                },
            ),
            lx.data.Extraction(
                extraction_class="annotation_name",
                extraction_text="@GeneratedValue",
                char_interval=lx.data.CharInterval(start_pos=87, end_pos=102),
                attributes={
                    "category": "ORMアノテーション",
                    "purpose": "主キー自動生成",
                    "importance": "medium",
                    "section": "3",
                },
            ),
            lx.data.Extraction(
                extraction_class="performance_metric",
                extraction_text="30%のパフォーマンス向上",
                char_interval=lx.data.CharInterval(start_pos=180, end_pos=194),
                attributes={
                    "category": "性能指標",
                    "metric_type": "パフォーマンス改善",
                    "importance": "high",
                    "section": "4",
                },
            ),
        ]

    def extract_sources(
        self, retrieved_context: str, prompt: str
    ) -> List[SourceLocation]:
        """RAG参考資料からソース位置を抽出"""
        try:
            print("LangExtractでソース抽出開始")
            print(
                f"処理対象RAG参考資料: {retrieved_context[:100]}..."
            )  # 最初の100文字を表示

            result = self._perform_extraction(retrieved_context, prompt)
            source_locations = self._process_extraction_results(
                result.extractions, retrieved_context
            )

            print(f"LangExtract抽出完了: {len(source_locations)}件のソース位置")
            return source_locations

        except Exception as e:
            raise Exception(f"LangExtract抽出エラー: {e}") from e

    def _perform_extraction(self, retrieved_context: str, prompt: str):
        """LangExtract抽出を実行"""
        # プロンプトのプレースホルダーを置換
        formatted_prompt = prompt.replace("{retrieved_context}", retrieved_context)

        return lx.extract(
            text_or_documents=retrieved_context,
            prompt_description=formatted_prompt,
            examples=self.examples,
            extraction_passes=self.config.langextract_extraction_passes,
            max_workers=self.config.langextract_max_workers,
            max_char_buffer=self.config.langextract_max_char_buffer,
            model_id=self.gemini_client.model_name,
            api_key=self.gemini_client.api_key,
            temperature=self.config.gemini_temperature,
        )

    def _process_extraction_results(
        self, extractions: List[lx.data.Extraction], context: str
    ) -> List[SourceLocation]:
        """抽出結果を処理してSourceLocationリストを作成"""
        source_locations = []

        for extraction in extractions:
            section_num = self._extract_section_number(
                context, extraction.extraction_text
            )
            if section_num is None:
                section_num = getattr(extraction, "attributes", {}).get(
                    "document_section", 0
                )

            source_location = SourceLocation(
                start_char=getattr(extraction.char_interval, "start_pos", 0),
                end_char=getattr(
                    extraction.char_interval,
                    "end_pos",
                    len(extraction.extraction_text),
                ),
                keyword=extraction.extraction_text,
                category=extraction.extraction_class,
                section=section_num,
            )
            source_locations.append(source_location)

        return source_locations

    def _extract_section_number(
        self, retrieved_context: str, extracted_text: str
    ) -> int:
        """RAG参考資料から該当するセクション番号を特定"""
        # セクション形式の参考資料からセクション番号を抽出
        lines = retrieved_context.split("\n")
        current_section = 0
        text_position = retrieved_context.find(extracted_text)

        if text_position == -1:
            return current_section

        # 抽出されたテキストの位置までの行を確認してセクション番号を特定
        char_count = 0
        for line in lines:
            if char_count <= text_position < char_count + len(line):
                break
            char_count += len(line) + 1  # +1 for newline
            if line.startswith("[セクション ") and line.endswith("]"):
                try:
                    current_section = int(line.split(" ")[1].rstrip("]"))
                except (IndexError, ValueError):
                    pass

        return current_section


class LangChainRAGEngine:
    """LangChainベースのRAGエンジン（Ollama使用）"""

    def __init__(self, config: SystemConfig, ollama_client: OllamaClient):
        self.config = config
        self.ollama_client = ollama_client
        self.vectorstore = None
        self.qa_chain = None

    def setup_vectorstore(self, documents: List[Document]) -> None:
        """ベクトルストアを設定"""
        try:
            # 既存のChromaDBディレクトリをクリア（デバッグ用）
            import shutil

            if os.path.exists(self.config.chroma_persist_directory):
                print(f"既存のChromaDBを削除中: {self.config.chroma_persist_directory}")
                shutil.rmtree(self.config.chroma_persist_directory)

            splits = self._split_documents(documents)
            embeddings = self._create_embeddings()
            self.vectorstore = self._create_vectorstore(splits, embeddings)

            print(f"✓ ベクトルストア作成完了: {len(splits)}個のチャンク")

        except Exception as e:
            raise Exception(f"ベクトルストア設定エラー: {e}") from e

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """ドキュメントを分割"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.text_splitter_chunk_size,
            chunk_overlap=self.config.text_splitter_chunk_overlap,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)

        print(f"\n=== 元文書数: {len(documents)} → 分割後チャンク数: {len(splits)} ===")
        print("\n=== 全チャンクの内容 ===")
        for i, split in enumerate(splits):
            print(f"\n--- チャンク {i} ---")
            print(f"元セクション: {split.metadata.get('section', 'Unknown')}")
            print(f"長さ: {len(split.page_content)} 文字")
            print(f"内容:\n{split.page_content}")
            print(f"完全なメタデータ: {split.metadata}")
            print("-" * 60)

        # ベクトルストア作成前にチャンクの検証
        print("\n=== チャンクの検証 ===")
        for i, split in enumerate(splits):
            if len(split.page_content.strip()) < 10:
                print(f"⚠️ チャンク {i} の内容が短すぎます: '{split.page_content}'")
            if "キャッシュ" in split.page_content:
                print(f"✓ チャンク {i} にキャッシュ関連の内容が含まれています")

        return splits

    def _create_embeddings(self) -> HuggingFaceEmbeddings:
        """埋め込みモデルを作成"""
        return HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={"device": self.config.embedding_device},
        )

    def _create_vectorstore(
        self, splits: List[Document], embeddings: HuggingFaceEmbeddings
    ) -> Chroma:
        """Chromaベクトルストアを作成"""
        print("\n=== ベクトルストア作成前の最終確認 ===")
        for i, split in enumerate(splits[:3]):  # 最初の3つをチェック
            print(f"チャンク {i}: {len(split.page_content)}文字")
            print(f"内容: {split.page_content[:100]}...")
            print(f"メタデータ: {split.metadata}")

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=self.config.chroma_persist_directory,
        )

        # ベクトルストア作成後の確認
        print("\n=== ベクトルストア作成後の確認 ===")
        print(f"保存された文書数: {vectorstore._collection.count()}")

        return vectorstore

    def setup_qa_chain(self, prompt_template: str) -> None:
        """QAチェーンを設定"""
        try:
            if not self.vectorstore:
                raise ValueError("ベクトルストアが設定されていません")

            prompt = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

            retriever = self._create_retriever()
            llm = self.ollama_client.get_langchain_llm()

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": prompt,
                    "document_variable_name": "context",
                },
            )

            print("✓ QAチェーン設定完了")

        except Exception as e:
            raise Exception(f"QAチェーン設定エラー: {e}") from e

    def _create_retriever(self):
        """リトリーバーを作成"""
        return self.vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": self.config.retriever_k,
                "fetch_k": self.config.retriever_fetch_k,
                "lambda_mult": self.config.retriever_lambda_mult,
            },
        )

    def query(self, question: str) -> Dict[str, Any]:
        """質問に対してRAG検索を実行"""
        try:
            if not self.qa_chain:
                raise ValueError("QAチェーンが設定されていません")

            result = self.qa_chain.invoke({"query": question})
            return self._process_query_result(result)

        except Exception as e:
            raise Exception(f"RAG検索エラー: {e}") from e

    def _process_query_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """クエリ結果を処理"""
        seen_sections = set()
        retrieved_context_with_sections = []
        unique_documents = []

        print("RAG検索結果:", result)
        print(f"取得した文書数: {len(result['source_documents'])}")

        print("\n=== RAG検索で取得された全文書 ===")
        for i, doc in enumerate(result["source_documents"]):
            section_num = doc.metadata.get("section", 0)
            print(f"\n--- 検索結果文書 {i} ---")
            print(f"セクション番号: {section_num}")
            print(f"内容長: {len(doc.page_content)} 文字")
            print(f"メタデータ: {doc.metadata}")
            print(f"内容:\n{doc.page_content}")
            print("-" * 50)

            # 同じセクション番号の文書は一度だけ追加
            if section_num not in seen_sections:
                seen_sections.add(section_num)

                # 内容が空でないことを確認
                if doc.page_content.strip():
                    content_with_section = (
                        f"[セクション {section_num}]\n{doc.page_content}"
                    )
                    retrieved_context_with_sections.append(content_with_section)
                    unique_documents.append(doc)
                    print(f"セクション {section_num} を追加しました")
                else:
                    print(f"⚠️ セクション {section_num} の内容が空です")

        final_context = "\n\n".join(retrieved_context_with_sections)
        print(f"最終的な retrieved_context 長: {len(final_context)}")
        print(f"最終的な retrieved_context の先頭200文字: {final_context[:200]}...")

        return {
            "answer": result["result"],
            "source_documents": unique_documents,
            "retrieved_context": final_context,
        }


class HybridRAGSystem:
    """ハイブリッドRAGシステムメインクラス（Ollama + Gemini）"""

    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig.from_env()

        # コンポーネントを初期化
        self.prompt_loader = PromptLoader(self.config)
        self.document_loader = DocumentLoader(self.config)
        self.ollama_client = OllamaClient(self.config)
        self.gemini_client = GeminiClient(self.config)
        self.langextract_client = LangExtractClient(self.config, self.gemini_client)
        self.rag_engine = LangChainRAGEngine(self.config, self.ollama_client)

        # データを読み込み
        self.context = self.document_loader.load_documents()
        self.documents = self.document_loader.load_documents_as_langchain()

        # RAGシステムを設定
        self._setup_rag_system()
        self._print_initialization_info()

    def _setup_rag_system(self) -> None:
        """RAGシステムを設定"""
        try:
            self.rag_engine.setup_vectorstore(self.documents)
            rag_prompt_template = self.prompt_loader.get_prompt("INITIAL_ANSWER_PROMPT")
            self.rag_engine.setup_qa_chain(rag_prompt_template)
        except Exception as e:
            print(f"⚠️ LangChain RAG設定でエラー: {e}")
            print("フォールバック: 従来の方式を使用します")
            self.rag_engine = None

    def _print_initialization_info(self) -> None:
        """初期化情報を表示"""
        print("✓ ハイブリッドRAGシステム初期化完了")
        print(f"✓ Ollamaモデル: {self.ollama_client.model_name}")
        print(f"✓ Geminiモデル: {self.gemini_client.model_name}")
        print(f"✓ 参考資料読み込み: {len(self.context)}文字")
        print(f"✓ LangChain文書数: {len(self.documents)}個")

    def process_question(self, question: str) -> HybridRAGResult:
        """質問を処理してハイブリッドRAG結果を生成"""
        print("\n=== 質問処理開始 ===")
        print(f"質問: {question}")

        # RAG回答生成
        print("\n1. RAG回答生成中...")
        rag_result = self._generate_rag_answer(question)
        answer, retrieved_context = self._extract_answer_and_context(rag_result)
        print(f"✓ 回答生成完了: {len(answer)}文字")

        # ソース位置特定
        print("\n2. ソース位置特定中...")
        source_locations = self._extract_source_locations(retrieved_context)
        print(f"✓ ソース位置特定完了: {len(source_locations)}件")

        # 結果をまとめる
        result = self._create_hybrid_rag_result(
            question, answer, retrieved_context, source_locations
        )

        print("\n=== 処理完了 ===")
        return result

    def _extract_answer_and_context(self, rag_result) -> tuple[str, str]:
        """RAG結果から回答とコンテキストを抽出"""
        if isinstance(rag_result, dict):
            answer = rag_result["answer"]
            retrieved_context = rag_result.get("retrieved_context", "")
            if self.config.debug:
                print(f"✓ 回答生成完了: {answer}")
                print(f"✓ 検索された参考資料: {retrieved_context}")
        else:
            answer = rag_result
            retrieved_context = ""
        return answer, retrieved_context

    def _create_hybrid_rag_result(
        self,
        question: str,
        answer: str,
        retrieved_context: str,
        source_locations: List[SourceLocation],
    ) -> HybridRAGResult:
        """HybridRAGResultを作成"""
        return HybridRAGResult(
            question=question,
            answer=answer,
            extracted_keywords=[],  # 後方互換性のため保持
            source_locations=source_locations,
            metadata={
                "rag_model": self.ollama_client.model_name,
                "langextract_model": self.gemini_client.model_name,
                "context_length": len(self.context),
                "retrieved_context_length": len(retrieved_context),
                "answer_length": len(answer),
                "keyword_count": 0,  # 後方互換性のため保持
                "source_count": len(source_locations),
            },
        )

    def _generate_rag_answer(self, question: str) -> Dict[str, str]:
        """RAG回答を生成（LangChain使用）"""
        if self.rag_engine:
            try:
                rag_result = self.rag_engine.query(question)
                print("answer:", rag_result["answer"])
                print("retrieved_context:", rag_result["retrieved_context"])
                return {
                    "answer": rag_result["answer"],
                    "retrieved_context": rag_result["retrieved_context"],
                }
            except Exception as e:
                print(f"⚠️ LangChain RAGでエラー: {e}")
                print("フォールバック: 従来方式を使用")

        # フォールバック処理
        return self._fallback_rag_answer(question)

    def _fallback_rag_answer(self, question: str) -> Dict[str, str]:
        """フォールバック用のRAG回答生成"""
        prompt_template = self.prompt_loader.get_prompt("INITIAL_ANSWER_PROMPT")
        prompt = prompt_template.format(context=self.context, question=question)
        answer = self.ollama_client.generate_response(prompt)

        # セクション情報を含める
        fallback_context_with_sections = []
        sections = self.context.split("\n\n")
        for i, section in enumerate(sections):
            if section.strip():
                fallback_context_with_sections.append(
                    f"[セクション {i}]\n{section.strip()}"
                )

        return {
            "answer": answer,
            "retrieved_context": "\n\n".join(fallback_context_with_sections),
        }

    def _extract_source_locations(self, retrieved_context: str) -> List[SourceLocation]:
        """RAG参考資料からソース位置を特定"""
        prompt = self.prompt_loader.get_prompt("LANGEXTRACT_SOURCE_PROMPT")
        return self.langextract_client.extract_sources(retrieved_context, prompt)


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

    print(f"\n【ソース位置情報】({len(result.source_locations)}件)")
    for i, source in enumerate(result.source_locations, 1):
        print(f"  {i}. 抽出内容: {source.keyword}")
        print(f"     カテゴリ: {source.category}")
        print(f"     セクション: {source.section}")
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
