import os
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import langextract as lx
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langextract import inference

# 警告を無視
warnings.filterwarnings("ignore")


@dataclass
class ExtractedData:
    """抽出されたデータの詳細情報"""

    value: Any
    exact_source: str
    page_number: Optional[int] = None
    line_number: Optional[int] = None
    confidence_score: float = 1.0
    cross_references: List[str] = None
    calculation_source: Optional[str] = None

    def __post_init__(self):
        if self.cross_references is None:
            self.cross_references = []


@dataclass
class VerificationData:
    """検証データ"""

    source_components: Dict[str, str]
    calculated_value: Any
    verification_notes: str


class LangExtractRAG:
    """LangExtractを統合した拡張RAGシステム"""

    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None
        self.documents = []
        self.langextract_examples = self._create_langextract_examples()

    def _create_langextract_examples(self):
        """LangExtract用の抽出例を作成"""
        examples = [
            lx.data.ExampleData(
                text="Pythonの機械学習ライブラリであるscikit-learnは、プロジェクトの85%で使用されており、2007年に最初にリリースされました。",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="library_name",
                        extraction_text="scikit-learn",
                        attributes={"category": "機械学習", "language": "Python"},
                    ),
                    lx.data.Extraction(
                        extraction_class="usage_statistics",
                        extraction_text="85%",
                        attributes={
                            "metric": "プロジェクト利用率",
                            "context": "scikit-learn使用率",
                        },
                    ),
                    lx.data.Extraction(
                        extraction_class="release_year",
                        extraction_text="2007年",
                        attributes={
                            "event": "最初のリリース",
                            "software": "scikit-learn",
                        },
                    ),
                ],
            ),
            lx.data.ExampleData(
                text="TensorFlowはGoogleが開発したディープラーニングフレームワークで、産業界で広く採用されています。",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="framework_name",
                        extraction_text="TensorFlow",
                        attributes={
                            "category": "ディープラーニング",
                            "developer": "Google",
                        },
                    ),
                    lx.data.Extraction(
                        extraction_class="adoption_info",
                        extraction_text="産業界で広く採用",
                        attributes={"scope": "産業界", "level": "広範囲"},
                    ),
                ],
            ),
        ]
        return examples

    def load_documents(self):
        """サンプルドキュメントを読み込む"""
        with open("sample_documents.txt", "r", encoding="utf-8") as f:
            content = f.read()

        # ドキュメントをセクションに分割し、行番号情報を保持
        sections = content.split("\n\n")
        documents = []
        line_offset = 0

        for i, section in enumerate(sections):
            if section.strip():
                # 各行に行番号を付与
                lines = section.strip().split("\n")
                line_numbers = list(
                    range(line_offset + 1, line_offset + len(lines) + 1)
                )

                doc = Document(
                    page_content=section.strip(),
                    metadata={
                        "source": "sample_documents.txt",
                        "section": i,
                        "start_line": line_offset + 1,
                        "end_line": line_offset + len(lines),
                        "line_numbers": ",".join(
                            map(str, line_numbers)
                        ),  # リストを文字列に変換
                    },
                )
                documents.append(doc)
                line_offset += len(lines) + 2  # セクション間の空行を考慮

        self.documents = documents
        return documents

    def create_vector_store(self, documents):
        """ベクトルストアを作成"""
        # テキストスプリッターを初期化（より細かく分割して詳細な位置情報を保持）
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # より小さなチャンクで詳細な位置情報を保持
            chunk_overlap=50,
            length_function=len,
        )

        # ドキュメントを分割し、各チャンクに詳細な位置情報を付与
        splits = []
        for doc in documents:
            doc_splits = text_splitter.split_documents([doc])
            for i, split in enumerate(doc_splits):
                # チャンク内の行番号を計算
                split.metadata.update(
                    {
                        "chunk_id": f"{doc.metadata['section']}_{i}",
                        "chunk_start_char": split.page_content.find(
                            split.page_content.split("\n")[0]
                        )
                        if "\n" in split.page_content
                        else 0,
                    }
                )
                splits.append(split)

        # 複雑なメタデータをフィルタリング（ChromaDB対応）
        filtered_splits = filter_complex_metadata(splits)

        # 埋め込みモデルを初期化
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

        # Chromaベクトルストアを作成
        persist_directory = os.getenv(
            "CHROMA_PERSIST_DIRECTORY", "./chroma_db_enhanced"
        )
        vectorstore = Chroma.from_documents(
            documents=filtered_splits,
            embedding=embeddings,
            persist_directory=persist_directory,
        )

        self.vectorstore = vectorstore
        return vectorstore

    def load_enhanced_prompt_template(self):
        """詳細な抽出用のプロンプトテンプレートを作成"""
        template = """
あなたは詳細なデータ抽出の専門家です。以下のルールに従って、正確で検証可能な回答を提供してください：

1. 提供された文書から具体的な数値、事実、データを抽出する
2. 各データポイントについて、正確なソース位置（行番号、文章）を特定する
3. 可能な場合、他の文書部分での言及や関連情報も特定する
4. 計算が必要な場合、その計算過程と根拠を明示する
5. 信頼度や確実性についても言及する

参考文書:
{context}

質問: {question}

以下の形式で回答してください：

【抽出データ】
- データ項目1: 値
  - 正確なソース: "文書内の正確な引用"
  - 位置情報: セクション番号、推定行位置
  - 信頼度: 高/中/低
  - 関連参照: 他の関連箇所

【検証情報】
- 計算過程や検証方法
- 矛盾や不確実性がある場合の指摘

【総合回答】
質問に対する包括的な回答
"""

        return PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

    def extract_with_langextract(self, query: str, context_text: str) -> Dict[str, Any]:
        """LangExtractを使用した詳細な情報抽出（必須）"""
        # LangExtractを直接使用（エラー時はフォールバックしない）
        return self._extract_with_langextract_ollama(query, context_text)

    def _extract_with_langextract_ollama(
        self, query: str, context_text: str
    ) -> Dict[str, Any]:
        """LangExtractのOllama統合版（必須使用）"""
        # 抽出タスクの定義
        prompt_description = f"""
        以下のテキストから、質問「{query}」に関連する情報を構造化して抽出してください：
        
        1. 技術名・ライブラリ名・フレームワーク名
        2. 数値データ（パーセンテージ、年、バージョン、統計など）
        3. 特徴・性能・利用状況の記述
        4. 開発者・組織情報
        5. リリース日・バージョン情報
        
        正確なテキストを抽出し、推測や意訳は行わないでください。
        各抽出項目に対して、意味のある属性を追加してコンテキストを提供してください。
        """

        # Ollama設定でLangExtractを実行
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "mistral")

        # LangExtractを必須として実行
        result = lx.extract(
            text_or_documents=context_text,
            prompt_description=prompt_description,
            examples=self.langextract_examples,
            extraction_passes=int(os.getenv("LANGEXTRACT_EXTRACTION_PASSES", "2")),
            max_workers=int(os.getenv("LANGEXTRACT_MAX_WORKERS", "3")),
            max_char_buffer=int(os.getenv("LANGEXTRACT_MAX_CHAR_BUFFER", "600")),
            language_model_type=inference.OllamaLanguageModel,
            language_model_params={
                "model": "tinyllama",
                "model_url": "http://localhost:11434",
            },
        )

        # result = lx.extract(
        #     text_or_documents=input_text,
        #     prompt_description=prompt,
        #     examples=examples,
        #     language_model_type=inference.OllamaLanguageModel,
        #     language_model_params={
        #         "model": "tinyllama",
        #         "model_url": "http://localhost:11434",
        #     },
        # )

        # 結果を構造化
        structured_result = {
            "extractions": [],
            "extraction_metadata": {
                "model_used": f"ollama/{ollama_model}",
                "api_base": ollama_base_url,
                "extraction_passes": int(
                    os.getenv("LANGEXTRACT_EXTRACTION_PASSES", "2")
                ),
                "source_text_length": len(context_text),
                "total_extractions": len(result.extractions),
                "method": "langextract_ollama",
            },
        }

        # 抽出結果を処理
        for extraction in result.extractions:
            extraction_data = {
                "class": extraction.extraction_class,
                "text": extraction.extraction_text,
                "attributes": extraction.attributes
                if hasattr(extraction, "attributes")
                else {},
                "source_position": {
                    "start": getattr(extraction, "start_char", None),
                    "end": getattr(extraction, "end_char", None),
                },
                "confidence": getattr(extraction, "confidence", 1.0),
            }
            structured_result["extractions"].append(extraction_data)

        return structured_result

    def create_qa_chain(self):
        """拡張QAチェーンを作成"""
        if not self.vectorstore:
            raise ValueError("ベクトルストアが初期化されていません")

        # 拡張プロンプトテンプレートを読み込み
        prompt_template = self.load_enhanced_prompt_template()

        # Ollamaローカルモデルを初期化
        llm = OllamaLLM(
            model="mistral",
            base_url="http://localhost:11434",
            temperature=0.1,  # より確定的な回答のため低く設定
            num_predict=1024,  # より詳細な回答のため長く設定
        )

        print("拡張RAG用Ollamaモデルを初期化しました")

        # RetrievalQAチェーンを作成
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 5}  # より多くの文書を検索
            ),
            return_source_documents=True,
            verbose=True,
            chain_type_kwargs={
                "prompt": prompt_template,
                "document_variable_name": "context",
            },
        )

        self.qa_chain = qa_chain
        return qa_chain

    def extract_detailed_data(self, query: str) -> Dict[str, Any]:
        """LangExtractを統合した詳細なデータ抽出を実行"""
        if not self.qa_chain:
            raise ValueError("QAチェーンが初期化されていません")

        # 基本的なQA実行
        result = self.qa_chain.invoke({"query": query})

        # 全ソース文書を結合してLangExtractで抽出
        combined_context = "\n\n".join(
            [doc.page_content for doc in result["source_documents"]]
        )

        # LangExtractで詳細抽出（必須）
        langextract_results = self.extract_with_langextract(query, combined_context)

        # ソース文書から詳細な位置情報を抽出（従来の方法も併用）
        detailed_sources = self._analyze_source_documents(
            result["source_documents"], query
        )

        # LangExtractの結果を統合
        for i, source in enumerate(detailed_sources):
            # 各ソース文書に対してもLangExtract抽出を実行（必須）
            source_langextract = self.extract_with_langextract(
                query, source["content_preview"]
            )
            source["langextract_data"] = source_langextract

        # 構造化された結果を作成
        enhanced_result = {
            "query": query,
            "answer": result["result"],
            "langextract_analysis": langextract_results,
            "detailed_sources": detailed_sources,
            "verification_data": self._create_verification_data(
                result["source_documents"], result["result"]
            ),
            "confidence_analysis": self._analyze_confidence(
                result["source_documents"], result["result"]
            ),
            "extraction_summary": self._create_extraction_summary(langextract_results),
        }

        return enhanced_result

    def _create_extraction_summary(self, langextract_results: Dict) -> Dict:
        """LangExtract結果のサマリーを作成"""
        extractions = langextract_results.get("extractions", [])

        summary = {
            "total_extractions": len(extractions),
            "extraction_types": {},
            "key_findings": [],
            "numerical_data": [],
            "technical_terms": [],
        }

        for extraction in extractions:
            extraction_class = extraction.get("class", "unknown")

            # 抽出タイプの統計
            if extraction_class not in summary["extraction_types"]:
                summary["extraction_types"][extraction_class] = 0
            summary["extraction_types"][extraction_class] += 1

            # 重要な発見を分類
            text = extraction.get("text", "")
            attributes = extraction.get("attributes", {})

            if any(
                keyword in extraction_class.lower()
                for keyword in ["statistics", "usage", "percent"]
            ):
                summary["numerical_data"].append({"value": text, "context": attributes})
            elif any(
                keyword in extraction_class.lower()
                for keyword in ["library", "framework", "technology"]
            ):
                summary["technical_terms"].append(
                    {"term": text, "attributes": attributes}
                )
            else:
                summary["key_findings"].append(
                    {
                        "type": extraction_class,
                        "content": text,
                        "attributes": attributes,
                    }
                )

        return summary

    def _analyze_source_documents(
        self, source_docs: List[Document], query: str
    ) -> List[Dict]:
        """ソース文書の詳細分析"""
        detailed_sources = []

        for i, doc in enumerate(source_docs):
            source_info = {
                "document_id": i + 1,
                "source_metadata": doc.metadata,
                "content_preview": doc.page_content[:200] + "..."
                if len(doc.page_content) > 200
                else doc.page_content,
                "relevance_score": self._calculate_relevance_score(
                    doc.page_content, query
                ),
                "exact_quotes": self._find_exact_quotes(doc.page_content, query),
            }

            detailed_sources.append(source_info)

        return detailed_sources

    def _calculate_relevance_score(self, text: str, query: str) -> float:
        """関連性スコアを計算"""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())

        # 共通単語の割合を計算
        if not query_words:
            return 0.0

        common_words = query_words.intersection(text_words)
        return len(common_words) / len(query_words)

    def _find_exact_quotes(self, text: str, query: str) -> List[str]:
        """質問に関連する正確な引用を抽出"""
        sentences = re.split(r"[.。!！?？]", text)
        query_words = set(query.lower().split())

        relevant_quotes = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            # 質問の単語が含まれている文を抽出
            if query_words.intersection(sentence_words):
                relevant_quotes.append(sentence.strip())

        return relevant_quotes[:3]  # 上位3つの引用

    def _create_verification_data(
        self, source_docs: List[Document], answer: str
    ) -> Dict:
        """検証データを作成"""
        return {
            "source_count": len(source_docs),
            "answer_length": len(answer),
            "cross_references": self._find_cross_references(source_docs),
            "consistency_check": self._check_consistency(source_docs, answer),
        }

    def _find_cross_references(self, source_docs: List[Document]) -> List[str]:
        """クロスリファレンスを見つける"""
        cross_refs = []
        for doc in source_docs:
            # セクション間の参照を検索
            if "参照" in doc.page_content or "関連" in doc.page_content:
                cross_refs.append(f"Section {doc.metadata.get('section', 'unknown')}")
        return cross_refs

    def _check_consistency(self, source_docs: List[Document], answer: str) -> str:
        """一貫性をチェック"""
        # 簡単な一貫性チェック
        if len(source_docs) > 1:
            return "複数のソースから情報を統合"
        else:
            return "単一ソースからの情報"

    def _analyze_confidence(self, source_docs: List[Document], answer: str) -> Dict:
        """信頼度を分析"""
        return {
            "confidence_level": "高"
            if len(source_docs) >= 3
            else "中"
            if len(source_docs) >= 2
            else "低",
            "reasoning": f"{len(source_docs)}個のソース文書から生成",
            "verification_needed": len(source_docs) < 2,
        }

    def demo_enhanced_rag(self):
        """拡張RAGのデモ"""
        print("\n=== LangExtract統合拡張RAGデモ ===")

        # ドキュメントを読み込み
        print("1. ドキュメントを読み込み中...")
        documents = self.load_documents()
        print(f"読み込んだドキュメント数: {len(documents)}")

        # ベクトルストアを作成
        print("2. 拡張ベクトルストアを作成中...")
        self.create_vector_store(documents)

        # QAチェーンを作成
        print("3. 拡張QAチェーンを作成中...")
        self.create_qa_chain()

        # サンプル質問
        sample_queries = [
            "Pythonの機械学習ライブラリについて具体的な情報を教えて",
        ]

        print("\nサンプル質問での拡張RAG実行:")
        for i, query in enumerate(sample_queries, 1):
            print(f"\n--- サンプル質問 {i} ---")
            print(f"質問: {query}")

            try:
                result = self.extract_detailed_data(query)
                self._display_enhanced_result(result)
            except Exception as e:
                print(f"エラー: {e}")

        # インタラクティブモード
        print("\n\n=== インタラクティブモード ===")
        print("質問を入力してください（終了するには 'quit' と入力）:")

        while True:
            question = input("\n質問: ")
            if question.lower() == "quit":
                break

            try:
                result = self.extract_detailed_data(question)
                self._display_enhanced_result(result)
            except Exception as e:
                print(f"エラーが発生しました: {e}")

    def _display_enhanced_result(self, result: Dict):
        """LangExtract統合の拡張結果を表示"""
        print("\n" + "=" * 80)
        print("【LangExtract統合拡張RAG結果】")
        print("=" * 80)

        print("\n【回答】")
        print(result["answer"])

        # LangExtract分析結果
        if "langextract_analysis" in result:
            langextract = result["langextract_analysis"]
            print("\n【LangExtract構造化抽出結果】")
            print(
                f"抽出モデル: {langextract['extraction_metadata'].get('model_used', 'N/A')}"
            )
            print(
                f"抽出パス数: {langextract['extraction_metadata'].get('extraction_passes', 'N/A')}"
            )
            print(
                f"総抽出数: {langextract['extraction_metadata'].get('total_extractions', 0)}件"
            )

            for i, extraction in enumerate(langextract.get("extractions", []), 1):
                print(f"\n  📊 抽出項目 {i}:")
                print(f"     分類: {extraction.get('class', 'N/A')}")
                print(f'     抽出テキスト: "{extraction.get("text", "N/A")}"')

                if extraction.get("attributes"):
                    print("     属性:")
                    for key, value in extraction["attributes"].items():
                        print(f"       • {key}: {value}")

                if extraction.get("source_position", {}).get("start") is not None:
                    pos = extraction["source_position"]
                    print(
                        f"     位置: 文字{pos.get('start', 'N/A')}-{pos.get('end', 'N/A')}"
                    )

                if extraction.get("confidence"):
                    print(f"     信頼度: {extraction['confidence']:.2f}")

        print("\n【詳細ソース情報】")
        for source in result["detailed_sources"]:
            print(f"\n  📄 文書 {source['document_id']}:")
            print(f"     セクション: {source['source_metadata'].get('section', 'N/A')}")
            print(f"     関連性スコア: {source['relevance_score']:.2f}")
            print(f"     内容プレビュー: {source['content_preview']}")

            # 各ソースのLangExtract結果
            if (
                "langextract_data" in source
                and source["langextract_data"]["extractions"]
            ):
                print(
                    f"     LangExtract抽出数: {len(source['langextract_data']['extractions'])}件"
                )
                for ext in source["langextract_data"]["extractions"][:2]:  # 上位2つ
                    print(
                        f'       • {ext.get("class", "N/A")}: "{ext.get("text", "N/A")}"'
                    )

            if source["exact_quotes"]:
                print("     正確な引用:")
                for quote in source["exact_quotes"][:1]:  # 最初の1つ
                    print(f'       • "{quote[:80]}..."')

        print("\n【検証情報】")
        verification = result["verification_data"]
        print(f"  • ソース文書数: {verification['source_count']}")
        print(f"  • 一貫性チェック: {verification['consistency_check']}")
        if verification["cross_references"]:
            print(
                f"  • クロスリファレンス: {', '.join(verification['cross_references'])}"
            )

        print("\n【信頼度分析】")
        confidence = result["confidence_analysis"]
        print(f"  • 信頼度レベル: {confidence['confidence_level']}")
        print(f"  • 根拠: {confidence['reasoning']}")
        if confidence["verification_needed"]:
            print("  • ⚠️  追加検証が推奨されます")

        print("=" * 80)


def main():
    """メイン関数"""
    load_dotenv()

    print("LangExtract統合拡張RAGシステム")
    print("=" * 50)
    print("通常のRAGを超えた詳細なソース追跡と検証機能を提供します")

    try:
        enhanced_rag = LangExtractRAG()
        enhanced_rag.demo_enhanced_rag()

        print("\n=== 拡張RAGデモが完了しました ===")
        print("このシステムは以下の追加機能を提供します:")
        print("• 詳細なソース位置情報（行番号、文字位置）")
        print("• 数値・データポイントの正確な抽出")
        print("• クロスリファレンスと検証情報")
        print("• 信頼度分析と一貫性チェック")
        print("• 計算過程の透明性")

    except KeyboardInterrupt:
        print("\n処理が中断されました。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
