import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import langextract as lx
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()


@dataclass
class SourceLocation:
    """ソース位置情報"""

    text: str  # 引用されたテキスト
    start_char: int  # 開始文字位置
    end_char: int  # 終了文字位置
    document_id: Optional[str] = None  # 文書ID
    section: Optional[str] = None  # セクション情報


@dataclass
class StructuredAnswer:
    """構造化された回答フォーマット"""

    query: str  # 元の質問
    main_answer: str  # メイン回答
    key_facts: List[Dict[str, Any]]  # 重要な事実のリスト
    sources: List[SourceLocation]  # ソース位置情報
    metadata: Dict[str, Any]  # 追加メタデータ

    def to_json(self) -> str:
        """JSON文字列に変換"""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_langextract_result(
        cls, query: str, result, context: str
    ) -> "StructuredAnswer":
        """LangExtractの結果から構造化回答を作成"""
        key_facts = []
        sources = []

        for extraction in result.extractions:
            # 重要な事実を抽出
            fact_data = {
                "type": extraction.extraction_class,
                "value": extraction.extraction_text,
                "attributes": getattr(extraction, "attributes", {}),
            }
            key_facts.append(fact_data)

            # ソース位置情報を作成
            source_loc = SourceLocation(
                text=extraction.extraction_text,
                start_char=getattr(extraction, "start_char", 0),
                end_char=getattr(
                    extraction, "end_char", len(extraction.extraction_text)
                ),
            )
            sources.append(source_loc)

        # メイン回答を生成（最初の抽出結果または全体サマリー）
        main_answer = f"質問「{query}」に対する回答: "
        if key_facts:
            main_answer += "、".join([f"{fact['value']}" for fact in key_facts[:3]])

        return cls(
            query=query,
            main_answer=main_answer,
            key_facts=key_facts,
            sources=sources,
            metadata={
                "total_extractions": len(key_facts),
                "source_text_length": len(context),
                "extraction_method": "langextract_gemini",
            },
        )


class StructuredRAGExtractor:
    """構造化RAG抽出器（Gemini版）"""

    def __init__(
        self,
        gemini_model: str = None,
        api_key: str = None,
    ):
        self.gemini_model = gemini_model or os.getenv("GEMINI_MODEL")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY in .env file or pass as parameter."
            )

        self.examples = self._create_examples()

    def _create_examples(self):
        """LangExtract用の例を作成（最小限に制限）"""
        return [
            lx.data.ExampleData(
                text="Vue.js 3のComposition APIは開発者の75%に支持されており、2020年9月にリリースされました。",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="framework_name",
                        extraction_text="Vue.js",
                        attributes={
                            "category": "フロントエンドフレームワーク",
                            "version": "3",
                        },
                    ),
                    lx.data.Extraction(
                        extraction_class="developer_support",
                        extraction_text="75%",
                        attributes={
                            "metric": "開発者支持率",
                            "data_type": "統計",
                        },
                    ),
                    lx.data.Extraction(
                        extraction_class="release_date",
                        extraction_text="2020年9月",
                        attributes={"event": "リリース", "data_type": "日付"},
                    ),
                ],
            ),
        ]

    def extract_structured_answer(self, query: str, context: str) -> StructuredAnswer:
        """構造化された回答を抽出"""

        # より明確で制限的なプロンプト
        prompt_description = f"""
        以下の提供されたテキストから、質問「{query}」に関連する情報のみを抽出してください。
        
        重要な注意事項：
        - 提供されたテキスト以外の情報は一切抽出しないでください
        - 特に例や学習データからの抽出は禁止です
        - 推測や補完は行わないでください
        
        抽出対象（提供テキスト内のみ）:
        1. ライブラリ・フレームワーク名
        2. バージョン番号  
        3. 統計データ（パーセンテージ）
        4. パフォーマンス指標
        5. 技術的特徴・機能名
        6. 年・日付
        
        提供されたテキストにない情報は絶対に抽出してはいけません。
        """

        print(f"抽出開始: {query}")
        print(f"使用モデル: {self.gemini_model}")
        print(f"処理対象テキスト: {context[:100]}...")
        print(f"テキスト長: {len(context)}文字")

        try:
            # Gemini設定でLangExtractを実行
            result = lx.extract(
                text_or_documents=context,
                prompt_description=prompt_description,
                examples=self.examples,
                extraction_passes=2,  # パス数を減らして混乱を回避
                max_workers=3,  # ワーカー数を3に制限
                max_char_buffer=600,  # Geminiはより大きなバッファを処理可能
                model_id=self.gemini_model,
                api_key=self.api_key,
                temperature=0.1,  # より決定的な出力
            )

            print(f"抽出成功: {len(result.extractions)}件の抽出")

            # 構造化回答を作成
            structured_answer = StructuredAnswer.from_langextract_result(
                query, result, context
            )
            return structured_answer

        except Exception as e:
            # エラーを素直に再発生させる
            print(f"LangExtract抽出でエラーが発生: {e}")
            raise e


def demo_structured_rag():
    """構造化RAGのデモ（Gemini版）"""
    print("=== 構造化RAG with LangExtract (Gemini) デモ ===")

    try:
        # 抽出器の初期化
        extractor = StructuredRAGExtractor()
        print(f"✓ Gemini抽出器を初期化しました（モデル: {extractor.gemini_model}）")
    except ValueError as e:
        print(f"❌ 初期化エラー: {e}")
        print("Please set GEMINI_API_KEY in .env file")
        return

    # サンプルのRAGコンテキスト（技術系質問に対応）
    sample_contexts = [
        {
            "query": "PythonのライブラリscikitLearnの使用率は？",
            "context": """
            機械学習プロジェクトにおいて、scikit-learnは最も人気のあるライブラリの一つです。
            2023年の調査によると、Pythonの機械学習プロジェクトの85%でscikit-learnが使用されており、
            TensorFlowの78%、PyTorchの65%を上回っています。
            scikit-learnは2007年にDavid Cournapeau氏によって最初にリリースされました。
            """,
        },
        # {
        #     "query": "React 18の新機能と性能改善について教えて",
        #     "context": """
        #     React 18では、Concurrent Renderingが導入され、UIの応答性が大幅に改善されました。
        #     Suspenseの新しいAPIにより、非同期データの読み込みが30%高速化されています。
        #     また、Automatic Batching機能により、レンダリング回数が平均で40%削減され、
        #     開発者満足度調査では92%が肯定的な評価を示しています。
        #     """,
        # },
        # {
        #     "query": "Node.jsの最新バージョンの特徴は？",
        #     "context": """
        #     Node.js 18.0.0では、Fetch APIが標準で組み込まれ、外部ライブラリが不要になりました。
        #     V8エンジンのアップデートにより、JavaScript実行速度が前バージョンから15%向上しています。
        #     また、ESMサポートの改善により、モジュール読み込み時間が25%短縮されました。
        #     LTSサポートは2025年4月まで提供される予定です。
        #     """,
        # },
    ]

    # 各サンプルで構造化抽出を実行
    for i, sample in enumerate(sample_contexts, 1):
        print(f"\n--- サンプル {i} ---")
        print(f"質問: {sample['query']}")
        print(f"コンテキスト: {sample['context'][:100]}...")

        # 構造化回答を取得
        try:
            structured_answer = extractor.extract_structured_answer(
                sample["query"], sample["context"]
            )
        except Exception as e:
            print(f"致命的エラーが発生しました: {e}")
            print("処理を終了します。")
            return

        # JSON形式で出力
        print("\n【構造化回答（JSON）】")
        print(structured_answer.to_json())

        print("\n【バックエンド用データ】")
        print(f"メイン回答: {structured_answer.main_answer}")
        print(f"抽出された事実数: {len(structured_answer.key_facts)}")

        if structured_answer.sources:
            print("【ソース詳細】")
            for j, source in enumerate(structured_answer.sources, 1):
                print(f'  {j}. テキスト: "{source.text}"')
                print(f"     位置: {source.start_char}-{source.end_char}文字目")

        print("-" * 60)

    print("\n=== Gemini RAG デモが完了しました ===")


if __name__ == "__main__":
    demo_structured_rag()
