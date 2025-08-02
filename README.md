# RAGサンプルプロジェクト

LangChainを使用したRAG（Retrieval-Augmented Generation）のサンプル実装です。

## 機能

1. **シンプルRAG**: 文書検索のみ（LLMなし）
2. **完全RAG**: Ollamaを使用した回答生成機能付き

## セットアップ

### 1. 依存関係のインストール
```bash
uv sync
```

### 2. Ollama設定（完全RAG機能を使用する場合）

1. Ollamaをインストール:
   ```bash
   # Linux/macOS
   curl -fsSL https://ollama.com/install.sh | sh
   
   # または公式サイトからダウンロード: https://ollama.com/
   ```

2. モデルをダウンロード:
   ```bash
   ollama pull llama3.2:3b
   # または軽量版: ollama pull llama3.2:1b
   ```

3. Ollamaサーバーを起動:
   ```bash
   ollama serve
   ```

## 実行方法

```bash
python main.py
```

実行時にモードを選択できます：
- `1`: シンプルデモ（検索のみ）
- `2`: 完全RAGデモ（Ollama + 回答生成）

## 使用技術

- **LangChain**: RAGフレームワーク
- **Chroma**: ベクトルデータベース
- **HuggingFace**: 埋め込みモデル
- **Ollama**: ローカルLLM実行環境

## ファイル構成

- `main.py`: メインプログラム
- `sample_documents.txt`: サンプル文書
- `.env`: 環境変数設定
- `chroma_db/`: ベクトルデータベース（自動生成）