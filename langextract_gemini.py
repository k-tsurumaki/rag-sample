import os
import textwrap

import langextract as lx
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()

# 1. Define the prompt and extraction rules
prompt = textwrap.dedent("""\
    Extract characters, emotions, and relationships in order of appearance.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context.""")

# 2. Provide a high-quality example to guide the model
examples = [
    lx.data.ExampleData(
        text="ROMEO. But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.",
        extractions=[
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="ROMEO",
                attributes={"emotional_state": "wonder"},
            ),
            lx.data.Extraction(
                extraction_class="emotion",
                extraction_text="But soft!",
                attributes={"feeling": "gentle awe"},
            ),
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="Juliet is the sun",
                attributes={"type": "metaphor"},
            ),
        ],
    )
]

# The input text to be processed
input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"

print("=== LangExtract with Gemini Demo ===")
print(f"Input text: {input_text}")
print("Using Gemini model...")

try:
    # Run the extraction with Gemini
    result = lx.extract(
        text_or_documents=input_text,
        prompt_description=prompt,
        examples=examples,
        model_id="gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),  # Only use this for testing/development
    )

    print("\n✓ Extraction completed successfully!")
    print(f"Total extractions: {len(result.extractions)}")

    print("\nExtraction Result:")
    for i, extraction in enumerate(result.extractions, 1):
        print(f"\n{i}. Class: {extraction.extraction_class}")
        print(f'   Text: "{extraction.extraction_text}"')
        print(f"   Attributes: {extraction.attributes}")

        # ソース位置情報があれば表示
        if hasattr(extraction, "start_char") and hasattr(extraction, "end_char"):
            print(f"   Position: {extraction.start_char}-{extraction.end_char}")

except Exception as e:
    print(f"❌ Error occurred during extraction: {e}")
    print("\nPlease check:")
    print("1. GEMINI_API_KEY is set in .env file")
    print("2. Gemini API is accessible")
    print("3. Model name is correct")
    exit(1)

print("\n=== Demo completed ===")
