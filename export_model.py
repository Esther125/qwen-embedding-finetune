import argparse
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, models
import os


def main():
    parser = argparse.ArgumentParser(description="Export Swift-trained checkpoint to SentenceTransformer format")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to fine-tuned checkpoint folder (e.g. checkpoint-1125)")
    parser.add_argument("--output", type=str, required=True, help="Output directory for SentenceTransformer model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="Base model name")
    args = parser.parse_args()

    print(f"🔄 Exporting from checkpoint: {args.ckpt}")
    print(f"📚 Base model: {args.base_model}")
    print(f"💾 Exporting to SentenceTransformer format: {args.output}")

    # Step 1: Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.ckpt, trust_remote_code=True)

    # Step 2: Save HuggingFace model to temp dir (inside output/0_Transformer/)
    transformer_path = os.path.join(args.output, "0_Transformer")
    os.makedirs(transformer_path, exist_ok=True)
    tokenizer.save_pretrained(transformer_path)
    model.save_pretrained(transformer_path)

    # Step 3: Add pooling layer
    word_embedding_model = models.Transformer(transformer_path, max_seq_length=512)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode="mean"
    )

    sbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Step 4: Save full SentenceTransformer
    sbert_model.save(args.output)

    print("✅ Done! You can now load it using:")
    print(f"   SentenceTransformer('{args.output}')")


if __name__ == "__main__":
    main()
