import os
import shutil
from vector_store.setup import setup_vector_store
from config import CHROMA_PATH, DOCS_PATH


def main():
    print("=" * 50)
    print("Multi-Agent Research — Document Ingestion")
    print("=" * 50)

    if not os.path.exists(DOCS_PATH):
        print(f"ERROR: '{DOCS_PATH}' folder not found.")
        print("Create a 'docs/' folder and add PDF or TXT files.")
        return

    files = [f for f in os.listdir(DOCS_PATH) if f.endswith((".pdf", ".txt"))]
    if not files:
        print(f"ERROR: No PDF or TXT files found in '{DOCS_PATH}'.")
        print("Add your documents and run again.")
        return

    print(f"Found {len(files)} file(s) to ingest:")
    for f in files:
        print(f"  - {f}")

    if os.path.exists(CHROMA_PATH):
        print(f"\nExisting vector store found at '{CHROMA_PATH}'.")
        choice = input("Rebuild from scratch? (y/n): ").strip().lower()
        if choice == "y":
            shutil.rmtree(CHROMA_PATH)
            print("Old vector store deleted.")
        else:
            print("Keeping existing vector store. Exiting.")
            return

    print("\nStarting ingestion...\n")
    setup_vector_store()

    print("\n" + "=" * 50)
    print("Ingestion complete.")
    print(f"Vector store saved at: {CHROMA_PATH}")
    print("You can now run: python main.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
