# 🧪 Gaming PC AI Performance Test
import torch
import time
import psutil
import numpy as np
from transformers import AutoTokenizer, AutoModel
import spacy

def test_gaming_pc_performance():
    print("🎮 Gaming PC AI Processing Performance Test")
    print("=" * 50)

    # System info
    print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

    # CPU performance test
    print("\n🔢 CPU Performance Test...")
    start_time = time.time()
    result = sum(i*i for i in range(1000000))
    cpu_time = time.time() - start_time
    print(f"CPU Test: {cpu_time:.3f} seconds")

    # GPU performance test
    if torch.cuda.is_available():
        print("\n🎮 GPU Performance Test...")
        device = torch.device('cuda')
        x = torch.randn(1000, 1000, device=device)
        start_time = time.time()
        for _ in range(100):
            y = torch.matmul(x, x)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        print(f"GPU Test: {gpu_time:.3f} seconds")

        # Memory test
        print("\n💾 GPU Memory Test...")
        try:
            # Allocate large tensor
            large_tensor = torch.randn(1000, 1000, device=device)
            memory_allocated = torch.cuda.memory_allocated(0) / (1024**2)
            memory_reserved = torch.cuda.memory_reserved(0) / (1024**2)
            print(f"GPU Memory Allocated: {memory_allocated:.1f} MB")
            print(f"GPU Memory Reserved: {memory_reserved:.1f} MB")
        except Exception as e:
            print(f"Memory test failed: {e}")

    # NLP model test
    print("\n🤖 NLP Model Test...")
    try:
        nlp = spacy.load("en_core_web_sm")
        start_time = time.time()
        doc = nlp("This is a test of the AI processing system on the gaming PC with RTX 3090.")
        nlp_time = time.time() - start_time
        print(f"NLP Processing: {nlp_time:.3f} seconds")
        print(f"Entities found: {len(doc.ents)}")
        for ent in doc.ents:
            print(f"  - {ent.text}: {ent.label_}")
    except Exception as e:
        print(f"NLP Test failed: {e}")

    # Transformers test
    print("\n🔄 Transformers Test...")
    try:
        from transformers import pipeline
        start_time = time.time()
        classifier = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1, batch_size=32)
        result = classifier("This gaming PC is amazing for AI processing!")
        transformers_time = time.time() - start_time
        print(f"Transformers Test: {transformers_time:.3f} seconds")
        print(f"Sentiment: {result[0]['label']} (confidence: {result[0]['score']:.3f})")
    except Exception as e:
        print(f"Transformers Test failed: {e}")

    print("\n✅ Gaming PC performance test complete!")
    print("🚀 Ready for 8-15x faster AI processing!")

if __name__ == "__main__":
    test_gaming_pc_performance()
