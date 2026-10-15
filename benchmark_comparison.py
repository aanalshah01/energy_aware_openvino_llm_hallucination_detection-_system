import pandas as pd
from pipeline import OpenVINOHallucinationPipeline

def run_comparative_benchmark():
    print("   LOADING OPENVINO FP16 & INT8 PIPELINES        ")

    
    # Initialize both model precision pipelines
    pipeline_fp16 = OpenVINOHallucinationPipeline(
        model_dir="./openvino_models/qwen_fp16", device="AUTO"
    )
    pipeline_int8 = OpenVINOHallucinationPipeline(
        model_dir="./openvino_models/qwen_int8", device="AUTO"
    )

    # Test questions from your academic course material / textbook database
    test_queries = [
        "What are the main contributions of Marie Curie to physics and chemistry?",
        "Explain Einstein's theory of relativity and his Nobel Prize win.",
        "What laws were formulated by Isaac Newton in Principia Mathematica?"
    ]

    results = []

    print("\nStarting evaluation across test queries...\n")
    for idx, query in enumerate(test_queries, 1):
        print(f"--- Running Query {idx}/{len(test_queries)}: '{query}' ---")
        
        # Process using FP16
        res_fp16 = pipeline_fp16.process_query(query)
        
        # Process using INT8
        res_int8 = pipeline_int8.process_query(query)

        # Store comparison data
        results.append({
            "Query": query[:35] + "...",
            "Precision": "FP16",
            "Latency (s)": res_fp16["metrics"]["latency_sec"],
            "Peak RAM (MB)": res_fp16["metrics"]["peak_memory_mb"],
            "Energy (J)": res_fp16["metrics"]["energy_consumed_joules"],
            "Hallucination Score (%)": res_fp16["hallucination_score_pct"],
            "Flagged": res_fp16["flagged"]
        })

        results.append({
            "Query": query[:35] + "...",
            "Precision": "INT8",
            "Latency (s)": res_int8["metrics"]["latency_sec"],
            "Peak RAM (MB)": res_int8["metrics"]["peak_memory_mb"],
            "Energy (J)": res_int8["metrics"]["energy_consumed_joules"],
            "Hallucination Score (%)": res_int8["hallucination_score_pct"],
            "Flagged": res_int8["flagged"]
        })

    # Print clean benchmark summary table
    df = pd.DataFrame(results)
    print("          BENCHMARK EXPERIMENTAL RESULTS          ")
    print(df.to_string(index=False))

    # Calculate average performance gains
    avg_fp16_lat = df[df["Precision"] == "FP16"]["Latency (s)"].mean()
    avg_int8_lat = df[df["Precision"] == "INT8"]["Latency (s)"].mean()
    speedup = ((avg_fp16_lat - avg_int8_lat) / avg_fp16_lat) * 100

    avg_fp16_h = df[df["Precision"] == "FP16"]["Hallucination Score (%)"].mean()
    avg_int8_h = df[df["Precision"] == "INT8"]["Hallucination Score (%)"].mean()

    print("\n--- Summary Highlights ---")
    print(f"• INT8 Speedup: {speedup:.2f}% faster query response time.")
    print(f"• Avg FP16 Hallucination Rate: {avg_fp16_h:.2f}%")
    print(f"• Avg INT8 Hallucination Rate: {avg_int8_h:.2f}%")

if __name__ == "__main__":
    run_comparative_benchmark()