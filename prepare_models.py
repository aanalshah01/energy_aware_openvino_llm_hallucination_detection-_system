import os
import subprocess

def convert_llm(model_id="Qwen/Qwen2.5-1.5B-Instruct"):
    print(f"--- Exporting {model_id} to OpenVINO FP16 ---")
    subprocess.run([
        "optimum-cli", "export", "openvino",
        "--model", model_id,
        "--weight-format", "fp16",
        "openvino_models/qwen_fp16"
    ], check=True)

    print(f"--- Exporting {model_id} to OpenVINO INT8 ---")
    subprocess.run([
        "optimum-cli", "export", "openvino",
        "--model", model_id,
        "--weight-format", "int8",
        "openvino_models/qwen_int8"
    ], check=True)

    print("Model conversions completed successfully!")

if __name__ == "__main__":
    convert_llm()