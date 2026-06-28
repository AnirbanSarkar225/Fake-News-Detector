"""
ONNX Export & Quantization for TruthShield Transformer Models.

Exports trained DistilBERT / MuRIL models to ONNX format for
fast CPU inference with optional INT8 quantization.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --model distilbert --quantize
    python scripts/export_onnx.py --model all
"""

import os, sys, io, argparse, shutil

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def get_dir_size_mb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 * 1024)


def export_model(source_dir, output_dir, quantize=False):
    """Export a HuggingFace model to ONNX format."""
    if not os.path.isdir(source_dir) or not os.path.exists(os.path.join(source_dir, "config.json")):
        print(f"  ✗ Model not found at {source_dir}")
        return False

    print(f"  Source: {source_dir}")
    src_size = get_dir_size_mb(source_dir)
    print(f"  Source size: {src_size:.1f} MB")

    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
    except ImportError:
        print("  ✗ 'optimum' not installed. Run: pip install optimum[onnxruntime]")
        return False

    os.makedirs(output_dir, exist_ok=True)

    print("  Exporting to ONNX...")
    try:
        model = ORTModelForSequenceClassification.from_pretrained(source_dir, export=True)
        tokenizer = AutoTokenizer.from_pretrained(source_dir)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        return False

    out_size = get_dir_size_mb(output_dir)
    print(f"  ONNX size: {out_size:.1f} MB ({out_size / src_size * 100:.0f}% of original)")

    if quantize:
        print("  Applying INT8 quantization...")
        try:
            from optimum.onnxruntime import ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig

            onnx_files = [f for f in os.listdir(output_dir) if f.endswith(".onnx")]
            for onnx_file in onnx_files:
                quantizer = ORTQuantizer.from_pretrained(output_dir, file_name=onnx_file)
                qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
                quantizer.quantize(save_dir=output_dir, quantization_config=qconfig)

            q_size = get_dir_size_mb(output_dir)
            print(f"  Quantized size: {q_size:.1f} MB ({q_size / src_size * 100:.0f}% of original)")
        except Exception as e:
            print(f"  ⚠ Quantization failed (ONNX export still valid): {e}")

    print(f"  ✓ Exported to {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export transformer models to ONNX")
    parser.add_argument("--model", choices=["distilbert", "muril", "all"], default="all",
                        help="Which model to export")
    parser.add_argument("--quantize", action="store_true",
                        help="Apply INT8 quantization after export")
    args = parser.parse_args()

    models = {
        "distilbert": (
            os.path.join(PROJECT_ROOT, "model", "distilbert_fakenews"),
            os.path.join(PROJECT_ROOT, "model", "distilbert_fakenews_onnx"),
        ),
        "muril": (
            os.path.join(PROJECT_ROOT, "model", "muril_fakenews"),
            os.path.join(PROJECT_ROOT, "model", "muril_fakenews_onnx"),
        ),
    }

    targets = list(models.keys()) if args.model == "all" else [args.model]
    success = 0
    for name in targets:
        src, dst = models[name]
        print(f"\n{'='*50}")
        print(f"Exporting {name.upper()}")
        print(f"{'='*50}")
        if export_model(src, dst, quantize=args.quantize):
            success += 1

    print(f"\n{'='*50}")
    print(f"Done. {success}/{len(targets)} models exported.")


if __name__ == "__main__":
    main()
