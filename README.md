# ViFact: Vietnamese Fact-Checking System

ViFact là hệ thống kiểm chứng thông tin tự động end-to-end dành cho tiếng Việt. Hệ thống kết hợp các kỹ thuật truy hồi lai, suy luận đa bước và sinh giải thích để cung cấp đánh giá chính xác và đáng tin cậy.

## 🌟 Tính năng chính

- Truy hồi lai (Hybrid Retrieval): kết hợp BM25, bi-encoder và cross-encoder
- Suy luận đa bước: tổng hợp thông tin từ nhiều bằng chứng
- Sinh giải thích: tạo lời giải thích trung thực, có thể truy xuất
- Ngưỡng tin cậy thích nghi: điều chỉnh độ tin cậy theo miền/bối cảnh
- Gắn nhãn nguồn: trích dẫn rõ ràng và phát hiện phản chứng

## 🧱 Kiến trúc hệ thống (khái quát)

```
ViFact/
├── vifact/
│   ├── modules/
│   │   ├── retrieval/       # BM25, Dense, Hybrid, Reranker
│   │   ├── confidence/      # Hiệu chuẩn & ngưỡng
│   │   ├── rationale/       # Trích xuất luận cứ (QATC)
│   │   ├── reasoner/        # Hợp nhất đa bằng chứng
│   │   ├── explainer/       # Sinh giải thích
│   │   └── attribution/     # Gắn nhãn nguồn
│   ├── data/                # Loader, tiền xử lý, validate
│   ├── utils/               # Logging, tiện ích
│   ├── config.py            # Cấu hình
│   └── cli.py               # Lệnh xử lý dữ liệu
├── data/
│   ├── raw/                 # Dữ liệu thô (append-only)
│   ├── processed/           # Dữ liệu đã xử lý (shards)
│   └── corpus/              # Corpus văn bản
├── config/                  # YAML cấu hình
├── models/                  # Chỉ mục/ mô hình đã build
└── scripts/                 # Tiện ích pipeline
```

## 📦 Dữ liệu

- ViFactCheck: 7,232 cặp tuyên bố–bằng chứng
- ViWikiFC: Wikipedia tiếng Việt (>20k tuyên bố)
- Vietnamese News Corpus: kho văn bản báo chí
- VLSP Datasets: bổ sung NLI, MRC

## 🚀 Bắt đầu nhanh

### Yêu cầu hệ thống
- Python 3.8+
- PyTorch 1.9+
- Transformers 4.20+
- CUDA 11.0+ (khuyến nghị)

### Cài đặt

```bash
# Clone repository
git clone https://github.com/your-username/vifact.git
cd vifact

# Cài đặt dependencies
pip install -r requirements.txt

# Hoặc cài đặt từ source
pip install -e .
```

### Demo nhanh (minh họa)

```bash
# Chạy demo kiểm tra cài đặt
python demo.py

# Hoặc sử dụng CLI
vifact demo
```

### Sử dụng cơ bản (minh họa)

#### 1. Python API

```python
from vifact import ViFact, ViFactConfig

config = ViFactConfig.from_yaml("config/vifact_config.yaml")
vifact = ViFact(config)
vifact.load_modules()
vifact.load_data()

claim = "Việt Nam có 54 dân tộc."
result = vifact.verify(claim)

print(f"Verdict: {result.verdict}")
print(f"Confidence: {result.confidence:.3f}")
print(f"Explanation: {result.explanation}")
print(f"Sources: {len(result.sources)} sources found")
```

#### 2. Command Line

```bash
vifact verify "Việt Nam có 54 dân tộc."
vifact train --data-path data/raw/ise-dsc01-warmup.json --epochs 10
vifact evaluate --model-path models/vifact-trained
```

### Huấn luyện & đánh giá (minh họa)

```bash
python train.py --config config/vifact_config.yaml --epochs 10
vifact train --epochs 10 --batch-size 16

python evaluate.py --model-path models/vifact-trained
vifact evaluate --model-path models/vifact-trained
```

## 📈 Kết quả dự kiến

| Metric | ViFact-Base | ViFact-Large |
|--------|-------------|--------------|
| Accuracy | 85.2% | 88.7% |
| Evidence Recall | 92.1% | 94.3% |
| Faithfulness | 89.4% | 91.8% |

## ⚙️ Cấu hình

Hệ thống dùng YAML để điều chỉnh tham số (xem thêm trong `config/vifact_config.yaml`).

```yaml
# config/vifact_config.yaml
system:
  name: "ViFact"
  version: "0.1.0"
  device: "auto"
```

### Cài đặt môi trường phát triển

```bash
# Clone và cài đặt
git clone https://github.com/your-username/vifact.git
cd vifact

# Cài đặt dependencies phát triển
pip install -r requirements.txt
pip install -e ".[dev]"

# Chạy tests
pytest tests/

# Format code (nếu dùng)
black vifact/
flake8 vifact/
```

### Roadmap phát triển

Xem [ROADMAP.md](ROADMAP.md) để biết kế hoạch chi tiết.

## 📚 Trích dẫn

```bibtex
@article{vifact2026,
  title={ViFact: An End-to-End Vietnamese Fact-Checking System with Explainable Multi-Step Reasoning},
  author={Your Name and Collaborators},
  journal={Conference/Journal Name},
  year={2026}
}
```

## 📝 Giấy phép

MIT License - xem file `LICENSE` để biết thêm chi tiết.

## 🤝 Đóng góp

Chúng tôi hoan nghênh mọi đóng góp! Vui lòng:

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Tạo Pull Request

## 📫 Liên hệ

- Email: your.email@domain.com
- GitHub Issues: [Link to issues page]
- Documentation: [Link to docs]

## 🙏 Acknowledgments

- ViFactCheck dataset creators
- Vietnamese NLP community
- Open source contributors

## 🔧 Chuẩn bị dữ liệu cho Retrieval (Phase 1)

Các bước ngắn gọn để xử lý dữ liệu lớn, xây corpus và chỉ mục:

1) Tiền xử lý + chia nhỏ (sharding)

```bash
python -m vifact.cli process \
  --input data/raw/ise-dsc01-warmup.json \
  --output data/processed \
  --skip-invalid
```

2) Tạo splits

```bash
python -m vifact.cli split \
  --input data/processed \
  --output data/processed/splits \
  --train 0.8 --val 0.1 --seed 42
```

3) Xây corpus văn bản (dùng cho indexing)

```bash
python scripts/build_corpus.py \
  --input data/processed \
  --output data/corpus/docs.jsonl
```

4) Chỉ mục BM25 và Dense

```bash
# BM25
python scripts/build_index.py \
  --corpus data/corpus/docs.jsonl \
  --output models/bm25.pkl.gz

# Dense bi-encoder
python scripts/build_dense_index.py \
  --corpus data/corpus/docs.jsonl \
  --output models/dense_index.npz
```

5) Kiểm tra retrieval nhanh

```bash
# BM25 / Dense / Hybrid
python scripts/retrieval_eval.py --mode bm25  --index models/bm25.pkl.gz           --data data/processed --k 10 --limit 1000
python scripts/retrieval_eval.py --mode dense  --dense-index models/dense_index.npz --data data/processed --k 10 --limit 1000
python scripts/retrieval_eval.py --mode hybrid --index models/bm25.pkl.gz --dense-index models/dense_index.npz --data data/processed --k 10 --limit 1000
```

Gợi ý: Dùng `JSONL` + `ijson` để stream dữ liệu lớn; đặt file gốc ở `data/raw/` (append-only), chuẩn hóa Unicode (UTF-8, NFKC). Tinh chỉnh tham số trong `config/vifact_config.yaml` (`retrieval.bm25.k1/b`, `retrieval.dense.model_name/batch_size`).

## ✅ Tiêu chí hoàn thành tiền xử lý

- Schema hợp lệ: không còn lỗi thiếu trường (`claim/context/verdict`), `verdict ∈ {SUPPORTED, REFUTED, NEI}`.
- Shards sẵn sàng: có `data/processed/shard_*.jsonl`, đếm bản ghi khớp (trừ record bị bỏ qua).
- Splits đủ: `train/val/test.jsonl` phân phối hợp lý.
- Corpus nhất quán: `data/corpus/docs.jsonl` gồm `doc_id,text`, trùng lặp thấp.
- Index sẵn sàng: `models/bm25.pkl.gz` và/hoặc `models/dense_index.npz` tạo thành công (N, avgdl hợp lý).
- Sanity-check: `recall@10` và `mrr` > 0 trên mẫu kiểm tra; không lỗi runtime.

## 🛡️ Confidence: Hiệu chuẩn & Ngưỡng (Phase 2)

Chuẩn hoá xác suất và đặt ngưỡng tự tin (abstain → NEI) để giảm dự đoán sai khi độ tin cậy thấp.

1) Hiệu chuẩn nhiệt độ (Temperature Scaling)

```bash
python scripts/confidence_calibrate.py \
  --preds data/val_preds.jsonl \
  --out models/confidence/calibration.json \
  --labels SUPPORTED REFUTED NEI
```

2) Tối ưu ngưỡng tin cậy (toàn cục + theo domain)

```bash
python scripts/confidence_tune.py \
  --preds data/val_preds.jsonl \
  --out models/confidence/thresholds.json \
  --labels SUPPORTED REFUTED NEI \
  --metric f1 \
  --calibration-json models/confidence/calibration.json
```

3) Áp dụng vào dự đoán (sản xuất/test)

```bash
python scripts/confidence_apply.py \
  --preds data/test_preds.jsonl \
  --out data/test_preds_final.jsonl \
  --labels SUPPORTED REFUTED NEI \
  --calibration-json models/confidence/calibration.json \
  --thresholds-json models/confidence/thresholds.json
```

Đầu vào dự đoán (`*.jsonl`): mỗi dòng cần `id`, (tuỳ chọn) `domain`, và `logits` hoặc `probs`. Khi calibrate/tune nên có thêm `label` để tính ECE/NLL/Brier.

### 💡 Mẹo mã hóa (Windows)
- Nên dùng UTF-8 để hiển thị tiếng Việt chính xác trong PowerShell:
  - `chcp 65001`
  - `$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()`

## 🧠 Rationale & Explanation (Phase 2)

Huấn luyện trích xuất rationale (câu) và tạo lời giải thích ngắn kèm nguồn.

- Train trên Colab: xem `PHASE2_COLAB.md`
- Dự đoán rationale:

```bash
python scripts/predict_rationale.py \
  --data data/processed \
  --model /path/to/rationale_model \
  --out data/processed/rationales.jsonl \
  --topk 2
```

- Tạo explanation với retrieval + rationale:

```bash
python scripts/generate_explanations.py \
  --data data/processed \
  --out data/processed/explanations.jsonl \
  --bm25-index models/bm25.pkl.gz \
  --dense-index models/dense_index.npz \
  --rationales data/processed/rationales.jsonl \
  --k 5 --k-sparse 150 --k-dense 150 --w-sparse 0.7 --w-dense 0.3
```

- Đầu ra: mỗi dòng gồm `id, claim, verdict (ground truth), rationale, sources[], explanation`.
