# BAREC 2026 Sentence-Level Strict Track Two-Stage Baseline

Baseline một tệp cho bài toán dự đoán độ khó câu tiếng Ả Rập theo 19 mức của
[BAREC Shared Task 2026](https://barec.camel-lab.com/sharedtask2026). Toàn bộ
pipeline nằm trong `train.py` và có thể chạy bằng một lệnh:

```bash
python train.py
```

Pipeline sử dụng D3Tok thật của CAMeL Tools, một encoder AraBERTv2 và regression
head một chiều. Stage 1 giữ nguyên baseline MSE; Stage 2 load best checkpoint
Stage 1 rồi fine-tune thêm bằng `SoftQWK + 0.05 × MSE`. Hai stage dùng cùng một
model scalar, không có HMTL, auxiliary head, Huber hay CE. Khi Kaggle cung cấp
hai GPU T4, script tự khởi chạy PyTorch DDP; không cần gọi `torchrun` thủ công.

> [!IMPORTANT]
> Checkpoint mặc định
> [`CAMeL-Lab/readability-arabertv2-d3tok-CE`](https://huggingface.co/CAMeL-Lab/readability-arabertv2-d3tok-CE)
> là checkpoint **task-specific**. Trước khi dùng kết quả để nộp Strict Track
> 2026, hãy xác nhận trực tiếp với ban tổ chức rằng lịch sử huấn luyện và việc sử
> dụng checkpoint này được phép. Repository không mặc định coi tên model trên
> Hugging Face là bằng chứng về tính hợp lệ của Strict Track.

## 1. Quy tắc Strict Track

Baseline giữ ranh giới split bắt buộc:

- `train.csv`: split duy nhất được dùng để backpropagation và cập nhật trọng số.
- `dev.csv`: chỉ dùng tính metric, chọn best checkpoint và early stopping.
- `test.csv`: Open Test, chỉ dùng inference. Gold label có sẵn trong bản công
  khai không được dùng để train, tune hyperparameter hoặc chọn checkpoint.
- Blind Test: thay đường dẫn `TEST_PATH`; không thay đổi pipeline và không cần
  gold label.

Không merge Train + Dev, không pseudo-label Test, không dùng nhãn Test và không
dùng SAMER, LLM hay nguồn huấn luyện bổ sung trong baseline này. Người tham gia
vẫn chịu trách nhiệm kiểm tra quy định mới nhất trên website cuộc thi trước khi
nộp bài.

## 2. Dataset và kết quả xác minh

Dữ liệu trong `data/barec-corpus-v1/` đã được đối chiếu toàn bộ giá trị bảng và
thứ tự dòng với dataset chính thức
[`CAMeL-Lab/BAREC-Shared-Task-2026-sent`](https://huggingface.co/datasets/CAMeL-Lab/BAREC-Shared-Task-2026-sent),
revision:

```text
5de96756ba123fe6b02c2728c74f06f43fc9d503
```

| File cục bộ | Split chính thức | Số dòng | Kết quả |
|---|---|---:|---|
| `train.csv` | `train` | 54.845 | Khớp toàn bộ ô và thứ tự |
| `dev.csv` | `validation` | 7.310 | Khớp toàn bộ ô và thứ tự |
| `test.csv` | `test` (Open Test) | 7.286 | Khớp toàn bộ ô và thứ tự |
| **Tổng** |  | **69.441** | **0 ô sai khác** |

Các kiểm tra bổ sung:

- đúng 15 cột theo schema chính thức;
- không có missing value hoặc duplicate ID trong từng split;
- không có ID hay `Document` giao nhau giữa các split;
- `Readability_Level_19` là số nguyên trong `[1, 19]`;
- một số chuỗi câu trùng nội dung giữa các split cũng tồn tại trong bản chính
  thức; đây không phải sai lệch do bản CSV cục bộ.

Schema theo đúng thứ tự:

```text
ID, Sentence, Word_Count, Readability_Level, Readability_Level_19,
Readability_Level_7, Readability_Level_5, Readability_Level_3, Annotator,
Document, Source, Book, Author, Domain, Text_Class
```

Dataset chính thức được phân phối dưới dạng Parquet. Ba file trong repository là
bản CSV UTF-8 tương đương về nội dung bảng. Hash canonical (độc lập với CRLF/LF)
đã dùng trong lần đối chiếu là:

| Split | SHA-256 canonical |
|---|---|
| Train | `40f5d8299610bdc3e7688a7c1376b1ef078d05266cf053b551c6dfefb91f553a` |
| Dev | `5380060472fb84e7b27998c0bd81d225c8e2f91da0d07319d192fe3cc7c34323` |
| Test | `d387c308235b6a8f0646b4703ff8de860b0093f87f076239eff7fb9440fc4476` |

License của dataset là
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Việc repository
lưu bản CSV không thay đổi yêu cầu attribution/share-alike của dữ liệu gốc.

## 3. Cấu trúc repository

```text
.
├── train.py
├── train_blind.py
├── requirements.txt
├── README.md
├── .gitignore
├── .gitattributes
└── data/
    └── barec-corpus-v1/
        ├── dataset_metadata.json
        ├── train.csv
        ├── dev.csv
        └── test.csv
```

`data/` được commit có chủ đích để notebook Kaggle có thể clone và chạy ngay.
Model, cache preprocessing, checkpoint, log và submission không được commit.

## 4. Chạy nhanh trên Kaggle T4 x2

Trong Kaggle Notebook, chọn accelerator **GPU T4 x2**, bật Internet trong giai
đoạn cài dependency/tải model và chạy:

```bash
git clone --branch Khangtest --single-branch \
  https://github.com/POG42069/BAREC_SENT_STRICT_TASK.git
cd BAREC_SENT_STRICT_TASK

python -m pip install -r requirements.txt
camel_data -i light

python train.py --smoke-test
python train.py
```

`python train.py` chạy cả hai stage rồi tạo Open Test submission. Lệnh
`python train_blind.py` cũng chạy đúng hai stage đó, nhưng inference trên Blind
Test và ghi output dưới `outputs/blind/`.

`requirements.txt` cố ý không pin `torch`: Kaggle đã cung cấp PyTorch có CUDA.
Không nên cài lại PyTorch sau đó vì wheel CPU hoặc CUDA không tương thích có thể
làm mất khả năng sử dụng hai T4.

Lệnh `camel_data -i light` cài morphology database và MLE disambiguator cần cho
`calima-msa-r13`. Nếu cấu hình `AUTO_DOWNLOAD_CAMEL_DATA=True`, script cũng thử
chuẩn bị resource khi thiếu; cài thủ công trước vẫn là cách dễ chẩn đoán nhất.

### Chạy Blind Test 2026 riêng tư

`train_blind.py` tải đúng dataset sentence-level riêng tư, chạy cùng pipeline
Stage 1 → Stage 2 bằng Train, chọn model cuối bằng Dev rồi chỉ dùng Blind Test để
inference. Script loại mọi cột giống label trước khi gọi pipeline, nên Blind
không thể tham gia loss, checkpoint selection hoặc metric.

Trên Kaggle, mở **Add-ons → Secrets**, tạo Secret tên `HF_TOKEN`, dán token do
ban tổ chức cấp và bật quyền truy cập cho notebook. Không dán token vào source,
cell notebook, tham số dòng lệnh, `Config`, Git hoặc output log. Sau đó chạy:

```bash
# Chỉ kiểm tra token, quyền truy cập và schema; chưa train.
python train_blind.py --download-only

# Kiểm tra pipeline trên tập con; ZIP này KHÔNG dùng để nộp.
python train_blind.py --smoke-test

# Full Train/Dev rồi inference toàn bộ Blind Test.
python train_blind.py
```

Trong cell Kaggle, gọi các lệnh trên bằng `!python ...` (process riêng); không
dùng `%run train_blind.py` trong chính notebook kernel.

Khi có hai T4, script tự launch hai DDP worker giống `train.py`. Dữ liệu Blind
thô và cache D3Tok được đặt trong `/kaggle/temp` (hoặc thư mục temp của hệ điều
hành), không nằm trong repository hay `/kaggle/working`. Kết quả cần tải về là:

```text
outputs/blind/prediction.zip
```

Nếu ban tổ chức cập nhật dataset trong cùng phiên, dùng
`python train_blind.py --refresh-blind`. Luôn giữ Kaggle Notebook ở chế độ
private; không commit/publish dữ liệu Blind, cache, log chứa mẫu dữ liệu hoặc
token. `eval.py` chỉ dành cho Open Test có gold label, không dùng nó để đánh giá
Blind Test.

## 5. Cài và chạy local

Tạo môi trường ảo, cài **PyTorch phù hợp với CUDA/CPU của máy trước**, rồi cài
các dependency còn lại:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate

# Cài torch theo hướng dẫn tại https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
camel_data -i light
python train.py --smoke-test
```

CPU được hỗ trợ như fallback nhưng D3Tok và fine-tuning toàn bộ corpus sẽ chậm.
Máy phát triển của repository chỉ có một GPU 4 GB, vì vậy không dùng local để
tuyên bố rằng đường chạy NCCL hai T4 đã được xác minh.

## 6. Cấu hình

Người dùng chỉ cần chỉnh `Config` ở đầu `train.py`. Các giá trị quan trọng:

| Nhóm | Trường mặc định | Ý nghĩa |
|---|---|---|
| Data | `TRAIN_PATH`, `DEV_PATH`, `TEST_PATH` | Đường dẫn tương đối với thư mục chứa `train.py` |
| Columns | `ID_COLUMN="ID"`, `TEXT_COLUMN="Sentence"` | ID và câu gốc |
| Label | `LABEL_COLUMN="Readability_Level_19"` | Nhãn 19 mức |
| Model | `MODEL_NAME` | Checkpoint encoder/tokenizer |
| Preprocess | `D3TOK_RESOURCE="calima-msa-r13"` | MLE disambiguator cho D3Tok |
| Length | `MAX_LENGTH=256` | Chiều dài sau HF tokenization |
| Stage 1 batch | `PER_DEVICE_BATCH_SIZE=8`, `GRADIENT_ACCUMULATION_STEPS=2` | Baseline MSE |
| Stage 1 optimizer | `ENCODER_LR=2e-5`, `HEAD_LR=1e-4` | Learning rate encoder/head |
| Stage 1 sampling | `SAMPLER_ALPHA=0.5` | Weighted sampling Train |
| Stage 2 switch | `QWK_FINETUNE_ENABLED=True` | Bật QWK fine-tuning sau Stage 1 |
| Stage 2 epochs | `QWK_FINETUNE_NUM_EPOCHS=2` | Số epoch QWK fine-tuning |
| Stage 2 batch | `QWK_FINETUNE_PER_DEVICE_BATCH_SIZE=16`, `QWK_FINETUNE_GRADIENT_ACCUMULATION_STEPS=1` | Global physical batch 32 trên T4 x2 |
| Stage 2 optimizer | `QWK_FINETUNE_ENCODER_LR=4e-6`, `QWK_FINETUNE_HEAD_LR=2e-5`, `QWK_FINETUNE_WARMUP_RATIO=0.05` | Optimizer/scheduler mới |
| Stage 2 loss | `QWK_FINETUNE_MSE_WEIGHT=0.05`, `SOFT_QWK_TEMPERATURE=1.0` | `SoftQWK + 0.05 × MSE` |
| Stage 2 sampling | `QWK_FINETUNE_USE_WEIGHTED_SAMPLER=False` | Giữ phân phối Train tự nhiên |
| DDP | `DDP_TIMEOUT_MINUTES=180` | Cho phép rank 0 hoàn tất cache D3Tok đầu tiên |
| Cache | `FORCE_REPROCESS=False` | Bỏ cache và D3Tok lại khi bật |
| Resume | `RESUME_FROM_CHECKPOINT=None`, `QWK_FINETUNE_RESUME_FROM_CHECKPOINT=None` | Resume riêng từng stage |

Để dùng Blind Test, chỉ đổi `TEST_PATH` sang CSV/TSV/Parquet mới. File đó phải
có ID và sentence; label là tùy chọn.

## 7. Đọc và kiểm tra dữ liệu

Script hỗ trợ CSV, TSV và Parquet. ID luôn được giữ dạng string. Alias cho ID,
sentence và label được nhận diện, nhưng nếu nhiều alias cùng tồn tại script dừng
với lỗi mơ hồ thay vì tự chọn.

Trước preprocessing, script kiểm tra cột bắt buộc, sentence rỗng, duplicate ID,
nhãn không nguyên/ngoài `[1, 19]`, overlap ID và overlap document. Dev/Test không
shuffle. `original_index` được giữ xuyên suốt để khôi phục đúng thứ tự Test.

## 8. Tiền xử lý tiếng Ả Rập

Thứ tự là một invariant của baseline:

```text
raw sentence
→ Unicode NFKC-compatible normalization
→ xóa Kashida/Tatweel U+0640
→ simple_word_tokenize
→ D3Tok thật (calima-msa-r13)
→ dediac_ar sau D3Tok
→ Hugging Face tokenizer
```

Chi tiết:

1. `normalize_unicode(text, compatibility=True)` chuẩn hóa Unicode.
2. `text.replace("\u0640", "")` xóa đúng Kashida/Tatweel `ـ`; không xóa chữ,
   số hoặc dấu câu.
3. `simple_word_tokenize` tạo word/punctuation sequence.
4. `MLEDisambiguator.pretrained("calima-msa-r13")` kết hợp
   `MorphologicalTokenizer(..., scheme="d3tok", split=True, diac=True)` thực
   hiện D3Tok thật. Không có regex giả lập segmentation.
5. `dediac_ar` chạy **sau D3Tok** để loại tanwin, fatha, damma, kasra, shadda,
   sukun và dagger alif. Dấu `+` do D3Tok tạo ra được giữ nguyên.
6. Các token được ghép lại bằng một khoảng trắng rồi mới đưa vào HF tokenizer.

Nếu riêng một câu làm D3Tok phát sinh exception, fallback bảo toàn câu đã được
Unicode-normalized, bỏ Kashida và bỏ dấu phụ. Script ghi ID/loại lỗi và tổng số
fallback; nó không âm thầm thay bằng chuỗi rỗng và không tạo D3Tok giả.

## 9. Cache preprocessing

Train, Dev và Test có cache Parquet riêng. Fingerprint bao gồm nội dung ID/text,
split, tên cột, phiên bản pipeline, CAMeL Tools và resource D3Tok. Thay đổi
Kashida/dediac/D3Tok sẽ làm cache cũ mất hiệu lực.

Trong DDP, chỉ rank 0 tạo cache bằng file tạm rồi atomic replace; các rank còn
lại chờ barrier trước khi đọc. Bật `FORCE_REPROCESS=True` khi muốn bỏ cache.

## 10. Kiến trúc và objective hai giai đoạn

```text
AutoModel AraBERTv2 encoder
→ last_hidden_state[:, 0, :] (CLS)
→ Dropout
→ Linear(hidden_size, 1)
→ raw readability score
```

Classification head 19 lớp của checkpoint không được sử dụng. Regression head
trả tensor `[batch_size]`. Cả hai stage dùng nguyên kiến trúc scalar này: không
có HMTL, projection/fusion, auxiliary classifier, Huber hoặc classification
head mới. Raw score không được round hoặc clamp trong loss. BERT pooler cũng
không được khởi tạo vì baseline lấy CLS trực tiếp từ `last_hidden_state`; cách
này tránh giữ hai tham số pooler không nối với loss và bảo đảm DDP có gradient
cho mọi tham số trainable.

Stage 1 train 5 epoch như baseline gốc bằng MSE trên nhãn float. Stage 2 strict
load best weights Stage 1, tạo AdamW, scheduler và GradScaler mới rồi train mặc
định thêm 2 epoch với:

```text
SoftQWK + 0.05 × MSE
```

SoftQWK biến mỗi raw score thành phân phối mềm trên tâm lớp `1..19` bằng
`softmax(-(score-k)² / temperature)`, sau đó tối ưu `1 − QWK`. Thành phần này
chạy FP32 ngoài autocast; observed matrix và histogram được cộng differentiably
giữa các rank để loss phản ánh global batch. Nếu global batch chỉ có một gold
class hoặc expected disagreement quá nhỏ, batch đó dùng connected-zero cho
SoftQWK và vẫn backward qua `0.05 × MSE`.

AdamW ở mỗi stage dùng parameter group riêng cho encoder/head, loại bias và
LayerNorm khỏi weight decay, linear warmup và gradient clipping. CUDA dùng FP16
autocast + GradScaler cho model forward; baseline không yêu cầu BF16 trên T4.

## 11. Weighted sampling

Weighted sampling chỉ áp dụng Train trong Stage 1:

```text
class_weight[c] = (1 / class_count[c]) ** 0.5
```

Sampler dùng replacement, `seed + epoch`, có `set_epoch`, và phân phối cùng số
sample/step cho mọi rank. Stage 2 tắt weighted sampler và dùng distributed random
sampling thông thường để SoftQWK quan sát phân phối Train tự nhiên. Dev và Test
luôn giữ phân phối, ID và thứ tự gốc.

## 12. Tự động DDP trên hai T4

Khi `python train.py` thấy ít nhất hai GPU và chưa ở worker mode, nó tự chạy lại:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 train.py --ddp-worker
```

Trên Kaggle/Linux CUDA, backend là NCCL. Mỗi process gắn với một `LOCAL_RANK`,
model được bọc `DistributedDataParallel`, và chỉ rank 0 hiển thị progress/lưu
file. Một GPU hoặc CPU dùng single-process fallback.

Effective batch Stage 1 mặc định trên T4 x2:

```text
8 per-device × 2 GPU × 2 accumulation = 32
```

Global physical batch Stage 2 mặc định:

```text
16 per-device × 2 GPU × 1 accumulation = 32
```

Với 54.845 câu Train và T4 x2, Stage 2 có khoảng `1713` iteration mỗi epoch
(khác `3427` iteration của Stage 1 vì batch mỗi GPU tăng từ 8 lên 16).

Stage 2 khóa accumulation ở `1`: trung bình loss SoftQWK từ nhiều micro-batch
không tương đương tính SoftQWK một lần trên batch lớn.

Việc triển khai DDP không đồng nghĩa đã xác minh runtime Kaggle. Hãy chạy
`--smoke-test` trên chính notebook T4 x2 và kiểm tra log có hai rank trước khi
full training.

## 13. Training, evaluation và chọn model cuối

Chỉ Train DataLoader thực hiện backward. Luồng mặc định là:

1. Stage 1 train 5 epoch bằng MSE, weighted sampler `alpha=0.5`, encoder LR
   `2e-5`, head LR `1e-4`, batch 8/GPU và accumulation 2.
2. Chọn best Stage 1 bằng Dev QWK; nếu hòa thì MAE thấp hơn thắng.
3. Strict-load best Stage 1, đánh giá nó làm ứng viên ban đầu cho model cuối và
   reset optimizer/scheduler/scaler.
4. Stage 2 train 2 epoch bằng `SoftQWK + 0.05 × MSE`, encoder LR `4e-6`, head LR
   `2e-5`, warmup `0.05`, batch 16/GPU, accumulation 1 và sampler weighted OFF.
5. Sau mỗi epoch Stage 2, chỉ thay model cuối nếu Dev QWK cao hơn; khi QWK hòa,
   MAE thấp hơn phải thắng. Nếu Stage 2 không cải thiện, pipeline tự động quay
   về best Stage 1.

Sau mỗi epoch, Dev được gather theo `original_index`, sắp lại và loại
padding/duplicate do phân phối trước khi tính:

- raw MSE;
- MAE;
- exact accuracy;
- adjacent accuracy (sai lệch tối đa 1 mức);
- Quadratic Weighted Kappa (QWK) với label cố định `1..19`.

Prediction cho metric được tính bằng `np.rint`, clip vào `[1, 19]`, rồi chuyển
sang integer. Early stopping và model selection chỉ dựa trên Dev, không nhìn
Open Test hoặc Blind Test.

Mỗi stage lưu `last.pt` riêng với model, optimizer, scheduler, scaler,
epoch/global step, best QWK/MAE, config và RNG state. Dùng
`RESUME_FROM_CHECKPOINT` cho Stage 1 hoặc
`QWK_FINETUNE_RESUME_FROM_CHECKPOINT` cho Stage 2; không đặt đồng thời hai
đường dẫn. Resume Stage 2 yêu cầu best Stage 1 vẫn tồn tại để pipeline luôn có
checkpoint fallback. Model state được lưu sau khi unwrap DDP nên load được trên
một hoặc nhiều GPU.

## 14. Smoke test và kiểm tra tối thiểu

Kiểm tra cú pháp:

```bash
python -m py_compile train.py train_blind.py
```

Smoke test thật (mẫu nhỏ, D3Tok/checkpoint thật, output riêng):

```bash
python train.py --smoke-test
```

Smoke mode chạy tập con qua cả Stage 1 và Stage 2 (mỗi stage một epoch ngắn), rồi
kiểm tra đọc dữ liệu, Kashida removal, D3Tok, dediac, HF tokenization, sampler,
MSE/SoftQWK forward-backward, checkpoint/fallback/reload, inference và submission
ZIP. Nó không phải một lần huấn luyện hợp lệ để báo cáo QWK.

Các invariant cần đạt:

- không còn `U+0640` sau preprocessing;
- không còn Arabic diacritics sau `dediac_ar`;
- dấu `+` của D3Tok không bị xóa;
- output model giữ shape `[batch_size]`, kể cả batch size 1;
- Stage 1 sampler dùng `alpha=0.5`; Stage 2 xác nhận weighted sampler tắt;
- SoftQWK finite, chạy FP32 và vẫn có gradient khi DDP;
- gather trả đúng số mẫu theo thứ tự gốc;
- submission validator từ chối header/ID/range/ZIP sai.

## 15. Output và định dạng nộp bài

Sau full training:

```text
outputs/
├── stage1/
│   ├── best_model/
│   ├── checkpoints/
│   │   └── last.pt
│   └── logs/
│       └── training_history.csv
├── stage2/
│   ├── best_model/
│   ├── checkpoints/
│   │   └── last.pt
│   └── logs/
│       └── training_history.csv
├── best_model/                 # model cuối: Stage 1 hoặc Stage 2
├── selection.json
├── logs/
│   └── preprocessing_report.json
├── diagnostics/
│   └── test_predictions_with_raw_scores.csv
├── prediction
└── prediction.zip
```

`selection.json` ghi stage/checkpoint cuối được chọn và metric Dev dùng cho quyết
định fallback. Với Blind, cùng cây Stage 1/Stage 2 và file selection nằm dưới
`outputs/blind/`; submission cuối là `outputs/blind/prediction.zip`.

File nộp gốc là `outputs/prediction`: CSV UTF-8 không BOM và **không có phần mở
rộng**. Header phải chính xác:

```csv
Sentence ID,Prediction
10100290001,7
10100290002,1
10100290003,8
```

Quy tắc:

- `Sentence ID` là string và đúng giá trị/thứ tự Test;
- `Prediction` là integer trong `[1, 19]`;
- không có DataFrame index, raw score, gold label hoặc cột phụ.

`outputs/prediction.zip` chứa trực tiếp đúng một entry tên `prediction`, không có
thư mục cha và không có file khác. Script mở lại ZIP và xác minh filename,
header, số dòng, ID/order, duplicate/missing ID và prediction range trước khi báo
thành công. Raw score (và gold Open Test nếu có) chỉ nằm trong diagnostics.

## 16. Troubleshooting

### Không tìm thấy `calima-msa-r13`

```bash
camel_data -i light
```

Sau đó chạy lại. Nếu môi trường không có Internet, tải CAMeL data/model vào một
phiên có Internet hoặc gắn cache hợp lệ; baseline không thay D3Tok bằng regex.

### CUDA out of memory

Nếu Stage 1 OOM, giảm `PER_DEVICE_BATCH_SIZE` hoặc `MAX_LENGTH` và có thể tăng
`GRADIENT_ACCUMULATION_STEPS` để giữ effective batch. Nếu Stage 2 batch 16/GPU
OOM, giảm `QWK_FINETUNE_PER_DEVICE_BATCH_SIZE` xuống 8 nhưng giữ accumulation
bằng `1`; không dùng accumulation để mô phỏng global SoftQWK batch. Xóa
checkpoint dở nếu không resume.

### Kaggle chỉ nhận một GPU

Kiểm tra accelerator là **GPU T4 x2**, restart session và kiểm tra:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.device_count())"
```

### NCCL/DDP lỗi hoặc treo

Kiểm tra hai process dùng cùng code/cache, không để nhiều rank cùng preprocess
hoặc ghi output, và xem log rank đầu tiên phát sinh lỗi. Chạy smoke test trước;
không full-train để che một lỗi khởi tạo DDP. Nếu log báo
`Expected to have finished reduction ... parameters that were not used`, hãy
`git pull origin Khangtest` để lấy bản đã loại BERT pooler rồi chạy lại; cache
D3Tok đã hoàn thành vẫn có thể tái sử dụng.

### `torch.cuda.is_available()` là `False`

Không cài lại `torch` bằng một wheel CPU sau khi khởi động Kaggle. Tạo session
mới hoặc cài đúng CUDA build theo hướng dẫn PyTorch.

### Sai cột hoặc Blind Test không có label

Đổi `ID_COLUMN`/`TEXT_COLUMN` trong `Config`. Blind Test không cần label; Train
và Dev bắt buộc có label 19 mức.

### Checkpoint không load được

Đảm bảo checkpoint và config/model name cùng scalar baseline, không trộn
checkpoint HMTL/ensemble cũ. Stage 1 và Stage 2 có optimizer state riêng. Nếu
không cần resume, đặt cả `RESUME_FROM_CHECKPOINT` và
`QWK_FINETUNE_RESUME_FROM_CHECKPOINT` về `None`.

### ZIP bị hệ thống chấm từ chối

Không tự đổi tên thành `prediction.csv`. Mở ZIP và xác nhận nó chỉ chứa
`prediction` tại root với header `Sentence ID,Prediction`.

## 17. Reproducibility và báo cáo kết quả

Seed được đặt cho Python, NumPy, PyTorch CPU/CUDA, DataLoader và sampler. Tuy vậy,
khác biệt CUDA/library vẫn có thể gây sai khác nhỏ. Hãy lưu config, commit hash,
log và best metrics đi kèm mỗi run.

Repository không ghi QWK hoặc runtime chưa đo, không tuyên bố NCCL/T4 x2 đã đạt
nếu chưa chạy trên Kaggle, và không dùng metric smoke test như kết quả cuộc thi.

## 18. Citation

Khi sử dụng dữ liệu, giữ attribution theo dataset card chính thức và trích dẫn:

```bibtex
@inproceedings{elmadani-etal-2025-readability,
  title     = {A Large and Balanced Corpus for Fine-grained Arabic Readability Assessment},
  author    = {Elmadani, Khalid N. and Habash, Nizar and Taha-Thomure, Hanada},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  year      = {2025},
  address   = {Vienna, Austria},
  publisher = {Association for Computational Linguistics}
}

@inproceedings{habash-etal-2025-guidelines,
  title     = {Guidelines for Fine-grained Sentence-level Arabic Readability Annotation},
  author    = {Habash, Nizar and Taha-Thomure, Hanada and Elmadani, Khalid N. and Zeino, Zeina and Abushmaes, Abdallah},
  booktitle = {Proceedings of the 19th Linguistic Annotation Workshop (LAW-XIX)},
  year      = {2025},
  address   = {Vienna, Austria},
  publisher = {Association for Computational Linguistics}
}
```

Nguồn chính thức:

- [BAREC Shared Task 2026](https://barec.camel-lab.com/sharedtask2026)
- [BAREC 2026 sentence-level dataset](https://huggingface.co/datasets/CAMeL-Lab/BAREC-Shared-Task-2026-sent)
- [Checkpoint mặc định](https://huggingface.co/CAMeL-Lab/readability-arabertv2-d3tok-CE)
