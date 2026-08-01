# BAREC 2026 Sentence-Level Strict Track

Baseline một tệp cho bài toán dự đoán độ khó câu tiếng Ả Rập theo 19 mức của
[BAREC Shared Task 2026](https://barec.camel-lab.com/sharedtask2026). Toàn bộ
pipeline nằm trong `train.py` và có thể chạy bằng một lệnh:

```bash
python train.py
```

Pipeline sử dụng D3Tok thật của CAMeL Tools và ensemble năm AraBERTv2 cùng kiến
trúc, được fine-tune độc lập với seed `42, 52, 62, 72, 82`. Mặc định, mỗi model
học chung trong một forward/backward pass theo cascade
`3 mức → 5 mức → 7 mức → regression 19 mức`. Kết quả cuối là trung bình
đều của năm raw score rồi dùng `np.floor`, clip vào `[1, 19]` và xuất thành
`prediction_down.zip`. Mỗi seed cũng chọn best checkpoint bằng Dev QWK sau
`floor`; hòa QWK thì Dev MAE của bản `floor` thấp hơn thắng. Không tối ưu
threshold và không học ensemble weight. Khi Kaggle cung cấp hai GPU T4,
script tự khởi chạy PyTorch DDP; không cần gọi `torchrun` thủ công.

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
- `Readability_Level_3`, `Readability_Level_5` và `Readability_Level_7` khớp
  đúng mapping phân cấp suy ra từ nhãn 19 mức;
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
camel_data -i disambig-bert-unfactored-msa

python train.py --smoke-test
python train.py
```

`requirements.txt` cố ý không pin `torch`: Kaggle đã cung cấp PyTorch có CUDA.
Không nên cài lại PyTorch sau đó vì wheel CPU hoặc CUDA không tương thích có thể
làm mất khả năng sử dụng hai T4.

Lệnh `camel_data -i disambig-bert-unfactored-msa` cài BERT unfactored
disambiguator MSA dùng để tạo D3Tok giống pipeline công khai của SBTW. Nếu cấu
hình `AUTO_DOWNLOAD_CAMEL_DATA=True`, script cũng thử chuẩn bị resource khi
thiếu; cài thủ công trước vẫn là cách dễ chẩn đoán nhất.

### Chạy Blind Test 2026 riêng tư

`train_blind.py` tải đúng dataset sentence-level riêng tư, huấn luyện bằng Train,
chọn checkpoint bằng Dev rồi chỉ dùng Blind Test để inference. Script loại mọi
cột giống label trước khi gọi pipeline, nhưng giữ `ID`, `Sentence`, `Document`,
`Domain` và `Text_Class`; vì vậy Blind dùng metadata thật cho structured input
mà vẫn không thể tham gia loss, checkpoint selection hoặc metric. `Word_Count`
được tính lại từ Surface view như ở các split công khai.

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
outputs/blind/prediction_down.zip
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
camel_data -i disambig-bert-unfactored-msa
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
| Objective | `MODEL_MODE="cascaded_hmtl"` | Cascade multi-task mặc định; `baseline_mse` giữ tương thích baseline cũ |
| Auxiliary labels | `LABEL_3_COLUMN`, `LABEL_5_COLUMN`, `LABEL_7_COLUMN` | Cột nhãn phân cấp dùng cho Train/Dev |
| Auxiliary loss | `AUX_CE3/5/7_WEIGHT=0.5` | Trọng số Cross Entropy của ba head phân loại |
| Cascade | `CASCADE_TEMPERATURE=1.0` | Temperature cho softmax truyền sang head kế tiếp |
| Preprocess | `D3TOK_RESOURCE="msa"` | BERT unfactored disambiguator cho D3Tok |
| D3Tok batch | `D3TOK_BATCH_SIZE=256` | Số câu đưa qua BERT disambiguator mỗi lượt |
| Length | `MAX_LENGTH=512` | Giới hạn đầy đủ của BERT; D3Tok không bị truncate âm thầm |
| Batch | `PER_DEVICE_BATCH_SIZE=8` | Batch trên mỗi GPU |
| Accumulation | `GRADIENT_ACCUMULATION_STEPS=2` | Số micro-batch mỗi optimizer step |
| Optimizer | `ENCODER_LR=2e-5`, `HEAD_LR=1e-4` | Learning rate riêng |
| Sampling | `USE_WEIGHTED_SAMPLER=True`, `SAMPLER_ALPHA=0.5` | Weighted sampling theo nhãn 19 mức |
| Ensemble | `ENSEMBLE_SEEDS=(42, 52, 62, 72, 82)` | Năm lần fine-tune độc lập trên cùng Train |
| Seed artifacts | `SEED_RUNS_DIR="outputs/seeds"` | Best model/checkpoint/log riêng của từng seed |
| DDP | `DDP_TIMEOUT_MINUTES=180` | Cho phép rank 0 hoàn tất cache D3Tok đầu tiên |
| Cache | `FORCE_REPROCESS=False` | Bỏ cache và D3Tok lại khi bật |
| Resume | `RESUME_FROM_CHECKPOINT=None` | Template checkpoint có placeholder `{seed}` để tiếp tục từng member |

`train.py` dùng `TEST_PATH` cho Open Test. Blind Test riêng tư được tải và chuẩn
hóa bởi `train_blind.py`, không cần sửa đường dẫn trong source.

## 7. Đọc và kiểm tra dữ liệu

Script hỗ trợ CSV, TSV và Parquet. ID luôn được giữ dạng string. Alias cho ID,
sentence và label được nhận diện, nhưng nếu nhiều alias cùng tồn tại script dừng
với lỗi mơ hồ thay vì tự chọn.

Trước preprocessing, script kiểm tra cột bắt buộc, sentence rỗng, duplicate ID,
nhãn không nguyên/ngoài range, overlap ID và overlap document. Train/Dev bắt buộc
có đủ nhãn 3/5/7/19 và phải khớp mapping sau; auxiliary label được đổi về
zero-based ngay trước Cross Entropy:

```text
19: 1–4 | 5–7 | 8–9 | 10–11 | 12–13 | 14–15 | 16–19
 7:   1  |  2  |  3  |   4   |   5   |   6   |   7

19: 1–7 | 8–11 | 12–13 | 14–15 | 16–19
 5:   1  |   2  |   3   |   4   |   5

19: 1–11 | 12–13 | 14–19
 3:    1   |   2   |   3
```

Open/Blind Test không cần auxiliary label để inference. Dev/Test không shuffle;
`original_index` được giữ xuyên suốt để khôi phục đúng thứ tự Test.

## 8. Tiền xử lý tiếng Ả Rập

Thứ tự là một invariant của baseline:

```text
raw sentence
→ normalize_unicode(compatibility=True)
→ bỏ Kashida U+0640 nhưng vẫn giữ dấu phụ
→ đổi alif-maqsura nội từ thành ya như SBTW
→ simple_word_tokenize(split_digits=True)
→ gắn dấu phụ đứng riêng vào từ trước; bỏ mark-only token còn sót
→ BERTUnfactoredDisambiguator("msa", top=1)
→ lấy analysis["d3tok"]
→ dediac_ar và chuyển biên `_+`/`+_` → D3Tok view

song song từ câu đã normalize:
→ tính DC trước khi bỏ dấu phụ
→ dediac_ar, giữ dấu câu → Surface view
→ tính WC, WLA, WLS trên token của Surface view
```

Chi tiết:

1. Unicode được chuẩn hóa ở compatibility mode; chỉ Kashida bị xóa ở bước đầu,
   nên BERT D3Tok vẫn nhận được dấu phụ của câu gốc.
2. Alif-maqsura `ى` ở giữa từ được đổi thành ya `ي`, giống code SBTW.
3. `simple_word_tokenize(..., split_digits=True)` tạo word/punctuation sequence.
4. Các dạng lỗi khoảng trắng như `الله ُ` được sửa thành token `اللهُ`. Dấu phụ
   đứng đầu câu hoặc sau dấu câu, không có base letter để phân tích hình thái,
   bị loại khỏi input D3Tok nhưng vẫn được tính trong `[DC]`.
5. `BERTUnfactoredDisambiguator.pretrained(model_name="msa",
   pretrained_cache=False, top=1)` chọn phân tích hình thái theo ngữ cảnh.
6. Pipeline lấy trường `analysis["d3tok"]`, chạy `dediac_ar`, rồi chuyển `_+`
   thành ` +` và `+_` thành `+ ` đúng theo preprocessing công khai của SBTW.
7. Dấu câu được giữ trong cả D3Tok view và Surface view.

Bốn feature số được tính theo đúng thời điểm:

- `[DC]`: số Arabic diacritics chia cho số ký tự của câu đã normalize, trước
  `dediac_ar`;
- `[WC]`: số token trong Surface view;
- `[WLA]`: độ dài token trung bình trong Surface view;
- `[WLS]`: độ lệch chuẩn độ dài token trong Surface view.

Input cuối cùng là một sentence pair:

```text
[CLS] D3Tok view [SEP]
Surface view [SEP]
[WC] value [DC] value [WLA] value [WLS] value
[DOM_*] [TC_*] [SEP]
```

`Domain` được ánh xạ vào `DOM_AH`, `DOM_SS`, `DOM_STEM`; `Text_Class` được ánh
xạ vào `TC_FOUNDATIONAL`, `TC_ADVANCED`, `TC_SPECIALIZED`. Nếu Blind Test thiếu
hai cột này, pipeline dùng token `UNKNOWN` tương ứng. `Document`, `Book`,
`Author` và `Annotator` không được concat vì cardinality cao hoặc có nguy cơ học
thuộc nguồn dữ liệu.

Các field marker là token học được, được thêm vào tokenizer trước khi model
khởi tạo; embedding của encoder được resize cho khớp vocabulary mới. Pipeline
thử giữ nguyên D3Tok, Surface và toàn bộ feature. Nếu tổng vượt 512 token, các
feature được bỏ nguyên nhóm từ cuối danh sách ưu tiên:
`Text_Class → Domain → WLS → WLA → DC → WC`. Vì vậy một label như `[WLA]`
không bao giờ còn lại mà thiếu value đi kèm. Chỉ sau khi đã bỏ mọi feature mà
input vẫn quá dài, Surface mới bị truncate. D3Tok không bị truncate âm thầm;
nếu riêng D3Tok đã vượt giới hạn, pipeline dừng với chẩn đoán cần chunking.

Nếu một token hợp lệ vẫn không có trường `d3tok`, chỉ token đó dùng surface
fallback; các token khác trong câu vẫn giữ D3Tok. Chỉ khi toàn bộ lời gọi BERT
cho câu phát sinh exception thì pipeline mới dùng Surface view cho cả câu.
Script ghi ID/loại lỗi và tổng số fallback; nó không âm thầm thay bằng chuỗi
rỗng và không tạo D3Tok giả.

## 9. Cache preprocessing

Train, Dev và Test có cache Parquet riêng. Fingerprint bao gồm nội dung ID/text,
split, tên cột, phiên bản pipeline, CAMeL Tools và resource D3Tok. Việc chuyển
sang hai text view cùng feature mới làm toàn bộ cache cũ tự động mất hiệu lực.

Trong DDP, chỉ rank 0 tạo cache bằng file tạm rồi atomic replace; các rank còn
lại chờ barrier trước khi đọc. Bật `FORCE_REPROCESS=True` khi muốn bỏ cache.

Checkpoint cũ có kích thước embedding trước khi thêm field token nên không tương
thích để resume. Hãy bắt đầu một output/checkpoint mới cho pipeline này.

## 10. Kiến trúc và objective

```text
AutoModel AraBERTv2 encoder
→ h = last_hidden_state[:, 0, :] (CLS)
├─→ logits3 = Linear(H, 3)(Dropout(h))
│   └─→ p3 = softmax(logits3 / 1.0)
├─→ logits5 = Linear(H + 3, 5)(Dropout(concat(h, p3)))
│   └─→ p5 = softmax(logits5 / 1.0)
├─→ logits7 = Linear(H + 5, 7)(Dropout(concat(h, p5)))
│   └─→ p7 = softmax(logits7 / 1.0)
└─→ score19 = Linear(H + 7, 1)(Dropout(concat(h, p7)))
```

`MODEL_MODE="cascaded_hmtl"` là mặc định. Mỗi head sau nhận cả CLS gốc và phân
phối xác suất của head ngay trước, nên coarse prediction là tín hiệu bổ sung chứ
không phải bottleneck duy nhất. Không dùng `argmax`, `round`, gold label hoặc
`.detach()` trong cascade. Vì vậy loss 19 mức truyền gradient qua Head 7, Head 5,
Head 3 và encoder. Forward trả `score19`, `logits3`, `logits5`, `logits7`.

Tất cả objective được tính cùng một batch và chỉ gọi backward một lần:

```python
total_loss = (
    mse(score19, label19)
    + 0.5 * cross_entropy(logits3, label3)
    + 0.5 * cross_entropy(logits5, label5)
    + 0.5 * cross_entropy(logits7, label7)
)
```

Loss được tính FP32; encoder vẫn dùng FP16 autocast + GradScaler trên CUDA. Raw
score không được round trong loss. Classification head 19 lớp của checkpoint và
BERT pooler không được sử dụng. Mode `baseline_mse` giữ
`ArabicReadabilityRegressor` một head để chạy ablation/checkpoint baseline cũ;
checkpoint giữa hai mode không được load lẫn nhau.

AdamW dùng parameter group riêng cho encoder/head, loại bias và LayerNorm khỏi
weight decay, linear warmup, gradient clipping và gradient accumulation. CUDA
dùng FP16 autocast + GradScaler; pipeline không yêu cầu BF16 trên T4.

## 11. Train sampling

Weighted sampling được bật mặc định và dựa trên nhãn 19 mức:

```text
USE_WEIGHTED_SAMPLER = True
sample_weight(class) = (1 / class_count) ** 0.5
```

Sampler dùng replacement, deterministic theo `seed + epoch`, và chia đều số
sample/step cho từng DDP rank. Dev và Test không weighted sample, giữ ID và thứ
tự gốc. Có thể đặt `USE_WEIGHTED_SAMPLER=False` chỉ khi chạy ablation baseline.

## 12. Tự động DDP trên hai T4

Khi `python train.py` thấy ít nhất hai GPU và chưa ở worker mode, nó tự chạy lại:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 train.py --ddp-worker
```

Trên Kaggle/Linux CUDA, backend là NCCL. Mỗi process gắn với một `LOCAL_RANK`,
model được bọc `DistributedDataParallel`, và chỉ rank 0 hiển thị progress/lưu
file. Một GPU hoặc CPU dùng single-process fallback.

Effective batch mặc định trên T4 x2:

```text
8 per-device × 2 GPU × 2 accumulation = 32
```

Việc triển khai DDP không đồng nghĩa đã xác minh runtime Kaggle. Hãy chạy
`--smoke-test` trên chính notebook T4 x2 và kiểm tra log có hai rank trước khi
full training.

## 13. Training, evaluation và best checkpoint

Chỉ Train DataLoader thực hiện backward. Sau mỗi epoch, Dev được gather theo
`original_index`, sắp lại và loại padding/duplicate do phân phối trước khi tính:

- raw MSE;
- MAE;
- exact accuracy;
- adjacent accuracy (sai lệch tối đa 1 mức);
- Quadratic Weighted Kappa (QWK) với label cố định `1..19`;
- accuracy riêng cho các head 3/5/7 mức.

Progress/log ghi riêng `total_loss`, `mse19`, `ce3`, `ce5`, `ce7`, average loss,
encoder LR và head LR. Auxiliary metric chỉ để chẩn đoán. Với mỗi seed trong
`ENSEMBLE_SEEDS`, script khởi tạo lại toàn bộ bốn head, thứ tự DataLoader,
dropout và weighted sampler; cả năm member vẫn chỉ backpropagate trên cùng Train.
Checkpoint được chọn riêng bằng Dev `down_qwk` của `score19` sau `np.floor`; nếu
QWK hòa, Dev `down_mae` thấp hơn thắng. Early stopping dùng cùng policy floor,
không nhìn Test và không dùng auxiliary accuracy. Round/up QWK vẫn được ghi để
chẩn đoán nhưng không tham gia lựa chọn.

Sau khi có năm best checkpoint, script chạy lại từng member trên Dev/Test, kiểm
tra ID và thứ tự hoàn toàn trùng nhau, rồi lấy trung bình đều của **raw regression
score** trước mọi phép rời rạc hóa. Sau khi lấy mean, script dùng `np.floor`,
clip vào `[1, 19]` và chuyển sang integer để tạo submission `down`. Báo cáo và
diagnostics vẫn chứa metric/prediction round/up để so sánh, nhưng chỉ bản down
là policy chính. Không threshold optimization, không QWK-weighting và không
chọn/bỏ seed theo Open Test.

Checkpoint resume của mỗi seed lưu model mode, encoder, toàn bộ auxiliary và
regression heads, optimizer, scheduler, scaler, epoch/global step,
`best_down_qwk/best_down_mae`, config, selection policy và RNG state. Để resume
ensemble, đặt:

```python
RESUME_FROM_CHECKPOINT = "outputs/seeds/seed_{seed}/checkpoints/last.pt"
```

Phải giữ best state tương ứng tại
`outputs/seeds/seed_<N>/best_model/model_state.pt`; script fail fast nếu cặp
artifact không đầy đủ. Model state được lưu sau khi unwrap DDP nên dùng được ở
một hoặc nhiều GPU. Checkpoint resume tạo trước policy floor không tương thích;
script yêu cầu bắt đầu lại seed đó từ epoch 1 thay vì dùng nhầm best model được
chọn theo round.

## 14. Smoke test và kiểm tra tối thiểu

Kiểm tra cú pháp:

```bash
python -m py_compile train.py train_blind.py
```

Smoke test thật (mẫu nhỏ, D3Tok/checkpoint thật, output riêng):

```bash
python train.py --smoke-test
```

Smoke mode dùng hai seed `42, 52` để kiểm tra cả phép ensemble mà không phải chạy
đủ năm member. Nó kiểm tra pipeline đọc dữ liệu, Unicode/Kashida, BERT D3Tok khi
dấu phụ vẫn còn, D3Tok/Surface view, feature block, sentence-pair tokenization,
sampler, forward/backward, metric, checkpoint/reload, inference, raw-score
averaging và submission down ZIP. Nó không phải một lần huấn luyện hợp lệ để
báo cáo QWK.

Các invariant cần đạt:

- không còn `U+0640` sau preprocessing;
- BERT D3Tok nhận câu còn dấu phụ;
- không có token chỉ chứa dấu phụ đi vào BERT D3Tok;
- không còn Arabic diacritics sau `dediac_ar`;
- dấu câu còn trong cả D3Tok và Surface;
- dấu `+` của D3Tok không bị xóa;
- D3Tok không bị truncate; feature chỉ bị bỏ theo nhóm atomic rồi mới tới Surface;
- output model giữ shape `logits3=[B,3]`, `logits5=[B,5]`, `logits7=[B,7]` và
  `score19=[B]`, kể cả batch size 1;
- loss dùng đúng hệ số `0.5`, finite, và MSE19 tạo gradient cho toàn bộ cascade;
- sampler chia đều hai rank và deterministic theo seed/epoch;
- gather trả đúng số mẫu theo thứ tự gốc;
- ensemble luôn trung bình raw score trước `np.floor`;
- `np.floor` giữ nguyên số nguyên và đưa mọi raw score còn phần thập phân xuống
  số nguyên thấp hơn; round/up chỉ là diagnostics;
- validator từ chối header/ID/range/ZIP sai của submission down.

## 15. Output và định dạng nộp bài

Sau full training:

```text
outputs/
├── seeds/
│   ├── seed_42/
│   │   ├── best_model/
│   │   ├── checkpoints/
│   │   │   └── last.pt
│   │   ├── diagnostics/
│   │   └── logs/
│   ├── seed_52/
│   ├── seed_62/
│   ├── seed_72/
│   └── seed_82/
├── logs/
│   ├── preprocessing_report.json
│   └── ensemble_report.json
├── diagnostics/
│   ├── ensemble_dev_predictions_with_raw_scores.csv
│   └── test_predictions_with_raw_scores.csv
├── prediction_down
└── prediction_down.zip
```

`train_blind.py` tạo cùng hai artifact tại:

```text
outputs/blind/prediction_down
outputs/blind/prediction_down.zip
```

File nộp gốc `prediction_down` là CSV UTF-8 không BOM và **không có phần mở
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

Mỗi ZIP, bất kể tên file bên ngoài, chứa trực tiếp đúng một entry tên
`prediction`, không có thư mục cha và không có file khác. Script mở lại từng ZIP
và xác minh filename, nội dung byte, header, số dòng, ID/order,
duplicate/missing ID và prediction range trước khi báo thành công. Không tạo
thêm `prediction`/`prediction.zip` theo tên cũ.

Diagnostics ghi `raw_prediction`, `Prediction_down`, `Prediction_round`,
`Prediction_up` và gold label nếu đang chạy Open Test; các trường diagnostics
không xuất hiện trong submission.

## 16. Troubleshooting

### Không tìm thấy BERT D3Tok MSA

```bash
camel_data -i disambig-bert-unfactored-msa
```

Sau đó chạy lại. Nếu môi trường không có Internet, tải CAMeL data/model vào một
phiên có Internet hoặc gắn cache hợp lệ; baseline không thay D3Tok bằng regex.

### CUDA out of memory

Giảm `PER_DEVICE_BATCH_SIZE` hoặc `MAX_LENGTH`, tăng
`GRADIENT_ACCUMULATION_STEPS`, rồi xóa checkpoint dở nếu không resume. Effective
batch có thể giữ nguyên bằng accumulation.

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

Khi script tự khởi chạy hai GPU, nó đặt mặc định
`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600`. Điều này cho phép rank 1 chờ rank 0
xây cache D3Tok lâu hơn ngưỡng watchdog mặc định mà không bị `SIGABRT`.

### `torch.cuda.is_available()` là `False`

Không cài lại `torch` bằng một wheel CPU sau khi khởi động Kaggle. Tạo session
mới hoặc cài đúng CUDA build theo hướng dẫn PyTorch.

### Sai cột hoặc Blind Test không có label

Đổi `ID_COLUMN`/`TEXT_COLUMN` trong `Config`. Blind Test không cần label; Train
và Dev bắt buộc có đủ label 3/5/7/19 đúng mapping phân cấp.

### Checkpoint không load được

Đảm bảo checkpoint, `MODEL_MODE` và model name tương thích; checkpoint
`baseline_mse` không load vào `cascaded_hmtl` hoặc ngược lại. Multi-seed resume
yêu cầu placeholder `{seed}`, ví dụ
`outputs/seeds/seed_{seed}/checkpoints/last.pt`, và best model tương ứng của từng
seed. Nếu không cần resume, đặt `RESUME_FROM_CHECKPOINT=None`.

### ZIP bị hệ thống chấm từ chối

Không tự đổi tên thành `prediction.csv`. Mở `prediction_down.zip`, rồi xác nhận
ZIP chỉ chứa `prediction` tại root với header `Sentence ID,Prediction`.

## 17. Reproducibility và báo cáo kết quả

Năm seed cố định được đặt cho Python, NumPy, PyTorch CPU/CUDA, DataLoader và
sampler. `ensemble_report.json` ghi Dev down QWK của từng member, mean/std của
metric này, metric ensemble cho down/round/up và xác nhận policy là uniform
raw-score mean rồi mới `floor`. Tuy vậy, khác biệt CUDA/library vẫn có thể gây
sai khác nhỏ. Hãy lưu config, commit hash, log và best metrics đi kèm mỗi run.

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
