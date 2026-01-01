# Hướng dẫn gộp train/test files và tạo K-Fold

## Mục đích

Gộp 4 file `train_X.csv`, `train_Y.csv`, `test_X.csv`, `test_Y.csv` thành dữ liệu tổng hợp, sau đó chia thành k folds với Stratified K-Fold.

## Cách sử dụng

### Bước 1: Xác định số lượng features (nếu có)

Nếu bạn biết số lượng features của từng omics (gene, cpg, mirna), bạn cần thông tin này. Nếu không biết, có thể:
- Xem trong file `feature_counts.txt` (nếu có)
- Hoặc không chỉ định (sẽ để 0 và dùng toàn bộ features)

### Bước 2: Chạy script

```powershell
python merge_and_kfold.py `
    --train-x-path train_X.csv `
    --train-y-path train_Y.csv `
    --test-x-path test_X.csv `
    --test-y-path test_Y.csv `
    --output-dir ./kfold_output_merged `
    --k-folds 5 `
    --num-gene 500 --num-cpg 500 --num-mirna 100 `
    --seed 42
```

**Tham số:**
- `--train-x-path`: Đường dẫn đến file train_X.csv
- `--train-y-path`: Đường dẫn đến file train_Y.csv
- `--test-x-path`: Đường dẫn đến file test_X.csv
- `--test-y-path`: Đường dẫn đến file test_Y.csv
- `--output-dir`: Thư mục lưu kết quả (mặc định: ./kfold_output_merged)
- `--k-folds`: Số lượng folds (mặc định: 5)
- `--seed`: Random seed (mặc định: 42)
- `--num-gene`, `--num-cpg`, `--num-mirna`: Số lượng features (optional, để tạo feature_counts.txt)

### Ví dụ đầy đủ

Nếu các file ở thư mục `deep-learning-omics/`:

```powershell
cd D:\semester2025.1\deep_learning\project\source_code_paper2023\deep-learning-omics

python merge_and_kfold.py `
    --train-x-path train_X.csv `
    --train-y-path train_Y.csv `
    --test-x-path test_X.csv `
    --test-y-path test_Y.csv `
    --output-dir ./kfold_output_merged `
    --k-folds 5 `
    --num-gene 500 --num-cpg 500 --num-mirna 100 `
    --seed 42
```

Nếu các file ở thư mục khác:

```powershell
python merge_and_kfold.py `
    --train-x-path ../train-test/train_X.csv `
    --train-y-path ../train-test/train_Y.csv `
    --test-x-path ../train-test/test_X.csv `
    --test-y-path ../train-test/test_Y.csv `
    --output-dir ./kfold_output_merged `
    --k-folds 5 `
    --num-gene 500 --num-cpg 500 --num-mirna 100
```

## Quy trình

1. **Load 4 files**: train_X, train_Y, test_X, test_Y
2. **Gộp lại**:
   - X = concat(train_X, test_X)
   - Y = concat(train_Y, test_Y)
3. **Tạo StratifiedKFold** với số folds chỉ định
4. **Chia thành k folds**, mỗi fold có train_X, train_Y, test_X, test_Y
5. **Lưu kết quả** vào thư mục output

## Kết quả

Sau khi chạy, bạn sẽ có:

```
kfold_output_merged/
├── feature_counts.txt
└── folds/
    ├── fold_1/
    │   ├── train_X.csv
    │   ├── train_Y.csv
    │   ├── test_X.csv
    │   └── test_Y.csv
    ├── fold_2/
    └── ...
```

## Lưu ý

- Script sẽ kiểm tra số lượng features và classes phải khớp giữa train và test
- Nếu không chỉ định `--num-gene`, `--num-cpg`, `--num-mirna`, file `feature_counts.txt` sẽ có num_gene = tổng số features, num_cpg=0, num_mirna=0
- Sau khi tạo folds, bạn có thể dùng `run_kfold.py` để train như bình thường

## Ví dụ output

```
================================================================================
MERGE TRAIN/TEST AND CREATE K-FOLD SPLITS
================================================================================

Step 1: Loading train/test files...
  Train X: (538, 1100) (samples × features)
  Train Y: (538, 5) (samples × classes)
  Test X:  (134, 1100) (samples × features)
  Test Y:  (134, 5) (samples × classes)

Step 2: Merging train and test data...
  Merged X: (672, 1100) (samples × features)
  Merged Y: (672, 5) (samples × classes)

  Classes: [0, 1, 2, 3, 4]
  Class distribution: {0: 134, 1: 135, 2: 134, 3: 134, 4: 135}

  Features: gene=500, cpg=500, mirna=100

Step 3: Creating 5 stratified folds...
Fold 1:
  Train: 538 samples - {0: 107, 1: 108, 2: 107, 3: 107, 4: 109}
  Test:  134 samples  - {0: 27, 1: 27, 2: 27, 3: 27, 4: 26}
...

✅ Saved 5 folds to ./kfold_output_merged/folds/
✅ Feature counts saved to ./kfold_output_merged/feature_counts.txt
```

