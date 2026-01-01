"""
Script để gộp train_X, train_Y, test_X, test_Y thành X, Y 
sau đó áp dụng Stratified K-Fold Cross-Validation
"""
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


def merge_and_kfold(args):
    """
    Gộp train/test files và chia thành k folds.
    """
    print("=" * 80)
    print("MERGE TRAIN/TEST AND CREATE K-FOLD SPLITS")
    print("=" * 80 + "\n")

    # Bước 1: Load các file train/test
    print("Step 1: Loading train/test files...")
    train_X = pd.read_csv(args.train_x_path, dtype=np.float32)
    train_Y = pd.read_csv(args.train_y_path, dtype=np.float32)
    test_X = pd.read_csv(args.test_x_path, dtype=np.float32)
    test_Y = pd.read_csv(args.test_y_path, dtype=np.float32)

    print(f"  Train X: {train_X.shape} (samples × features)")
    print(f"  Train Y: {train_Y.shape} (samples × classes)")
    print(f"  Test X:  {test_X.shape} (samples × features)")
    print(f"  Test Y:  {test_Y.shape} (samples × classes)")

    # Kiểm tra số lượng features
    if train_X.shape[1] != test_X.shape[1]:
        raise ValueError(f"Number of features mismatch: train_X has {train_X.shape[1]}, test_X has {test_X.shape[1]}")
    
    if train_Y.shape[1] != test_Y.shape[1]:
        raise ValueError(f"Number of classes mismatch: train_Y has {train_Y.shape[1]}, test_Y has {test_Y.shape[1]}")

    # Bước 2: Gộp dữ liệu
    print("\nStep 2: Merging train and test data...")
    X = pd.concat([train_X, test_X], axis=0, ignore_index=True)
    Y = pd.concat([train_Y, test_Y], axis=0, ignore_index=True)

    print(f"  Merged X: {X.shape} (samples × features)")
    print(f"  Merged Y: {Y.shape} (samples × classes)")

    # Bước 3: Tạo labels cho stratification
    stratify_labels = Y.values.argmax(axis=1)
    classes = sorted(pd.unique(stratify_labels))
    
    print(f"\n  Classes: {classes}")
    print(f"  Class distribution: {pd.Series(stratify_labels).value_counts().sort_index().to_dict()}")

    # Bước 4: Tách features nếu có thông tin về số lượng omics
    # Nếu không có, sẽ đọc từ file hoặc dùng toàn bộ
    if args.num_gene and args.num_cpg and args.num_mirna:
        num_gene = args.num_gene
        num_cpg = args.num_cpg
        num_mirna = args.num_mirna
        print(f"\n  Features: gene={num_gene}, cpg={num_cpg}, mirna={num_mirna}")
    else:
        # Nếu không có thông tin, giả sử toàn bộ là features (không tách omics)
        num_gene = X.shape[1]
        num_cpg = 0
        num_mirna = 0
        print(f"\n  Total features: {num_gene} (no omics split specified)")

    # Tạo output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Lưu feature counts
    meta_path = os.path.join(args.output_dir, "feature_counts.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"num_gene={num_gene}\n")
        f.write(f"num_cpg={num_cpg}\n")
        f.write(f"num_mirna={num_mirna}\n")
        f.write(f"classes={classes}\n")

    # Bước 5: Tạo StratifiedKFold
    print(f"\nStep 3: Creating {args.k_folds} stratified folds...")
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    
    # Create folds directory
    folds_dir = os.path.join(args.output_dir, "folds")
    os.makedirs(folds_dir, exist_ok=True)

    # Split into k folds
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, stratify_labels), 1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        Y_train = Y.iloc[train_idx]
        Y_test = Y.iloc[test_idx]

        fold_dir = os.path.join(folds_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        # Save train/test splits for this fold
        train_x_path = os.path.join(fold_dir, "train_X.csv")
        test_x_path = os.path.join(fold_dir, "test_X.csv")
        train_y_path = os.path.join(fold_dir, "train_Y.csv")
        test_y_path = os.path.join(fold_dir, "test_Y.csv")

        X_train.to_csv(train_x_path, index=False)
        X_test.to_csv(test_x_path, index=False)
        Y_train.to_csv(train_y_path, index=False)
        Y_test.to_csv(test_y_path, index=False)

        # Print fold statistics
        train_labels = Y_train.values.argmax(axis=1)
        test_labels = Y_test.values.argmax(axis=1)
        
        print(f"Fold {fold_idx}:")
        print(f"  Train: {len(X_train)} samples - {pd.Series(train_labels).value_counts().sort_index().to_dict()}")
        print(f"  Test:  {len(X_test)} samples  - {pd.Series(test_labels).value_counts().sort_index().to_dict()}")

    print(f"\n✅ Saved {args.k_folds} folds to {folds_dir}/")
    print(f"✅ Feature counts saved to {meta_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge train/test files and create k-fold splits"
    )
    parser.add_argument("--train-x-path", required=True, help="Path to train_X.csv")
    parser.add_argument("--train-y-path", required=True, help="Path to train_Y.csv")
    parser.add_argument("--test-x-path", required=True, help="Path to test_X.csv")
    parser.add_argument("--test-y-path", required=True, help="Path to test_Y.csv")
    parser.add_argument("--output-dir", default="./kfold_output_merged", help="Directory to write fold splits")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--num-gene", type=int, help="Number of gene features (for feature_counts.txt)")
    parser.add_argument("--num-cpg", type=int, help="Number of CpG features (for feature_counts.txt)")
    parser.add_argument("--num-mirna", type=int, help="Number of miRNA features (for feature_counts.txt)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_and_kfold(args)

