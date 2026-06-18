# =============================================================================
# PIPELINE TERPADU: PREDIKSI LUNG CANCER
# Menggabungkan 4 tugas menjadi satu alur kerja yang berkesinambungan:
#   BAGIAN 1 - Data Preprocessing & Penanganan Missing Value
#   BAGIAN 2 - Penanganan Class Imbalance (Baseline vs SMOTE vs Undersampling)
#   BAGIAN 3 - Feature Engineering (Scaling, PCA, RFE, Mutual Information)
#   BAGIAN 4 - Modeling (Logistic Regression, SVM, XGBoost) & Interpretasi (LIME)
#
# Mata Kuliah : Big Data dalam Dunia Kesehatan
# Dataset     : survey lung cancer (1).csv (Kaggle)
# Bahasa      : R (RStudio)
#
# CATATAN PENGGABUNGAN:
#  - Seluruh tahap kini memakai SATU sumber data (FILE_PATH di bawah) yang
#    dibaca SATU KALI di awal. Tidak ada lagi load ulang / data sintetis
#    seperti pada skrip asli Bagian 3 (yang sebelumnya ditulis di Python).
#  - Penamaan variabel dirapikan & dikonsistenkan (df_raw, df_clean,
#    train_raw, train_balanced, X_train_final, dst).
#  - Encoding GENDER (M=1, F=0) dan LUNG_CANCER (YES=1, NO=0) disamakan di
#    seluruh pipeline (skrip asli Bagian 2 sempat memakai encoding berbeda).
#  - Implementasi SMOTE manual yang rawan bug pada skrip asli Bagian 2 diganti
#    dengan package smotefamily (sama seperti yang sudah dipakai di skrip
#    asli Bagian 4), dipakai konsisten di Bagian 2 dan Bagian 4.
#  - Bagian Feature Engineering (semula Python/sklearn) diterjemahkan ke R
#    agar seluruh pipeline berjalan dalam satu bahasa & satu file:
#      RFE (sklearn) -> caret::rfe dengan random forest
#      Mutual Information (sklearn) -> infotheo::mutinformation
#  - Bagian interpretasi LIME pada skrip asli (yang sempat dobel/duplikat)
#    dirapikan menjadi satu blok yang bersih.
# =============================================================================

# -----------------------------------------------------------------------------
# 0. KONFIGURASI GLOBAL & LIBRARY
# -----------------------------------------------------------------------------
FILE_PATH <- "D:/Big Data/survey lung cancer (1).csv"   # sesuaikan jika perlu
SEED      <- 42
set.seed(SEED)

pkg_list <- c(
  "readr", "dplyr", "tidyr", "ggplot2", "gridExtra", "knitr",   # data wrangling & viz
  "VIM", "Metrics",                                             # imputasi & evaluasi MAE
  "caret", "smotefamily", "randomForest",                       # split, imbalance, RFE
  "infotheo",                                                   # mutual information
  "e1071", "xgboost", "pROC", "lime"                            # modeling & interpretasi
)
pkg_baru <- pkg_list[!(pkg_list %in% installed.packages()[, "Package"])]
if (length(pkg_baru) > 0) install.packages(pkg_baru, dependencies = TRUE)
invisible(lapply(pkg_list, library, character.only = TRUE))

cat(strrep("=", 78), "\n", sep = "")
cat("PIPELINE LUNG CANCER: PREPROCESSING -> IMBALANCE -> FEATURE ENG -> MODEL\n")
cat(strrep("=", 78), "\n", sep = "")

# Fungsi evaluasi yang dipakai berulang kali di Bagian 2, 3, dan 4
fungsi_eval <- function(prob, aktual_label, ambang = 0.5) {
  pred_label <- factor(ifelse(prob > ambang, "YES", "NO"), levels = c("NO", "YES"))
  
  # Gunakan caret:: explicitly (opsional, tapi lebih aman)
  cm <- caret::confusionMatrix(pred_label, aktual_label, positive = "YES")
  
  # WAJIB: Gunakan pROC:: explicitly agar tidak dibajak oleh package Metrics
  auc_val <- as.numeric(pROC::auc(pROC::roc(aktual_label, prob, levels = c("NO", "YES"),
                                            direction = "<", quiet = TRUE)))
  
  list(cm = cm, accuracy = unname(cm$overall["Accuracy"]),
       f1 = unname(cm$byClass["F1"]), auc = auc_val)
}

# =============================================================================
# BAGIAN 1 (Tugas 2): DATA PREPROCESSING & PENANGANAN MISSING VALUE
# =============================================================================
cat("\n", strrep("=", 78), "\n", sep = "")
cat("BAGIAN 1: PREPROCESSING & PENANGANAN MISSING VALUE\n")
cat(strrep("=", 78), "\n", sep = "")

## 1.1 Load dataset ------------------------------------------------------------
df_raw <- read_csv(FILE_PATH, show_col_types = FALSE)

# Rapikan nama kolom: hilangkan spasi di ujung & ganti spasi tengah dengan "_"
names(df_raw) <- trimws(names(df_raw))
names(df_raw) <- gsub(" ", "_", names(df_raw))

cat(sprintf("Dataset dimuat: %d baris x %d kolom\n", nrow(df_raw), ncol(df_raw)))
cat("Kolom:", paste(names(df_raw), collapse = ", "), "\n")

## 1.3 Cek missing value pada data asli ------------------------------------------
jumlah_na <- colSums(is.na(df_raw))
tabel_mv <- data.frame(
  Kolom          = names(jumlah_na),
  Jumlah_Missing = as.integer(jumlah_na),
  Persen_Missing = round(jumlah_na / nrow(df_raw) * 100, 2)
)
cat("\nTabel missing value (data asli):\n")
print(kable(tabel_mv))
cat(sprintf("Total missing value: %d\n", sum(jumlah_na)))

## 1.4 Encoding kolom kategorikal -------------------------------------------------
df_encoded <- df_raw %>%
  mutate(
    GENDER      = ifelse(GENDER == "M", 1, 0),       # M = 1, F = 0
    LUNG_CANCER = ifelse(LUNG_CANCER == "YES", 1, 0)  # YES = 1, NO = 0
  )

fitur_kolom <- setdiff(names(df_encoded), "LUNG_CANCER")
df_fitur    <- as.data.frame(df_encoded[, fitur_kolom])
target_asli <- df_encoded$LUNG_CANCER

cat(sprintf("\nJumlah fitur prediktor : %d\n", length(fitur_kolom)))
cat(sprintf("Distribusi target asli : NO(0)=%d | YES(1)=%d\n",
            sum(target_asli == 0), sum(target_asli == 1)))

## 1.5 Simulasi missing value 10% pada fitur prediktor -----------------------------
# Dataset Kaggle ini secara default tidak memiliki missing value, sehingga untuk
# keperluan latihan penanganan missing value, 10% sel pada fitur prediktor
# (bukan target) disimulasikan hilang secara acak.
set.seed(SEED)
df_fitur_asli <- df_fitur
df_fitur_sim  <- df_fitur

n_baris   <- nrow(df_fitur_sim)
n_kolom   <- ncol(df_fitur_sim)
n_missing <- round(0.10 * n_baris * n_kolom)

posisi_missing <- data.frame(
  baris = sample.int(n_baris, n_missing, replace = TRUE),
  kolom = sample.int(n_kolom, n_missing, replace = TRUE)
)
posisi_missing <- posisi_missing[!duplicated(posisi_missing), ]

nilai_asli_vec <- mapply(function(r, k) df_fitur_asli[[k]][r],
                          posisi_missing$baris, posisi_missing$kolom)

for (i in seq_len(nrow(posisi_missing))) {
  df_fitur_sim[posisi_missing$baris[i], posisi_missing$kolom[i]] <- NA
}

cat(sprintf("\nTotal sel yang disimulasikan hilang: %d (%.2f%% dari %d sel)\n",
            nrow(posisi_missing), 100 * nrow(posisi_missing) / (n_baris * n_kolom),
            n_baris * n_kolom))

## 1.6 Tiga metode imputasi: Mean, Median, KNN --------------------------------------

# -- Mean --
df_mean <- df_fitur_sim
for (k in names(df_mean)) {
  if (anyNA(df_mean[[k]])) df_mean[[k]][is.na(df_mean[[k]])] <- mean(df_mean[[k]], na.rm = TRUE)
}

# -- Median --
df_median <- df_fitur_sim
for (k in names(df_median)) {
  if (anyNA(df_median[[k]])) df_median[[k]][is.na(df_median[[k]])] <- median(df_median[[k]], na.rm = TRUE)
}

# -- KNN (normalisasi min-max dahulu, lalu VIM::kNN dengan k = 5) --
min_val <- sapply(df_fitur_sim, min, na.rm = TRUE)
max_val <- sapply(df_fitur_sim, max, na.rm = TRUE)
rentang <- max_val - min_val
rentang[rentang == 0] <- 1   # jaga-jaga jika ada kolom konstan

df_scaled     <- as.data.frame(scale(df_fitur_sim, center = min_val, scale = rentang))
df_knn_scaled <- kNN(df_scaled, k = 5, imp_var = FALSE)
df_knn <- as.data.frame(sweep(sweep(as.matrix(df_knn_scaled), 2, rentang, "*"), 2, min_val, "+"))
names(df_knn) <- names(df_fitur_sim)

cat("Imputasi Mean, Median, dan KNN selesai dijalankan.\n")

## 1.7 Evaluasi MAE & pilih metode terbaik -------------------------------------------
ambil_nilai <- function(df_imputed, posisi) {
  mapply(function(r, k) df_imputed[[k]][r], posisi$baris, posisi$kolom)
}

nilai_mean   <- ambil_nilai(df_mean,   posisi_missing)
nilai_median <- ambil_nilai(df_median, posisi_missing)
nilai_knn    <- ambil_nilai(df_knn,    posisi_missing)

mae_per_kolom <- data.frame(
  Kolom = fitur_kolom, MAE_Mean = NA_real_, MAE_Median = NA_real_, MAE_KNN = NA_real_
)
for (j in seq_along(fitur_kolom)) {
  idx <- which(posisi_missing$kolom == j)
  if (length(idx) > 0) {
    asli <- nilai_asli_vec[idx]
    mae_per_kolom$MAE_Mean[j]   <- mae(asli, nilai_mean[idx])
    mae_per_kolom$MAE_Median[j] <- mae(asli, nilai_median[idx])
    mae_per_kolom$MAE_KNN[j]    <- mae(asli, nilai_knn[idx])
  }
}
cat("\nMAE per kolom:\n")
print(kable(mae_per_kolom, digits = 4))

rata_mae <- data.frame(
  Metode = c("Mean", "Median", "KNN"),
  Rata_rata_MAE = round(c(mean(mae_per_kolom$MAE_Mean,   na.rm = TRUE),
                           mean(mae_per_kolom$MAE_Median, na.rm = TRUE),
                           mean(mae_per_kolom$MAE_KNN,    na.rm = TRUE)), 4)
)
cat("\nRata-rata MAE tiap metode:\n")
print(kable(rata_mae))

metode_terbaik <- rata_mae$Metode[which.min(rata_mae$Rata_rata_MAE)]
cat(sprintf("\nMetode imputasi terbaik (MAE terkecil): %s\n", metode_terbaik))

df_fitur_terbaik <- switch(metode_terbaik, "Mean" = df_mean, "Median" = df_median, "KNN" = df_knn)

## 1.8 Visualisasi distribusi AGE ------------------------------------------------------
df_plot_age <- data.frame(
  Asli = df_fitur_asli$AGE, Mean = df_mean$AGE, Median = df_median$AGE, KNN = df_knn$AGE
) %>%
  pivot_longer(everything(), names_to = "Metode", values_to = "Nilai") %>%
  mutate(Metode = factor(Metode, levels = c("Asli", "Mean", "Median", "KNN")))

p_density <- ggplot(df_plot_age, aes(Nilai, fill = Metode, color = Metode)) +
  geom_density(alpha = 0.3, linewidth = 0.8) +
  labs(title = "Distribusi AGE: Data Asli vs Hasil Imputasi", x = "Usia", y = "Densitas") +
  theme_minimal(base_size = 12) + theme(legend.position = "bottom")

p_box <- ggplot(df_plot_age, aes(Metode, Nilai, fill = Metode)) +
  geom_boxplot(alpha = 0.7, outlier.color = "red") +
  labs(title = "Boxplot AGE per Metode Imputasi", x = NULL, y = "Usia") +
  theme_minimal(base_size = 12) + theme(legend.position = "none")

p_mae <- ggplot(rata_mae, aes(Metode, Rata_rata_MAE, fill = Metode)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = Rata_rata_MAE), vjust = -0.5, fontface = "bold") +
  labs(title = "Rata-rata MAE Tiap Metode Imputasi", x = NULL, y = "MAE") +
  theme_minimal(base_size = 12) + theme(legend.position = "none")

grid.arrange(p_density, p_box, p_mae, ncol = 2, top = "Evaluasi Metode Imputasi - Lung Cancer Dataset")

## 1.9 Bentuk dataset bersih final (output Bagian 1) -------------------------------------
df_clean <- df_fitur_terbaik
df_clean$LUNG_CANCER       <- target_asli
df_clean$LUNG_CANCER_LABEL <- factor(ifelse(target_asli == 1, "YES", "NO"), levels = c("NO", "YES"))

cat(sprintf("\n[BAGIAN 1 SELESAI] df_clean: %d baris x %d kolom (metode imputasi: %s)\n",
            nrow(df_clean), ncol(df_clean), metode_terbaik))


# =============================================================================
# BAGIAN 2 (Tugas 3): PENANGANAN CLASS IMBALANCE
# =============================================================================
cat("\n", strrep("=", 78), "\n", sep = "")
cat("BAGIAN 2: CLASS IMBALANCE - BASELINE vs SMOTE vs UNDERSAMPLING\n")
cat(strrep("=", 78), "\n", sep = "")

## 2.1 Analisis distribusi kelas pada df_clean -------------------------------------
dist_kelas   <- table(df_clean$LUNG_CANCER_LABEL)
persen_kelas <- round(prop.table(dist_kelas) * 100, 1)

cat("\nDistribusi kelas (seluruh dataset bersih):\n")
cat(sprintf("YES: %d sampel (%.1f%%)\n", dist_kelas["YES"], persen_kelas["YES"]))
cat(sprintf("NO : %d sampel (%.1f%%)\n", dist_kelas["NO"],  persen_kelas["NO"]))
cat(sprintf("Rasio imbalance (YES:NO) = %.2f : 1\n", dist_kelas["YES"] / dist_kelas["NO"]))

## 2.2 Split train-test (80:20, stratified) -----------------------------------------
set.seed(SEED)
idx_train <- createDataPartition(df_clean$LUNG_CANCER_LABEL, p = 0.80, list = FALSE)
train_raw <- df_clean[idx_train, ]
test_raw  <- df_clean[-idx_train, ]

cat(sprintf("\nTraining: %d sampel | Testing: %d sampel\n", nrow(train_raw), nrow(test_raw)))
cat("Distribusi training:\n"); print(table(train_raw$LUNG_CANCER_LABEL))
cat("Distribusi testing :\n"); print(table(test_raw$LUNG_CANCER_LABEL))

glm_data <- function(df) df[, c(fitur_kolom, "LUNG_CANCER")]

## 2.3 Baseline model (tanpa balancing) -----------------------------------------------
model_baseline <- glm(LUNG_CANCER ~ ., data = glm_data(train_raw), family = binomial)
prob_baseline  <- predict(model_baseline, newdata = test_raw, type = "response")
eval_baseline  <- fungsi_eval(prob_baseline, test_raw$LUNG_CANCER_LABEL)
cat(sprintf("\n[Baseline] Accuracy=%.4f | F1=%.4f | ROC-AUC=%.4f\n",
            eval_baseline$accuracy, eval_baseline$f1, eval_baseline$auc))

## 2.4 SMOTE (package smotefamily, k = 5) -----------------------------------------------
set.seed(SEED)
X_train_imb <- train_raw[, fitur_kolom]
y_train_imb <- as.character(train_raw$LUNG_CANCER_LABEL)

smote_out   <- SMOTE(X_train_imb, y_train_imb, K = 5)
train_smote <- smote_out$data
names(train_smote)[names(train_smote) == "class"] <- "LUNG_CANCER_LABEL"
train_smote$LUNG_CANCER_LABEL <- factor(train_smote$LUNG_CANCER_LABEL, levels = c("NO", "YES"))
train_smote$LUNG_CANCER       <- ifelse(train_smote$LUNG_CANCER_LABEL == "YES", 1, 0)
train_smote <- train_smote[, c(fitur_kolom, "LUNG_CANCER", "LUNG_CANCER_LABEL")]

cat(sprintf("\n[SMOTE] Training: %d sampel\n", nrow(train_smote)))
print(table(train_smote$LUNG_CANCER_LABEL))

model_smote <- glm(LUNG_CANCER ~ ., data = glm_data(train_smote), family = binomial)
prob_smote  <- predict(model_smote, newdata = test_raw, type = "response")
eval_smote  <- fungsi_eval(prob_smote, test_raw$LUNG_CANCER_LABEL)
cat(sprintf("[SMOTE] Accuracy=%.4f | F1=%.4f | ROC-AUC=%.4f\n",
            eval_smote$accuracy, eval_smote$f1, eval_smote$auc))

## 2.5 Random Undersampling --------------------------------------------------------------
set.seed(SEED)
minoritas       <- train_raw[train_raw$LUNG_CANCER_LABEL == "NO", ]
mayoritas       <- train_raw[train_raw$LUNG_CANCER_LABEL == "YES", ]
mayoritas_under <- mayoritas[sample(nrow(mayoritas), nrow(minoritas)), ]
train_under     <- rbind(minoritas, mayoritas_under)

cat(sprintf("\n[Undersampling] Training: %d sampel\n", nrow(train_under)))
print(table(train_under$LUNG_CANCER_LABEL))

model_under <- glm(LUNG_CANCER ~ ., data = glm_data(train_under), family = binomial)
prob_under  <- predict(model_under, newdata = test_raw, type = "response")
eval_under  <- fungsi_eval(prob_under, test_raw$LUNG_CANCER_LABEL)
cat(sprintf("[Undersampling] Accuracy=%.4f | F1=%.4f | ROC-AUC=%.4f\n",
            eval_under$accuracy, eval_under$f1, eval_under$auc))

## 2.6 Tabel perbandingan & visualisasi --------------------------------------------------
tabel_imbalance <- data.frame(
  Metode   = c("Baseline", "SMOTE", "Undersampling"),
  N_Train  = c(nrow(train_raw), nrow(train_smote), nrow(train_under)),
  Accuracy = round(c(eval_baseline$accuracy, eval_smote$accuracy, eval_under$accuracy), 4),
  F1_Score = round(c(eval_baseline$f1, eval_smote$f1, eval_under$f1), 4),
  ROC_AUC  = round(c(eval_baseline$auc, eval_smote$auc, eval_under$auc), 4)
)
cat("\nTabel perbandingan metode penanganan imbalance:\n")
print(kable(tabel_imbalance))

p_imbalance <- tabel_imbalance %>%
  pivot_longer(c(Accuracy, F1_Score, ROC_AUC), names_to = "Metrik", values_to = "Skor") %>%
  ggplot(aes(Metode, Skor, fill = Metrik)) +
  geom_col(position = "dodge") +
  geom_text(aes(label = round(Skor, 3)), position = position_dodge(width = 0.9), vjust = -0.4, size = 3) +
  ylim(0, 1.1) +
  labs(title = "Perbandingan Performa: Baseline vs SMOTE vs Undersampling") +
  theme_minimal(base_size = 12)
print(p_imbalance)

## 2.7 Pilih strategi balancing terbaik (berdasarkan F1-Score) ----------------------------
metode_imbalance_terbaik <- tabel_imbalance$Metode[which.max(tabel_imbalance$F1_Score)]
cat(sprintf("\nStrategi imbalance terbaik (F1 tertinggi): %s\n", metode_imbalance_terbaik))

train_balanced <- switch(metode_imbalance_terbaik,
                          "Baseline" = train_raw, "SMOTE" = train_smote, "Undersampling" = train_under)

cat(sprintf("\n[BAGIAN 2 SELESAI] train_balanced: %d sampel (strategi: %s) | test_raw: %d sampel\n",
            nrow(train_balanced), metode_imbalance_terbaik, nrow(test_raw)))


# =============================================================================
# BAGIAN 3 (Tugas 4): FEATURE ENGINEERING
# =============================================================================
cat("\n", strrep("=", 78), "\n", sep = "")
cat("BAGIAN 3: FEATURE ENGINEERING (SCALING, PCA, RFE, MUTUAL INFORMATION)\n")
cat(strrep("=", 78), "\n", sep = "")

## 3.1 Normalisasi Min-Max (fit di train_balanced, terapkan ke test_raw) -------------------
pre_proc       <- preProcess(train_balanced[, fitur_kolom], method = "range")
X_train_scaled <- predict(pre_proc, train_balanced[, fitur_kolom])
X_test_scaled  <- predict(pre_proc, test_raw[, fitur_kolom])

y_train_label <- train_balanced$LUNG_CANCER_LABEL
y_train_num   <- train_balanced$LUNG_CANCER
y_test_label  <- test_raw$LUNG_CANCER_LABEL
y_test_num    <- test_raw$LUNG_CANCER

cat(sprintf("Scaling selesai. Rentang fitur train: [%.2f, %.2f]\n",
            min(sapply(X_train_scaled, min)), max(sapply(X_train_scaled, max))))

## 3.2 PCA -------------------------------------------------------------------------------
pca_model     <- prcomp(X_train_scaled, center = TRUE, scale. = FALSE)
var_explained <- pca_model$sdev^2 / sum(pca_model$sdev^2)
var_kumulatif <- cumsum(var_explained)
n_pc_terbaik  <- which(var_kumulatif >= 0.95)[1]

cat(sprintf("\nJumlah PC untuk mencapai >=95%% variance: %d dari %d komponen\n",
            n_pc_terbaik, length(var_explained)))

X_train_pca <- as.data.frame(predict(pca_model, X_train_scaled)[, 1:n_pc_terbaik, drop = FALSE])
X_test_pca  <- as.data.frame(predict(pca_model, X_test_scaled)[, 1:n_pc_terbaik, drop = FALSE])

df_scree <- data.frame(PC = seq_along(var_explained), Variance = var_explained, Kumulatif = var_kumulatif)

p_scree <- ggplot(df_scree, aes(PC, Variance * 100)) +
  geom_col(fill = "steelblue") +
  geom_vline(xintercept = n_pc_terbaik, color = "red", linetype = "dashed") +
  labs(title = "Scree Plot PCA", x = "Komponen", y = "Explained Variance (%)") +
  theme_minimal(base_size = 12)

p_kumulatif <- ggplot(df_scree, aes(PC, Kumulatif * 100)) +
  geom_line(color = "darkorange", linewidth = 1) + geom_point() +
  geom_hline(yintercept = 95, color = "red", linetype = "dashed") +
  geom_vline(xintercept = n_pc_terbaik, color = "green", linetype = "dashed") +
  labs(title = "Cumulative Explained Variance", x = "Jumlah Komponen", y = "Kumulatif (%)") +
  theme_minimal(base_size = 12)

grid.arrange(p_scree, p_kumulatif, ncol = 2, top = "PCA - Analisis Explained Variance")

## 3.3 RFE dengan Random Forest (caret::rfe) -----------------------------------------------
set.seed(SEED)
ctrl_rfe   <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
ukuran_rfe <- unique(pmin(c(3, 5, 8, length(fitur_kolom)), length(fitur_kolom)))

rfe_result <- rfe(x = X_train_scaled, y = y_train_label, sizes = ukuran_rfe, rfeControl = ctrl_rfe)
fitur_rfe  <- predictors(rfe_result)

cat(sprintf("\nJumlah fitur terpilih RFE: %d dari %d\n", length(fitur_rfe), length(fitur_kolom)))
cat("Fitur terpilih RFE:", paste(fitur_rfe, collapse = ", "), "\n")

## 3.4 Mutual Information (package infotheo) -----------------------------------------------
X_train_disc <- discretize(X_train_scaled, disc = "equalfreq", nbins = 5)
y_train_int  <- as.integer(y_train_label)

mi_scores <- sapply(names(X_train_disc), function(k) mutinformation(X_train_disc[[k]], y_train_int))
mi_df <- data.frame(Fitur = names(mi_scores), MI_Score = as.numeric(mi_scores)) %>% arrange(desc(MI_Score))

cat("\nSkor Mutual Information (urut menurun):\n")
print(kable(mi_df, digits = 4))

ambang_mi <- median(mi_df$MI_Score)
fitur_mi  <- mi_df$Fitur[mi_df$MI_Score >= ambang_mi]
cat(sprintf("\nThreshold MI (median) = %.4f\n", ambang_mi))
cat("Fitur terpilih MI:", paste(fitur_mi, collapse = ", "), "\n")

p_mi <- ggplot(mi_df, aes(reorder(Fitur, MI_Score), MI_Score, fill = MI_Score >= ambang_mi)) +
  geom_col() + coord_flip() +
  geom_hline(yintercept = ambang_mi, color = "red", linetype = "dashed") +
  scale_fill_manual(values = c("TRUE" = "#2ecc71", "FALSE" = "#bdc3c7"), guide = "none") +
  labs(title = "Mutual Information per Fitur", x = NULL, y = "MI Score") +
  theme_minimal(base_size = 12)
print(p_mi)

## 3.5 Bandingkan performa Logistic Regression antar set fitur ------------------------------
evaluasi_lr <- function(X_tr, y_tr_label, X_te, y_te_label) {
  data_tr <- X_tr
  data_tr$LUNG_CANCER <- ifelse(y_tr_label == "YES", 1, 0)
  model <- glm(LUNG_CANCER ~ ., data = data_tr, family = binomial)
  prob  <- predict(model, newdata = X_te, type = "response")
  fungsi_eval(prob, y_te_label)
}

eval_semua <- evaluasi_lr(X_train_scaled, y_train_label, X_test_scaled, y_test_label)
eval_rfe   <- evaluasi_lr(X_train_scaled[, fitur_rfe, drop = FALSE], y_train_label,
                           X_test_scaled[, fitur_rfe, drop = FALSE], y_test_label)
eval_mi    <- evaluasi_lr(X_train_scaled[, fitur_mi, drop = FALSE], y_train_label,
                           X_test_scaled[, fitur_mi, drop = FALSE], y_test_label)
eval_pca   <- evaluasi_lr(X_train_pca, y_train_label, X_test_pca, y_test_label)

tabel_fe <- data.frame(
  Metode   = c("Semua Fitur", "RFE", "Mutual Information", "PCA"),
  N_Fitur  = c(length(fitur_kolom), length(fitur_rfe), length(fitur_mi), n_pc_terbaik),
  Accuracy = round(c(eval_semua$accuracy, eval_rfe$accuracy, eval_mi$accuracy, eval_pca$accuracy), 4),
  F1_Score = round(c(eval_semua$f1, eval_rfe$f1, eval_mi$f1, eval_pca$f1), 4),
  ROC_AUC  = round(c(eval_semua$auc, eval_rfe$auc, eval_mi$auc, eval_pca$auc), 4)
)
cat("\nTabel perbandingan performa Logistic Regression antar metode feature selection:\n")
print(kable(tabel_fe))

p_fe <- tabel_fe %>%
  pivot_longer(c(Accuracy, F1_Score, ROC_AUC), names_to = "Metrik", values_to = "Skor") %>%
  ggplot(aes(Metode, Skor, fill = Metrik)) +
  geom_col(position = "dodge") +
  geom_text(aes(label = round(Skor, 3)), position = position_dodge(width = 0.9), vjust = -0.4, size = 3) +
  ylim(0, 1.1) +
  labs(title = "Perbandingan Performa: Sebelum vs Sesudah Feature Selection") +
  theme_minimal(base_size = 12)
print(p_fe)

## 3.6 Tentukan fitur final yang dipakai di Bagian 4 (berdasarkan F1 tertinggi) ---------------
metode_fe_terbaik <- tabel_fe$Metode[which.max(tabel_fe$F1_Score)]
cat(sprintf("\nMetode feature selection terbaik (F1 tertinggi): %s\n", metode_fe_terbaik))

if (metode_fe_terbaik == "PCA") {
  fitur_final   <- colnames(X_train_pca)
  X_train_final <- X_train_pca
  X_test_final  <- X_test_pca
  cat("Catatan: fitur final berupa komponen utama (PC), bukan gejala asli.\n")
} else {
  fitur_final <- switch(metode_fe_terbaik,
                         "Semua Fitur" = fitur_kolom, "RFE" = fitur_rfe, "Mutual Information" = fitur_mi)
  X_train_final <- X_train_scaled[, fitur_final, drop = FALSE]
  X_test_final  <- X_test_scaled[, fitur_final, drop = FALSE]
}

cat(sprintf("\n[BAGIAN 3 SELESAI] Fitur final dipakai untuk modeling: %s (%d fitur)\n",
            metode_fe_terbaik, length(fitur_final)))


# =============================================================================
# BAGIAN 4 (Tugas 5-6): MODELING (LR, SVM, XGBoost) & INTERPRETASI (LIME)
# =============================================================================
cat("\n", strrep("=", 78), "\n", sep = "")
cat("BAGIAN 4: MODELING & INTERPRETASI\n")
cat(strrep("=", 78), "\n", sep = "")

## 4.1 Siapkan data final untuk training ----------------------------------------------------
train_model_df <- X_train_final
train_model_df$LUNG_CANCER       <- y_train_num
train_model_df$LUNG_CANCER_LABEL <- y_train_label

test_model_df <- X_test_final
test_model_df$LUNG_CANCER       <- y_test_num
test_model_df$LUNG_CANCER_LABEL <- y_test_label

cat(sprintf("Data training final: %d sampel x %d fitur (metode: %s)\n",
            nrow(train_model_df), length(fitur_final), metode_fe_terbaik))

## 4.2 A. Logistic Regression -----------------------------------------------------------------
cat("\n--- A. Logistic Regression ---\n")
set.seed(SEED)
model_lr <- glm(LUNG_CANCER ~ ., data = train_model_df[, c(fitur_final, "LUNG_CANCER")], family = binomial)
prob_lr  <- predict(model_lr, newdata = test_model_df, type = "response")
pred_lr  <- factor(ifelse(prob_lr > 0.5, "YES", "NO"), levels = c("NO", "YES"))
cm_lr    <- confusionMatrix(pred_lr, test_model_df$LUNG_CANCER_LABEL, positive = "YES")
print(cm_lr)

## 4.3 B. Support Vector Machine ----------------------------------------------------------------
cat("\n--- B. Support Vector Machine (kernel radial) ---\n")
set.seed(SEED)
model_svm <- svm(LUNG_CANCER_LABEL ~ ., data = train_model_df[, c(fitur_final, "LUNG_CANCER_LABEL")],
                  kernel = "radial", probability = TRUE)
pred_svm  <- predict(model_svm, newdata = test_model_df[, fitur_final, drop = FALSE])
prob_svm_matrix <- attr(predict(model_svm, newdata = test_model_df[, fitur_final, drop = FALSE],
                                 probability = TRUE), "probabilities")
prob_svm <- prob_svm_matrix[, "YES"]
cm_svm   <- confusionMatrix(pred_svm, test_model_df$LUNG_CANCER_LABEL, positive = "YES")
print(cm_svm)

## 4.4 C. XGBoost -----------------------------------------------------------------------------
cat("\n--- C. XGBoost ---\n")
set.seed(SEED)
X_train_mat <- as.matrix(train_model_df[, fitur_final])
X_test_mat  <- as.matrix(test_model_df[, fitur_final])

# Menggunakan parameter baru XGBoost 2.0+
# Target y diubah menjadi TRUE/FALSE (logical) agar diterima oleh klasifikasi
model_xgb <- xgboost(x = X_train_mat, 
                     y = train_model_df$LUNG_CANCER == 1,  
                     max_depth = 3, 
                     learning_rate = 0.1, 
                     nrounds = 100, 
                     objective = "binary:logistic", 
                     verbosity = 0)

# Prediksi menggunakan parameter newdata
prob_xgb <- predict(model_xgb, newdata = X_test_mat)
pred_xgb <- factor(ifelse(prob_xgb > 0.5, "YES", "NO"), levels = c("NO", "YES"))
cm_xgb   <- confusionMatrix(pred_xgb, test_model_df$LUNG_CANCER_LABEL, positive = "YES")
print(cm_xgb)

## 4.5 Tabel perbandingan akhir -----------------------------------------------------------------
# Tambahkan pROC:: di depan auc() dan roc()
auc_lr  <- as.numeric(pROC::auc(pROC::roc(test_model_df$LUNG_CANCER_LABEL, prob_lr,  levels = c("NO", "YES"), direction = "<", quiet = TRUE)))
auc_svm <- as.numeric(pROC::auc(pROC::roc(test_model_df$LUNG_CANCER_LABEL, prob_svm, levels = c("NO", "YES"), direction = "<", quiet = TRUE)))
auc_xgb <- as.numeric(pROC::auc(pROC::roc(test_model_df$LUNG_CANCER_LABEL, prob_xgb, levels = c("NO", "YES"), direction = "<", quiet = TRUE)))

tabel_model <- data.frame(
  Model       = c("Logistic Regression", "SVM", "XGBoost"),
  Accuracy    = round(c(cm_lr$overall["Accuracy"], cm_svm$overall["Accuracy"], cm_xgb$overall["Accuracy"]), 4),
  Sensitivity = round(c(cm_lr$byClass["Sensitivity"], cm_svm$byClass["Sensitivity"], cm_xgb$byClass["Sensitivity"]), 4),
  F1_Score    = round(c(cm_lr$byClass["F1"], cm_svm$byClass["F1"], cm_xgb$byClass["F1"]), 4),
  ROC_AUC     = round(c(auc_lr, auc_svm, auc_xgb), 4)
)
cat("\nTabel perbandingan performa model akhir:\n")
print(kable(tabel_model))

## 4.6 Visualisasi: Confusion Matrix, FP/FN, Metrik, ROC, Densitas Probabilitas -------------------
cm_long <- bind_rows(
  as.data.frame(cm_lr$table)  %>% mutate(Model = "Logistic Regression"),
  as.data.frame(cm_svm$table) %>% mutate(Model = "SVM"),
  as.data.frame(cm_xgb$table) %>% mutate(Model = "XGBoost")
)

p_cm <- ggplot(cm_long, aes(Reference, Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), fontface = "bold", size = 5) +
  scale_fill_gradient(low = "lightblue", high = "dodgerblue4") +
  facet_wrap(~Model) +
  labs(title = "Confusion Matrix - Perbandingan Model", x = "Aktual", y = "Prediksi") +
  theme_minimal(base_size = 12)
print(p_cm)

p_error <- cm_long %>%
  filter(Prediction != Reference) %>%
  mutate(Tipe_Error = ifelse(Prediction == "YES", "False Positive", "False Negative")) %>%
  ggplot(aes(Model, Freq, fill = Tipe_Error)) +
  geom_col(position = "dodge", color = "black") +
  scale_fill_manual(values = c("False Positive" = "orange", "False Negative" = "red")) +
  labs(title = "False Positive vs False Negative per Model", y = "Jumlah Pasien") +
  theme_minimal(base_size = 12)
print(p_error)

p_metrik <- tabel_model %>%
  pivot_longer(c(Accuracy, Sensitivity, F1_Score, ROC_AUC), names_to = "Metrik", values_to = "Skor") %>%
  ggplot(aes(Model, Skor, fill = Metrik)) +
  geom_col(position = "dodge", color = "black") +
  geom_text(aes(label = round(Skor, 3)), position = position_dodge(width = 0.9), vjust = -0.4, size = 3) +
  ylim(0, 1.15) +
  labs(title = "Performa Model: Accuracy, Sensitivity, F1-Score, ROC-AUC") +
  theme_minimal(base_size = 12)
print(p_metrik)

# Tambahkan pROC:: di depan roc()
roc_lr_obj  <- pROC::roc(test_model_df$LUNG_CANCER_LABEL, prob_lr,  levels = c("NO", "YES"), direction = "<", quiet = TRUE)
roc_svm_obj <- pROC::roc(test_model_df$LUNG_CANCER_LABEL, prob_svm, levels = c("NO", "YES"), direction = "<", quiet = TRUE)
roc_xgb_obj <- pROC::roc(test_model_df$LUNG_CANCER_LABEL, prob_xgb, levels = c("NO", "YES"), direction = "<", quiet = TRUE)

p_roc <- ggroc(list("Logistic Regression" = roc_lr_obj, "SVM" = roc_svm_obj, "XGBoost" = roc_xgb_obj), linewidth = 1) +
  geom_abline(intercept = 1, slope = 1, linetype = "dashed", color = "gray") +
  labs(title = "Kurva ROC",
       subtitle = sprintf("AUC LR=%.3f | AUC SVM=%.3f | AUC XGB=%.3f", auc_lr, auc_svm, auc_xgb),
       x = "Specificity", y = "Sensitivity", color = "Model") +
  theme_minimal(base_size = 12)
print(p_roc)

df_prob_long <- data.frame(Aktual = test_model_df$LUNG_CANCER_LABEL, LR = prob_lr, SVM = prob_svm, XGBoost = prob_xgb) %>%
  pivot_longer(c(LR, SVM, XGBoost), names_to = "Model", values_to = "Probabilitas")

p_densitas <- ggplot(df_prob_long, aes(Probabilitas, fill = Aktual, color = Aktual)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~Model, ncol = 1) +
  scale_fill_manual(values = c("NO" = "lightgreen", "YES" = "lightcoral")) +
  scale_color_manual(values = c("NO" = "darkgreen", "YES" = "darkred")) +
  labs(title = "Distribusi Probabilitas Prediksi Kelas YES",
       subtitle = "Pemisahan ideal: kurva hijau dekat 0, kurva merah dekat 1") +
  theme_minimal(base_size = 12)
print(p_densitas)

## 4.7 Interpretasi model dengan LIME (model SVM) -----------------------------------------------
cat("\n--- Interpretasi LIME untuk model SVM ---\n")

model_type.svm <- function(x, ...) "classification"
predict_model.svm <- function(x, newdata, ...) {
  hasil <- predict(x, newdata, probability = TRUE)
  as.data.frame(attr(hasil, "probabilities"))
}

set.seed(SEED)
explainer_svm  <- lime::lime(train_model_df[, fitur_final, drop = FALSE], model_svm)
penjelasan_svm <- lime::explain(
  x = test_model_df[1:4, fitur_final, drop = FALSE],
  explainer = explainer_svm, n_labels = 1, n_features = min(5, length(fitur_final))
)

p_lime <- lime::plot_features(penjelasan_svm) + ggtitle("LIME: Penjelasan Prediksi 4 Pasien Pertama (Model SVM)")
print(p_lime)

## 4.8 Feature importance XGBoost (pelengkap interpretasi) -------------------------------------
xgb_importance <- xgb.importance(feature_names = fitur_final, model = model_xgb)
cat("\nFeature importance XGBoost:\n")
print(kable(xgb_importance, digits = 4))

p_xgb_imp <- xgb.ggplot.importance(xgb_importance) +
  labs(title = "Feature Importance - XGBoost") + theme_minimal(base_size = 12)
print(p_xgb_imp)

## 4.9 Simpan dataset bersih final & ringkasan akhir pipeline -----------------------------------
write_csv(df_clean, "lung_cancer_clean_final.csv")

cat("\n", strrep("=", 78), "\n", sep = "")
cat("RINGKASAN AKHIR PIPELINE\n")
cat(strrep("=", 78), "\n", sep = "")
cat(sprintf("1. Preprocessing       : metode imputasi terbaik   = %s\n", metode_terbaik))
cat(sprintf("2. Imbalance handling  : strategi terbaik          = %s (train: %d sampel)\n",
            metode_imbalance_terbaik, nrow(train_balanced)))
cat(sprintf("3. Feature engineering : metode terbaik            = %s (%d fitur dipakai)\n",
            metode_fe_terbaik, length(fitur_final)))
model_terbaik <- tabel_model$Model[which.max(tabel_model$F1_Score)]
cat(sprintf("4. Model terbaik       : %s (F1=%.4f, ROC-AUC=%.4f)\n",
            model_terbaik, max(tabel_model$F1_Score), tabel_model$ROC_AUC[which.max(tabel_model$F1_Score)]))
cat(sprintf("\nDataset bersih disimpan sebagai: lung_cancer_clean_final.csv\n"))
cat(strrep("=", 78), "\n", sep = "")
cat("PIPELINE SELESAI.\n")

# 1. Pastikan R bekerja di folder yang tepat
setwd("D:/Big Data")

# 2. Inisialisasi Git (jika belum ada)
system('git init')

# 3. Hubungkan ke GitHub (menggunakan set-url untuk menimpa jika sudah ada)
system('git remote add origin https://github.com/umamahtazkiya-eng/BigDataKelompok.git', ignore.stderr = TRUE)
system('git remote set-url origin https://github.com/umamahtazkiya-eng/BigDataKelompok.git')

# 4. Memilih 3 file spesifik yang Anda inginkan
system('git add "FINAL/lung_cancer_pipeline_FINAL.R" "survey lung cancer (1).csv" "FINAL.R"')

# 5. Memberikan pesan commit
system('git commit -m "Upload file FINAL, dataset asli, dan script FINAL.R"')

# 6. Mengubah nama branch ke main
system('git branch -M main')

# 7. Push ke GitHub
system('git push -u origin main')

cat("Proses Git via R selesai dijalankan! Silakan cek repositori GitHub Anda.\n")