# ==============================================================================
# 0. MEMUAT LIBRARY YANG DIBUTUHKAN
# ==============================================================================
# install.packages(c("e1071")) # Hilangkan komentar jika belum install e1071
library(caret)
library(smotefamily)
library(e1071)      # Library untuk Support Vector Machine (SVM)
library(xgboost)
library(ggplot2)
library(pROC)
library(dplyr)
library(tidyr)

# ==============================================================================
# 1. BACA & PEMBERSIHAN DATA (CLEANING)
# ==============================================================================
df <- read.csv("D:/Big Data/survey lung cancer (1).csv")

# Bersihkan nama kolom dari spasi berlebih
names(df) <- trimws(names(df))

# Bersihkan spasi berlebih pada teks di dalam data
df$LUNG_CANCER <- trimws(as.character(df$LUNG_CANCER))
df$GENDER <- trimws(as.character(df$GENDER))
df <- na.omit(df)

# Ubah GENDER menjadi numerik (M = 1, F = 0)
df$GENDER <- ifelse(df$GENDER == "M", 1, 0)

# Simpan target LUNG_CANCER sebagai faktor (Untuk SVM)
df$LUNG_CANCER <- as.factor(df$LUNG_CANCER)

# Buat juga target LUNG_CANCER versi numerik 1/0 (Untuk Logistic Regression & XGBoost)
df$LUNG_CANCER_NUM <- ifelse(df$LUNG_CANCER == "YES", 1, 0)

# ==============================================================================
# 2. SPLIT DATA (MENCEGAH DATA LEAKAGE)
# ==============================================================================
set.seed(42)
train_index <- createDataPartition(df$LUNG_CANCER, p = 0.8, list = FALSE)
train_data  <- df[train_index, ]
test_data   <- df[-train_index, ]

# ==============================================================================
# 3. TERAPKAN SMOTE (HANYA PADA DATA TRAIN)
# ==============================================================================
X_train <- train_data[, !(names(train_data) %in% c("LUNG_CANCER", "LUNG_CANCER_NUM"))]
X_train <- as.data.frame(lapply(X_train, as.numeric))
y_train <- as.character(train_data$LUNG_CANCER)

set.seed(42)
smote_out <- SMOTE(X_train, y_train, K = 5)
train_balanced <- smote_out$data
names(train_balanced)[names(train_balanced) == "class"] <- "LUNG_CANCER"

train_balanced$LUNG_CANCER <- as.factor(train_balanced$LUNG_CANCER)
train_balanced$LUNG_CANCER_NUM <- ifelse(train_balanced$LUNG_CANCER == "YES", 1, 0)

X_test <- test_data[, !(names(test_data) %in% c("LUNG_CANCER", "LUNG_CANCER_NUM"))]
X_test <- as.data.frame(lapply(X_test, as.numeric))

# ==============================================================================
# 4. PEMODELAN MACHINE LEARNING
# ==============================================================================

# --- A. LOGISTIC REGRESSION ---
cat("\n================ 1. LOGISTIC REGRESSION ================\n")
set.seed(42)
# Logistic Regression butuh target Numerik (0/1) 
lr_train_data <- train_balanced[, !(names(train_balanced) %in% "LUNG_CANCER")]

# Melatih model regresi logistik (family = "binomial")
lr_model <- glm(LUNG_CANCER_NUM ~ ., data = lr_train_data, family = "binomial")

# Memprediksi probabilitas
lr_prob <- predict(lr_model, X_test, type = "response")

# Mengubah probabilitas (> 0.5) menjadi YES
lr_pred <- as.factor(ifelse(lr_prob > 0.5, "YES", "NO"))
cm_lr <- confusionMatrix(lr_pred, test_data$LUNG_CANCER, positive = "YES")
print(cm_lr)

# --- B. SUPPORT VECTOR MACHINE (SVM) ---
cat("\n================ 2. SUPPORT VECTOR MACHINE ================\n")
set.seed(42)
# SVM butuh target Faktor 
svm_train_data <- train_balanced[, !(names(train_balanced) %in% "LUNG_CANCER_NUM")]

# Melatih model SVM (wajib tambah probability = TRUE agar bisa divisualisasikan ROC-nya)
svm_model <- svm(LUNG_CANCER ~ ., data = svm_train_data, kernel = "radial", probability = TRUE)

# Memprediksi kelas
svm_pred <- predict(svm_model, X_test)

# Memprediksi probabilitas untuk ROC
svm_prob_matrix <- attr(predict(svm_model, X_test, probability = TRUE), "probabilities")
svm_prob <- svm_prob_matrix[, "YES"] # Ambil kolom kelas YES saja

cm_svm <- confusionMatrix(svm_pred, test_data$LUNG_CANCER, positive = "YES")
print(cm_svm)

# --- C. XGBOOST ---
cat("\n================ 3. XGBOOST ================\n")
set.seed(42)
X_train_xgb <- train_balanced[, !(names(train_balanced) %in% c("LUNG_CANCER", "LUNG_CANCER_NUM"))]
y_train_xgb <- train_balanced$LUNG_CANCER == "YES"

xgb_model <- xgboost(data = as.matrix(X_train_xgb), 
                     label = y_train_xgb, 
                     max_depth = 3, 
                     eta = 0.1, 
                     nrounds = 100, 
                     objective = "binary:logistic", 
                     verbose = 0)
xgb_prob <- predict(xgb_model, as.matrix(X_test))
xgb_pred <- as.factor(ifelse(xgb_prob > 0.5, "YES", "NO"))
cm_xgb <- confusionMatrix(xgb_pred, test_data$LUNG_CANCER, positive = "YES")
print(cm_xgb)


# ==============================================================================
# 5. VISUALISASI HASIL EVALUASI
# ==============================================================================

# Gabungkan probabilitas prediksi dan data aktual ke dalam satu dataframe
df_probs <- data.frame(
  Actual = test_data$LUNG_CANCER,
  LR = lr_prob,
  SVM = svm_prob,
  XGB = xgb_prob
)

# --- VISUALISASI 1: CONFUSION MATRIX (HEATMAP) ---
cm_data <- bind_rows(
  as.data.frame(cm_lr$table) %>% mutate(Model = "Logistic Regression"),
  as.data.frame(cm_svm$table) %>% mutate(Model = "Support Vector Machine"),
  as.data.frame(cm_xgb$table) %>% mutate(Model = "XGBoost")
)

p1 <- ggplot(cm_data, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white", size = 0.5) +
  geom_text(aes(label = Freq), vjust = 0.5, fontface = "bold", size = 5) +
  scale_fill_gradient(low = "lightblue", high = "dodgerblue4") +
  facet_wrap(~ Model) +
  theme_minimal() +
  labs(title = "Confusion Matrix Heatmap",
       subtitle = "Perbandingan Prediksi vs Data Aktual",
       x = "Data Aktual (Reference)", y = "Prediksi Model")
print(p1)

# --- VISUALISASI 2: PERBANDINGAN FALSE POSITIVE (FP) & FALSE NEGATIVE (FN) ---
fp_fn_data <- cm_data %>%
  filter(Prediction != Reference) %>%
  mutate(Error_Type = ifelse(Prediction == "YES" & Reference == "NO", "False Positive (FP)", "False Negative (FN)"))

p2 <- ggplot(fp_fn_data, aes(x = Model, y = Freq, fill = Error_Type)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  scale_fill_manual(values = c("False Positive (FP)" = "orange", "False Negative (FN)" = "red")) +
  theme_minimal() +
  labs(title = "Perbandingan False Positives (FP) dan False Negatives (FN)",
       y = "Jumlah Pasien", x = "Model")
print(p2)

# --- VISUALISASI 3: DIAGRAM BATANG AKURASI, SENSITIVITY, & F1-SCORE ---
metrics_df <- data.frame(
  Model = c("Logistic Regression", "Support Vector Machine", "XGBoost"),
  Accuracy = c(cm_lr$overall["Accuracy"], cm_svm$overall["Accuracy"], cm_xgb$overall["Accuracy"]),
  Sensitivity = c(cm_lr$byClass["Sensitivity"], cm_svm$byClass["Sensitivity"], cm_xgb$byClass["Sensitivity"]),
  F1_Score = c(cm_lr$byClass["F1"], cm_svm$byClass["F1"], cm_xgb$byClass["F1"])
)

metrics_long <- pivot_longer(metrics_df, cols = c(Accuracy, Sensitivity, F1_Score), 
                             names_to = "Metric", values_to = "Score")

p3 <- ggplot(metrics_long, aes(x = Model, y = Score, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  geom_text(aes(label = round(Score, 2)), position = position_dodge(width = 0.9), vjust = -0.5) +
  scale_y_continuous(limits = c(0, 1.1)) +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal() +
  labs(title = "Performa Model: Akurasi, Sensitivity, dan F1-Score",
       y = "Skor (0.0 - 1.0)", x = "Model")
print(p3)

# --- VISUALISASI 4: GRAFIK ROC DAN AUC ---
roc_lr <- roc(df_probs$Actual, df_probs$LR, levels = c("NO", "YES"))
roc_svm <- roc(df_probs$Actual, df_probs$SVM, levels = c("NO", "YES"))
roc_xgb <- roc(df_probs$Actual, df_probs$XGB, levels = c("NO", "YES"))

p4 <- ggroc(list("Logistic Regression" = roc_lr, "SVM" = roc_svm, "XGBoost" = roc_xgb), size = 1) +
  geom_abline(intercept = 1, slope = 1, linetype = "dashed", color = "gray") +
  theme_minimal() +
  labs(title = "Kurva ROC (Receiver Operating Characteristic)",
       subtitle = paste0("AUC LR: ", round(auc(roc_lr), 3), 
                         " | AUC SVM: ", round(auc(roc_svm), 3), 
                         " | AUC XGB: ", round(auc(roc_xgb), 3)),
       x = "Specificity (1 - False Positive Rate)", y = "Sensitivity (True Positive Rate)",
       color = "Model")
print(p4)

# --- VISUALISASI 5: DISTRIBUSI PROBABILITAS ---
probs_long <- df_probs %>%
  pivot_longer(cols = c(LR, SVM, XGB), names_to = "Model", values_to = "Probability")

p5 <- ggplot(probs_long, aes(x = Probability, fill = Actual, color = Actual)) +
  geom_density(alpha = 0.5, size = 0.7) +
  facet_wrap(~ Model, ncol = 1) +
  scale_fill_manual(values = c("NO" = "lightgreen", "YES" = "lightcoral")) +
  scale_color_manual(values = c("NO" = "darkgreen", "YES" = "darkred")) +
  theme_minimal() +
  labs(title = "Distribusi Probabilitas Prediksi Kelas 'YES' (Kanker)",
       subtitle = "Pemisahan yang ideal: Kurva hijau di dekat 0, Kurva merah di dekat 1",
       x = "Probabilitas Diprediksi sebagai 'YES'", y = "Kepadatan (Density)")
print(p5)


# ==============================================================================
# 5. VISUALISASI METRIK (TERMASUK RECALL)
# ==============================================================================

# Membuat dataframe metrik (Termasuk Recall)
metrics_df <- data.frame(
  Model = c("Logistic Regression", "SVM", "XGBoost"),
  Accuracy = c(cm_lr$overall["Accuracy"], cm_svm$overall["Accuracy"], cm_xgb$overall["Accuracy"]),
  Recall = c(cm_lr$byClass["Recall"], cm_svm$byClass["Recall"], cm_xgb$byClass["Recall"]),
  Sensitivity = c(cm_lr$byClass["Sensitivity"], cm_svm$byClass["Sensitivity"], cm_xgb$byClass["Sensitivity"]),
  F1_Score = c(cm_lr$byClass["F1"], cm_svm$byClass["F1"], cm_xgb$byClass["F1"])
)

metrics_long <- pivot_longer(metrics_df, cols = c(Accuracy, Recall, F1_Score), 
                             names_to = "Metric", values_to = "Score")

p_metrics <- ggplot(metrics_long, aes(x = Model, y = Score, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  geom_text(aes(label = round(Score, 2)), position = position_dodge(width = 0.9), vjust = -0.5) +
  scale_y_continuous(limits = c(0, 1.1)) +
  scale_fill_brewer(palette = "Pastel1") +
  theme_minimal() +
  labs(title = "Performa Model: Akurasi, Recall, dan F1-Score", y = "Skor", x = "Model")
print(p_metrics)


# ==============================================================================
# 6. INTERPRETASI MODEL (LIME & SHAP)
# ==============================================================================

# ------------------------------------------------------------------------------
# A. LIME 
# Kita akan menjelaskan 4 pasien pertama dari data Test
# ------------------------------------------------------------------------------
cat("\nMembuat visualisasi LIME...\n")
set.seed(42)

# Gunakan lime::lime agar aman
explainer_svm <- lime::lime(X_train, svm_model)

# [SOLUSI ERROR] Gunakan lime::explain agar tidak tertukar dengan package shapr
explanation_svm <- lime::explain(
  x = X_test[1:4, ], 
  explainer = explainer_svm, 
  n_labels = 1, 
  n_features = 5
)

# ------------------------------------------------------------------------------
# A. LIME (Menggunakan Model SVM)
# Kita akan menjelaskan 4 pasien pertama dari data Test
# ------------------------------------------------------------------------------
cat("\nMembuat visualisasi LIME...\n")
set.seed(42)

# 1. [SOLUSI ERROR] Mengajari LIME bahwa model SVM kita adalah untuk Klasifikasi
model_type.svm <- function(x, ...) {
  return("classification")
}

# 2. [SOLUSI ERROR] Mengajari LIME cara mengekstrak probabilitas dari model SVM e1071
predict_model.svm <- function(x, newdata, ...) {
  # Ekstrak prediksi beserta probabilitasnya
  res <- predict(x, newdata, probability = TRUE)
  # Ambil matriks probabilitasnya saja, lalu jadikan dataframe
  probs <- as.data.frame(attr(res, "probabilities"))
  return(probs)
}

# 3. Sekarang jalankan lime explainer seperti biasa
explainer_svm <- lime::lime(X_train, svm_model)

# 4. Jelaskan prediksi 4 pasien pertama
explanation_svm <- lime::explain(
  x = X_test[1:4, ], 
  explainer = explainer_svm, 
  n_labels = 1, 
  n_features = 5
)

# 5. Plot penjelasan LIME
p_lime <- lime::plot_features(explanation_svm) + 
  ggtitle("LIME: Penjelasan Prediksi untuk 4 Pasien Pertama (SVM)")
print(p_lime)


