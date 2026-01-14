# =============================================================================
# 🏥 Diabetes Risk Classification Model
# =============================================================================
# Author: Jason Finkle
# Project: Healthcare ML Classification
# Description: Multi-model comparison for diabetes risk prediction
# =============================================================================

# -----------------------------------------------------------------------------
# 1. SETUP AND CONFIGURATION
# -----------------------------------------------------------------------------

# Load required libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(caret)
  library(randomForest)
  library(e1071)
  library(pROC)
  library(gridExtra)
  library(corrplot)
  library(scales)
  library(viridis)
  library(ggpubr)
  library(kernlab)
  library(gbm)
})

# Set seed for reproducibility
set.seed(42)

# Create output directory for figures
if (!dir.exists("figures")) dir.create("figures")

# Define color palette
colors <- c(
  primary = "#2E86AB",
  secondary = "#A23B72", 
  accent = "#F18F01",
  success = "#28A745",
  danger = "#DC3545",
  dark = "#343A40"
)

# Custom theme for plots
theme_diabetes <- function() {
  theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray40"),
      axis.title = element_text(size = 11),
      axis.text = element_text(size = 10),
      legend.title = element_text(size = 10, face = "bold"),
      legend.text = element_text(size = 9),
      panel.grid.minor = element_blank(),
      strip.text = element_text(size = 11, face = "bold")
    )
}

cat("✅ Libraries loaded and configuration complete\n")

# -----------------------------------------------------------------------------
# 2. DATA LOADING AND SAMPLING
# -----------------------------------------------------------------------------

# Load full dataset
df_full <- read_csv("diabetes_prediction_dataset.csv", show_col_types = FALSE)

cat("\n📊 FULL DATASET OVERVIEW\n")
cat("=", rep("=", 50), "\n", sep = "")
cat("Total observations:", nrow(df_full), "\n")
cat("Diabetes prevalence:", round(mean(df_full$diabetes) * 100, 2), "%\n")

# -----------------------------------------------------------------------------
# 3. DATA PREPROCESSING
# -----------------------------------------------------------------------------

cat("\n🔧 DATA PREPROCESSING\n")
cat("=", rep("=", 50), "\n", sep = "")

# Clean the full dataset first
df_clean <- df_full %>%
  # Remove unknown smoking status
  filter(!smoking_history %in% c("ever", "No Info")) %>%
  # Consolidate smoking categories
  mutate(
    smoking_status = case_when(
      smoking_history == "current" ~ "Current",
      smoking_history %in% c("not current", "former") ~ "Former",
      smoking_history == "never" ~ "Never",
      TRUE ~ NA_character_
    )
  ) %>%
  # Create ordinal encoding for smoking
  mutate(
    smoking_ordinal = case_when(
      smoking_status == "Current" ~ 2,
      smoking_status == "Former" ~ 1,
      smoking_status == "Never" ~ 0
    )
  ) %>%
  # Encode gender
  mutate(gender_binary = ifelse(gender == "Male", 1, 0))

cat("After cleaning:", nrow(df_clean), "observations\n")

# Stratified sampling to reduce dataset size for computational efficiency
SAMPLE_SIZE <- 15000

set.seed(42)
df_sampled <- df_clean %>%
  group_by(diabetes) %>%
  slice_sample(prop = SAMPLE_SIZE / nrow(df_clean)) %>%
  ungroup()

cat("Sampled dataset:", nrow(df_sampled), "observations\n")
cat("Sampled diabetes rate:", round(mean(df_sampled$diabetes) * 100, 2), "%\n")

# Create modeling dataset
# CRITICAL: "Yes" must be the FIRST level for twoClassSummary to calculate ROC correctly
df_model <- df_sampled %>%
  select(
    age, 
    bmi, 
    HbA1c_level, 
    blood_glucose_level,
    hypertension,
    heart_disease,
    smoking_ordinal,
    gender_binary,
    diabetes
  ) %>%
  mutate(
    diabetes = factor(ifelse(diabetes == 1, "Yes", "No"), levels = c("Yes", "No"))
  )

cat("\n✅ Final modeling dataset:", nrow(df_model), "observations,", 
    ncol(df_model) - 1, "features\n")
cat("Class distribution:\n")
print(table(df_model$diabetes))

# -----------------------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------------------------------------

cat("\n📊 GENERATING EDA VISUALIZATIONS\n")
cat("=", rep("=", 50), "\n", sep = "")

# 4.1 Target Distribution
p_target <- ggplot(df_model, aes(x = diabetes, fill = diabetes)) +
  geom_bar(width = 0.6, color = "white", linewidth = 0.5) +
  geom_text(stat = "count", aes(label = scales::comma(after_stat(count))), 
            vjust = -0.5, size = 4, fontface = "bold") +
  scale_fill_manual(values = c(colors["danger"], colors["success"])) +
  scale_y_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.15))) +
  labs(
    title = "Diabetes Prevalence in Dataset",
    subtitle = paste0("Sampled ", scales::comma(nrow(df_model)), " observations"),
    x = "Diabetes Status",
    y = "Count"
  ) +
  theme_diabetes() +
  theme(legend.position = "none")

ggsave("figures/01_target_distribution.png", p_target, width = 8, height = 6, dpi = 300)
cat("✅ Saved: figures/01_target_distribution.png\n")

# 4.2 Numeric Variable Distributions
numeric_vars <- c("age", "bmi", "HbA1c_level", "blood_glucose_level")

hist_list <- map(numeric_vars, function(var) {
  ggplot(df_model, aes(x = .data[[var]], fill = diabetes)) +
    geom_histogram(bins = 30, alpha = 0.7, position = "identity", color = "white") +
    scale_fill_manual(values = c(colors["danger"], colors["primary"])) +
    labs(title = var, x = NULL, y = "Count") +
    theme_diabetes() +
    theme(legend.position = "none")
})

p_histograms <- ggarrange(plotlist = hist_list, ncol = 2, nrow = 2, 
                          common.legend = TRUE, legend = "bottom")
p_histograms <- annotate_figure(p_histograms, 
                                 top = text_grob("Feature Distributions by Diabetes Status", 
                                                 face = "bold", size = 14))

ggsave("figures/02_feature_distributions.png", p_histograms, width = 12, height = 10, dpi = 300)
cat("✅ Saved: figures/02_feature_distributions.png\n")

# 4.3 Boxplots by Diabetes Status
box_list <- map(numeric_vars, function(var) {
  ggplot(df_model, aes(x = diabetes, y = .data[[var]], fill = diabetes)) +
    geom_boxplot(alpha = 0.8, outlier.alpha = 0.3) +
    scale_fill_manual(values = c(colors["danger"], colors["primary"])) +
    labs(title = var, x = NULL, y = NULL) +
    theme_diabetes() +
    theme(legend.position = "none")
})

p_boxplots <- ggarrange(plotlist = box_list, ncol = 2, nrow = 2,
                         common.legend = TRUE, legend = "bottom")
p_boxplots <- annotate_figure(p_boxplots,
                               top = text_grob("Feature Comparison by Diabetes Status",
                                               face = "bold", size = 14))

ggsave("figures/03_boxplots_by_status.png", p_boxplots, width = 12, height = 10, dpi = 300)
cat("✅ Saved: figures/03_boxplots_by_status.png\n")

# 4.4 Correlation Heatmap
cor_matrix <- df_model %>%
  mutate(diabetes_num = as.numeric(diabetes == "Yes")) %>%
  select(-diabetes) %>%
  cor()

png("figures/04_correlation_heatmap.png", width = 800, height = 700, res = 120)
corrplot(cor_matrix, 
         method = "color",
         type = "lower",
         order = "hclust",
         tl.col = "black",
         tl.srt = 45,
         addCoef.col = "black",
         number.cex = 0.7,
         col = colorRampPalette(c("#2E86AB", "white", "#DC3545"))(100),
         title = "Feature Correlation Matrix",
         mar = c(0, 0, 2, 0))
dev.off()
cat("✅ Saved: figures/04_correlation_heatmap.png\n")

# 4.5 Age vs Blood Glucose Scatter
scatter_sample <- df_model %>% slice_sample(n = 3000)

p_scatter <- ggplot(scatter_sample, aes(x = age, y = blood_glucose_level, color = diabetes)) +
  geom_point(alpha = 0.5, size = 1.5) +
  geom_smooth(method = "loess", se = TRUE, linewidth = 1.2) +
  scale_color_manual(values = c(colors["danger"], colors["primary"])) +
  labs(
    title = "Age vs Blood Glucose Level by Diabetes Status",
    subtitle = "LOESS smoothing shows trend differences",
    x = "Age (years)",
    y = "Blood Glucose Level (mg/dL)",
    color = "Diabetes"
  ) +
  theme_diabetes()

ggsave("figures/05_age_glucose_scatter.png", p_scatter, width = 10, height = 7, dpi = 300)
cat("✅ Saved: figures/05_age_glucose_scatter.png\n")

# -----------------------------------------------------------------------------
# 5. TRAIN/TEST SPLIT
# -----------------------------------------------------------------------------

cat("\n📊 SPLITTING DATA\n")
cat("=", rep("=", 50), "\n", sep = "")

train_index <- createDataPartition(df_model$diabetes, p = 0.7, list = FALSE)
train_data <- df_model[train_index, ]
test_data <- df_model[-train_index, ]

cat("Training set:", nrow(train_data), "observations\n")
cat("Test set:", nrow(test_data), "observations\n")
cat("Training diabetes (Yes) rate:", round(mean(train_data$diabetes == "Yes") * 100, 2), "%\n")
cat("Test diabetes (Yes) rate:", round(mean(test_data$diabetes == "Yes") * 100, 2), "%\n")

# -----------------------------------------------------------------------------
# 6. MODEL TRAINING
# -----------------------------------------------------------------------------

cat("\n🤖 TRAINING MODELS\n")
cat("=", rep("=", 50), "\n", sep = "")

# Cross-validation settings
cv_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Initialize results storage
results <- list()

# 6.1 Logistic Regression
cat("\n📈 Training Logistic Regression...\n")
log_model <- train(
  diabetes ~ .,
  data = train_data,
  method = "glm",
  family = "binomial",
  trControl = cv_control,
  metric = "ROC"
)
results$logistic <- log_model
cat("   CV AUC:", round(max(log_model$results$ROC), 4), "\n")

# 6.2 Random Forest
cat("\n🌲 Training Random Forest...\n")
rf_model <- train(
  diabetes ~ .,
  data = train_data,
  method = "rf",
  trControl = cv_control,
  metric = "ROC",
  ntree = 100,
  tuneGrid = data.frame(mtry = 3)
)
results$rf <- rf_model
cat("   CV AUC:", round(max(rf_model$results$ROC), 4), "\n")

# 6.3 Support Vector Machine
cat("\n🎯 Training SVM (Radial Kernel)...\n")
svm_model <- train(
  diabetes ~ .,
  data = train_data,
  method = "svmRadial",
  trControl = cv_control,
  metric = "ROC",
  tuneGrid = data.frame(sigma = 0.1, C = 1),
  preProcess = c("center", "scale")
)
results$svm <- svm_model
cat("   CV AUC:", round(max(svm_model$results$ROC), 4), "\n")

# 6.4 Gradient Boosting
cat("\n📊 Training Gradient Boosting...\n")
gbm_model <- train(
  diabetes ~ .,
  data = train_data,
  method = "gbm",
  trControl = cv_control,
  metric = "ROC",
  tuneGrid = data.frame(
    n.trees = 100,
    interaction.depth = 3,
    shrinkage = 0.1,
    n.minobsinnode = 10
  ),
  verbose = FALSE
)
results$gbm <- gbm_model
cat("   CV AUC:", round(max(gbm_model$results$ROC), 4), "\n")

# -----------------------------------------------------------------------------
# 7. MODEL EVALUATION ON TEST SET
# -----------------------------------------------------------------------------

cat("\n📊 EVALUATING MODELS ON TEST SET\n")
cat("=", rep("=", 50), "\n", sep = "")

models <- list(
  "Logistic Regression" = log_model,
  "Random Forest" = rf_model,
  "SVM" = svm_model,
  "Gradient Boosting" = gbm_model
)

# Calculate metrics for each model
metrics_list <- map_dfr(names(models), function(model_name) {
  model <- models[[model_name]]
  
  pred_class <- predict(model, test_data)
  pred_prob <- predict(model, test_data, type = "prob")[, "Yes"]
  
  cm <- confusionMatrix(pred_class, test_data$diabetes, positive = "Yes")
  roc_obj <- roc(test_data$diabetes, pred_prob, levels = c("No", "Yes"), quiet = TRUE)
  
  tibble(
    Model = model_name,
    Accuracy = cm$overall["Accuracy"],
    Sensitivity = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"],
    Precision = cm$byClass["Precision"],
    F1 = cm$byClass["F1"],
    AUC = as.numeric(auc(roc_obj))
  )
})

cat("\n📈 MODEL PERFORMANCE COMPARISON:\n")
print(metrics_list %>% arrange(desc(AUC)))

# -----------------------------------------------------------------------------
# 8. VISUALIZATIONS
# -----------------------------------------------------------------------------

cat("\n📊 GENERATING MODEL VISUALIZATIONS\n")
cat("=", rep("=", 50), "\n", sep = "")

# 8.1 ROC Curves Comparison
roc_data <- map_dfr(names(models), function(model_name) {
  model <- models[[model_name]]
  pred_prob <- predict(model, test_data, type = "prob")[, "Yes"]
  roc_obj <- roc(test_data$diabetes, pred_prob, levels = c("No", "Yes"), quiet = TRUE)
  
  tibble(
    Model = model_name,
    Sensitivity = roc_obj$sensitivities,
    Specificity = roc_obj$specificities,
    AUC = as.numeric(auc(roc_obj))
  )
})

p_roc <- ggplot(roc_data, aes(x = 1 - Specificity, y = Sensitivity, color = Model)) +
  geom_line(linewidth = 1.2) +
  geom_abline(linetype = "dashed", color = "gray50") +
  scale_color_viridis_d(option = "plasma", end = 0.9) +
  labs(
    title = "ROC Curves: Model Comparison",
    subtitle = "Higher AUC indicates better discrimination",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_diabetes() +
  theme(legend.position = "right")

ggsave("figures/06_roc_curves_comparison.png", p_roc, width = 10, height = 7, dpi = 300)
cat("✅ Saved: figures/06_roc_curves_comparison.png\n")

# 8.2 Model Metrics Comparison
metrics_long <- metrics_list %>%
  pivot_longer(cols = c(Accuracy, Sensitivity, Specificity, Precision, F1, AUC),
               names_to = "Metric", values_to = "Value")

p_metrics <- ggplot(metrics_long, aes(x = reorder(Model, Value), y = Value, fill = Model)) +
  geom_col(width = 0.7, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.3f", Value)), hjust = -0.1, size = 3) +
  facet_wrap(~Metric, scales = "free_x", ncol = 2) +
  coord_flip() +
  scale_fill_viridis_d(option = "plasma", end = 0.9) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title = "Model Performance Metrics Comparison",
    subtitle = "All metrics calculated on held-out test set",
    x = NULL,
    y = "Score"
  ) +
  theme_diabetes()

ggsave("figures/07_metrics_comparison.png", p_metrics, width = 12, height = 10, dpi = 300)
cat("✅ Saved: figures/07_metrics_comparison.png\n")

# 8.3 Feature Importance (Random Forest)
rf_importance <- varImp(rf_model)$importance %>%
  rownames_to_column("Feature") %>%
  arrange(desc(Overall)) %>%
  mutate(Feature = fct_reorder(Feature, Overall))
  
p_importance <- ggplot(rf_importance, aes(x = Feature, y = Overall, fill = Overall)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = sprintf("%.1f", Overall)), hjust = -0.1, size = 3.5) +
  coord_flip() +
  scale_fill_gradient(low = colors["primary"], high = colors["danger"]) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title = "Random Forest Feature Importance",
    subtitle = "Mean Decrease in Gini Impurity",
    x = NULL,
    y = "Importance Score"
  ) +
  theme_diabetes() +
  theme(legend.position = "none")

ggsave("figures/08_feature_importance.png", p_importance, width = 10, height = 6, dpi = 300)
cat("✅ Saved: figures/08_feature_importance.png\n")

# 8.4 Confusion Matrix Heatmap (Best Model)
best_model_name <- metrics_list %>% 
  arrange(desc(AUC)) %>% 
  slice(1) %>% 
  pull(Model)

best_model <- models[[best_model_name]]
best_pred <- predict(best_model, test_data)
cm_best <- confusionMatrix(best_pred, test_data$diabetes, positive = "Yes")

cm_df <- as.data.frame(cm_best$table) %>%
  mutate(
    Prediction = factor(Prediction, levels = c("Yes", "No")),
    Reference = factor(Reference, levels = c("No", "Yes"))
  )

p_cm <- ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white", linewidth = 2) +
  geom_text(aes(label = scales::comma(Freq)), size = 8, fontface = "bold", color = "white") +
  scale_fill_gradient(low = colors["primary"], high = colors["danger"]) +
  labs(
    title = paste("Confusion Matrix:", best_model_name),
    subtitle = paste("Accuracy:", scales::percent(cm_best$overall["Accuracy"], accuracy = 0.1)),
    x = "Actual",
    y = "Predicted"
  ) +
  theme_diabetes() +
  theme(legend.position = "none",
        panel.grid = element_blank())

ggsave("figures/09_confusion_matrix.png", p_cm, width = 8, height = 7, dpi = 300)
cat("✅ Saved: figures/09_confusion_matrix.png\n")

# 8.5 Probability Calibration Plot
best_prob <- predict(best_model, test_data, type = "prob")[, "Yes"]

calibration_data <- tibble(
  predicted_prob = best_prob,
  actual = as.numeric(test_data$diabetes == "Yes")
) %>%
  mutate(bin = cut(predicted_prob, breaks = seq(0, 1, 0.1), include.lowest = TRUE)) %>%
  group_by(bin) %>%
  summarise(
    mean_predicted = mean(predicted_prob),
    mean_actual = mean(actual),
    n = n(),
    .groups = "drop"
  ) %>%
  filter(!is.na(bin))

p_calibration <- ggplot(calibration_data, aes(x = mean_predicted, y = mean_actual)) +
  geom_abline(linetype = "dashed", color = "gray50") +
  geom_point(aes(size = n), color = colors["primary"], alpha = 0.8) +
  geom_line(color = colors["danger"], linewidth = 1) +
  scale_size_continuous(range = c(3, 12), labels = scales::comma) +
  labs(
    title = paste("Probability Calibration Plot:", best_model_name),
    subtitle = "Well-calibrated model follows the diagonal",
    x = "Mean Predicted Probability",
    y = "Observed Proportion (Actual)",
    size = "Sample Size"
  ) +
  theme_diabetes()

ggsave("figures/10_calibration_plot.png", p_calibration, width = 9, height = 7, dpi = 300)
cat("✅ Saved: figures/10_calibration_plot.png\n")

# -----------------------------------------------------------------------------
# 9. ENSEMBLE RISK SCORING
# -----------------------------------------------------------------------------

cat("\n🎯 ENSEMBLE RISK SCORING\n")
cat("=", rep("=", 50), "\n", sep = "")

# Get probabilities from top models
test_data_risk <- test_data %>%
  mutate(
    prob_rf = predict(rf_model, test_data, type = "prob")[, "Yes"],
    prob_gbm = predict(gbm_model, test_data, type = "prob")[, "Yes"],
    prob_svm = predict(svm_model, test_data, type = "prob")[, "Yes"],
    # Ensemble average
    ensemble_prob = (prob_rf + prob_gbm + prob_svm) / 3,
    # Risk tiers
    risk_tier = case_when(
      ensemble_prob <= 0.2 ~ "Low Risk",
      ensemble_prob <= 0.5 ~ "Moderate Risk",
      ensemble_prob <= 0.8 ~ "High Risk",
      TRUE ~ "Very High Risk"
    ),
    risk_tier = factor(risk_tier, levels = c("Low Risk", "Moderate Risk", 
                                              "High Risk", "Very High Risk"))
  )

cat("\n📊 Risk Tier Distribution:\n")
print(table(test_data_risk$risk_tier))

# 9.1 Risk Distribution Plot
p_risk <- ggplot(test_data_risk, aes(x = ensemble_prob, fill = risk_tier)) +
  geom_histogram(bins = 50, alpha = 0.8, color = "white") +
  scale_fill_manual(values = c(
    "Low Risk" = "#28A745",
    "Moderate Risk" = "#FFC107", 
    "High Risk" = "#FD7E14",
    "Very High Risk" = "#DC3545"
  )) +
  labs(
    title = "Ensemble Model Risk Score Distribution",
    subtitle = "Combined predictions from Random Forest, Gradient Boosting, and SVM",
    x = "Diabetes Risk Probability",
    y = "Count",
    fill = "Risk Tier"
  ) +
  theme_diabetes()

ggsave("figures/11_risk_distribution.png", p_risk, width = 11, height = 7, dpi = 300)
cat("✅ Saved: figures/11_risk_distribution.png\n")

# 9.2 Risk Density by Actual Status
p_density <- ggplot(test_data_risk, aes(x = ensemble_prob, fill = diabetes)) +
  geom_density(alpha = 0.6) +
  scale_fill_manual(values = c(colors["danger"], colors["primary"])) +
  geom_vline(xintercept = c(0.2, 0.5, 0.8), linetype = "dashed", alpha = 0.5) +
  labs(
    title = "Risk Score Density by Actual Diabetes Status",
    subtitle = "Good separation indicates effective classification",
    x = "Ensemble Risk Probability",
    y = "Density",
    fill = "Actual Status"
  ) +
  theme_diabetes()

ggsave("figures/12_risk_density.png", p_density, width = 11, height = 7, dpi = 300)
cat("✅ Saved: figures/12_risk_density.png\n")

# -----------------------------------------------------------------------------
# 10. FINAL SUMMARY
# -----------------------------------------------------------------------------

cat("\n")
cat("=", rep("=", 60), "\n", sep = "")
cat("                    📊 FINAL MODEL SUMMARY\n")
cat("=", rep("=", 60), "\n", sep = "")

cat("\n🏆 Best Performing Model:", best_model_name, "\n")
cat("   Test AUC:", round(metrics_list %>% filter(Model == best_model_name) %>% pull(AUC), 4), "\n")
cat("   Test Accuracy:", round(metrics_list %>% filter(Model == best_model_name) %>% pull(Accuracy), 4), "\n")
cat("   Test Sensitivity:", round(metrics_list %>% filter(Model == best_model_name) %>% pull(Sensitivity), 4), "\n")
cat("   Test Specificity:", round(metrics_list %>% filter(Model == best_model_name) %>% pull(Specificity), 4), "\n")

cat("\n📊 All Models Ranked by AUC:\n")
print(metrics_list %>% 
        arrange(desc(AUC)) %>%
        mutate(across(where(is.numeric), ~round(., 4))))

cat("\n📁 Figures saved to 'figures/' directory (12 total)\n")

cat("\n", rep("=", 60), "\n", sep = "")
cat("      Project by Jason Finkle | github.com/jfinkle00\n")
cat("=", rep("=", 60), "\n", sep = "")

# Save metrics
write_csv(metrics_list, "model_metrics_summary.csv")
cat("\n✅ Metrics saved to: model_metrics_summary.csv\n")
