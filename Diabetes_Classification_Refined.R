# =============================================================================
# 🏥 Diabetes Risk Classification Model
# =============================================================================
# Author: Jason Finkle
# Project: Healthcare ML Classification
# Description: Multi-model comparison for diabetes risk prediction using
#              clinical and demographic features
# =============================================================================

# -----------------------------------------------------------------------------
# 1. SETUP AND CONFIGURATION
# -----------------------------------------------------------------------------

# Load required libraries
suppressPackageStartupMessages({
  library(tidyverse)      # Data manipulation and visualization
  library(caret)          # ML framework
  library(randomForest)   # Random Forest
  library(e1071)          # SVM
  library(pROC)           # ROC curves
  library(gridExtra)      # Plot arrangement
  library(corrplot)       # Correlation plots
  library(scales)         # Formatting
  library(viridis)        # Color palettes
  library(ggpubr)         # Publication-ready plots
  library(xgboost)        # Gradient Boosting
  library(PRROC)          # Precision-Recall curves
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
# 2. DATA LOADING AND EXPLORATION
# -----------------------------------------------------------------------------

# Load dataset
df <- read_csv("diabetes_prediction_dataset.csv", show_col_types = FALSE)

cat("\n📊 DATASET OVERVIEW\n")
cat("=" , rep("=", 50), "\n", sep = "")
cat("Observations:", nrow(df), "\n")
cat("Features:", ncol(df), "\n")
cat("Target variable: diabetes (0 = No, 1 = Yes)\n\n")

# Display structure
glimpse(df)

# Check for missing values
cat("\n🔍 Missing Values Check:\n")
missing_summary <- colSums(is.na(df))
print(missing_summary)

# Target distribution
cat("\n📈 Diabetes Prevalence:\n")
diabetes_dist <- table(df$diabetes)
print(diabetes_dist)
cat("Percentage with diabetes:", round(mean(df$diabetes) * 100, 2), "%\n")

# Summary statistics for numeric variables
cat("\n📊 Numeric Variable Summary:\n")
df %>%
  select(age, bmi, HbA1c_level, blood_glucose_level) %>%
  summary() %>%
  print()

# -----------------------------------------------------------------------------
# 3. DATA PREPROCESSING
# -----------------------------------------------------------------------------

cat("\n🔧 DATA PREPROCESSING\n")
cat("=" , rep("=", 50), "\n", sep = "")

# Step 1: Clean smoking history
df_clean <- df %>%
  # Remove unknown smoking status
  filter(!smoking_history %in% c("ever", "No Info")) %>%
  # Consolidate categories
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
  mutate(
    gender_binary = ifelse(gender == "Male", 1, 0)
  ) %>%
  # Convert diabetes to factor
  mutate(
    diabetes = factor(diabetes, levels = c(0, 1), labels = c("No", "Yes"))
  )

cat("Original observations:", nrow(df), "\n")
cat("After cleaning:", nrow(df_clean), "\n")
cat("Observations removed:", nrow(df) - nrow(df_clean), "\n")

# Create modeling dataset (numeric features only)
df_model <- df_clean %>%
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
  )

cat("\n✅ Final modeling dataset:", nrow(df_model), "observations,", 
    ncol(df_model) - 1, "features\n")

# -----------------------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------------------------------------

cat("\n📊 GENERATING EDA VISUALIZATIONS\n")
cat("=" , rep("=", 50), "\n", sep = "")

# 4.1 Target Distribution
p_target <- ggplot(df_model, aes(x = diabetes, fill = diabetes)) +
  geom_bar(width = 0.6, color = "white", size = 0.5) +
  geom_text(stat = "count", aes(label = scales::comma(..count..)), 
            vjust = -0.5, size = 4, fontface = "bold") +
  scale_fill_manual(values = c(colors["success"], colors["danger"])) +
  scale_y_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.15))) +
  labs(
    title = "Diabetes Prevalence in Dataset",
    subtitle = paste0("Class Imbalance Ratio: 1:", 
                      round(sum(df_model$diabetes == "No") / sum(df_model$diabetes == "Yes"), 1)),
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
    scale_fill_manual(values = c(colors["primary"], colors["danger"])) +
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
    scale_fill_manual(values = c(colors["primary"], colors["danger"])) +
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
         number.cex = 0.8,
         col = colorRampPalette(c("#2E86AB", "white", "#DC3545"))(100),
         title = "Feature Correlation Matrix",
         mar = c(0, 0, 2, 0))
dev.off()
cat("✅ Saved: figures/04_correlation_heatmap.png\n")

# 4.5 Age vs Blood Glucose Scatter
p_scatter <- ggplot(df_model, aes(x = age, y = blood_glucose_level, color = diabetes)) +
  geom_point(alpha = 0.4, size = 1.5) +
  geom_smooth(method = "loess", se = TRUE, linewidth = 1.2) +
  scale_color_manual(values = c(colors["primary"], colors["danger"])) +
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
cat("=" , rep("=", 50), "\n", sep = "")

# Stratified split
train_index <- createDataPartition(df_model$diabetes, p = 0.7, list = FALSE)
train_data <- df_model[train_index, ]
test_data <- df_model[-train_index, ]

cat("Training set:", nrow(train_data), "observations\n")
cat("Test set:", nrow(test_data), "observations\n")
cat("Training diabetes rate:", round(mean(train_data$diabetes == "Yes") * 100, 2), "%\n")
cat("Test diabetes rate:", round(mean(test_data$diabetes == "Yes") * 100, 2), "%\n")

# -----------------------------------------------------------------------------
# 6. MODEL TRAINING WITH CROSS-VALIDATION
# -----------------------------------------------------------------------------

cat("\n🤖 TRAINING MODELS\n")
cat("=" , rep("=", 50), "\n", sep = "")

# Define cross-validation settings
cv_control <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
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
  ntree = 200,
  tuneGrid = expand.grid(mtry = c(2, 3, 4))
)
results$rf <- rf_model
cat("   CV AUC:", round(max(rf_model$results$ROC), 4), "\n")
cat("   Best mtry:", rf_model$bestTune$mtry, "\n")

# 6.3 Support Vector Machine
cat("\n🎯 Training SVM (Radial Kernel)...\n")
svm_model <- train(
  diabetes ~ .,
  data = train_data,
  method = "svmRadial",
  trControl = cv_control,
  metric = "ROC",
  tuneLength = 5,
  preProcess = c("center", "scale")
)
results$svm <- svm_model
cat("   CV AUC:", round(max(svm_model$results$ROC), 4), "\n")

# 6.4 XGBoost
cat("\n🚀 Training XGBoost...\n")
xgb_grid <- expand.grid(
  nrounds = c(100, 200),
  max_depth = c(3, 6),
  eta = c(0.1, 0.3),
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)

xgb_model <- train(
  diabetes ~ .,
  data = train_data,
  method = "xgbTree",
  trControl = cv_control,
  metric = "ROC",
  tuneGrid = xgb_grid,
  verbosity = 0
)
results$xgb <- xgb_model
cat("   CV AUC:", round(max(xgb_model$results$ROC), 4), "\n")

# 6.5 Gradient Boosting
cat("\n📊 Training Gradient Boosting...\n")
gbm_model <- train(
  diabetes ~ .,
  data = train_data,
  method = "gbm",
  trControl = cv_control,
  metric = "ROC",
  verbose = FALSE
)
results$gbm <- gbm_model
cat("   CV AUC:", round(max(gbm_model$results$ROC), 4), "\n")

# -----------------------------------------------------------------------------
# 7. MODEL EVALUATION
# -----------------------------------------------------------------------------

cat("\n📊 EVALUATING MODELS ON TEST SET\n")
cat("=" , rep("=", 50), "\n", sep = "")

# Generate predictions for all models
models <- list(
  "Logistic Regression" = log_model,
  "Random Forest" = rf_model,
  "SVM" = svm_model,
  "XGBoost" = xgb_model,
  "Gradient Boosting" = gbm_model
)

# Calculate metrics for each model
metrics_list <- map_dfr(names(models), function(model_name) {
  model <- models[[model_name]]
  
  # Predictions
  pred_class <- predict(model, test_data)
  pred_prob <- predict(model, test_data, type = "prob")[, "Yes"]
  
  # Confusion matrix
  cm <- confusionMatrix(pred_class, test_data$diabetes, positive = "Yes")
  
  # ROC
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

# Display results
cat("\n📈 MODEL PERFORMANCE COMPARISON:\n")
print(metrics_list %>% arrange(desc(AUC)))

# -----------------------------------------------------------------------------
# 8. VISUALIZATIONS
# -----------------------------------------------------------------------------

cat("\n📊 GENERATING MODEL VISUALIZATIONS\n")
cat("=" , rep("=", 50), "\n", sep = "")

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
  )

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
cat("=" , rep("=", 50), "\n", sep = "")

# Get probabilities from top 3 models
test_data_risk <- test_data %>%
  mutate(
    prob_rf = predict(rf_model, test_data, type = "prob")[, "Yes"],
    prob_xgb = predict(xgb_model, test_data, type = "prob")[, "Yes"],
    prob_gbm = predict(gbm_model, test_data, type = "prob")[, "Yes"],
    # Ensemble average
    ensemble_prob = (prob_rf + prob_xgb + prob_gbm) / 3,
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

# Risk distribution
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
    subtitle = "Combined predictions from Random Forest, XGBoost, and Gradient Boosting",
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
  scale_fill_manual(values = c(colors["primary"], colors["danger"])) +
  geom_vline(xintercept = c(0.2, 0.5, 0.8), linetype = "dashed", alpha = 0.5) +
  annotate("text", x = 0.1, y = 3, label = "Low", size = 3) +
  annotate("text", x = 0.35, y = 3, label = "Moderate", size = 3) +
  annotate("text", x = 0.65, y = 3, label = "High", size = 3) +
  annotate("text", x = 0.9, y = 3, label = "Very High", size = 3) +
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
cat("=" , rep("=", 60), "\n", sep = "")
cat("                    📊 FINAL MODEL SUMMARY\n")
cat("=" , rep("=", 60), "\n", sep = "")

cat("\n🏆 Best Performing Model:", best_model_name, "\n")
cat("   Test AUC:", round(metrics_list %>% filter(Model == best_model_name) %>% pull(AUC), 4), "\n")
cat("   Test Accuracy:", round(metrics_list %>% filter(Model == best_model_name) %>% pull(Accuracy), 4), "\n")
cat("   Test Sensitivity:", round(metrics_list %>% filter(Model == best_model_name) %>% pull(Sensitivity), 4), "\n")
cat("   Test Specificity:", round(metrics_list %>% filter(Model == best_model_name) %>% pull(Specificity), 4), "\n")

cat("\n📊 All Models Ranked by AUC:\n")
print(metrics_list %>% 
        arrange(desc(AUC)) %>%
        mutate(across(where(is.numeric), ~round(., 4))))

cat("\n📁 Figures generated and saved to 'figures/' directory:\n")
cat("   01. Target distribution\n")
cat("   02. Feature distributions\n")
cat("   03. Boxplots by diabetes status\n")
cat("   04. Correlation heatmap\n")
cat("   05. Age vs glucose scatter\n")
cat("   06. ROC curves comparison\n")
cat("   07. Metrics comparison\n")
cat("   08. Feature importance\n")
cat("   09. Confusion matrix\n")
cat("   10. Calibration plot\n")
cat("   11. Risk distribution\n")
cat("   12. Risk density\n")

cat("\n" , rep("=", 60), "\n", sep = "")
cat("      Project by Jason Finkle | github.com/jfinkle00\n")
cat("=" , rep("=", 60), "\n", sep = "")

# Save final metrics to CSV
write_csv(metrics_list, "model_metrics_summary.csv")
cat("\n✅ Metrics saved to: model_metrics_summary.csv\n")
