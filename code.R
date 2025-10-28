library(mlr3verse)
library(mlr3)
library(mlr3proba)
library(mlr3tuning)
library(mlr3extralearners)
library(paradox)
library(data.table)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(riskRegression)
library(pec)
library(survcomp)
library(rms)
library(Hmisc)
library(dcurves)
library(pROC)
library(survival)
library(survminer)
library(ggsurvfit)
library(showtext)
library(sysfonts)
library(reticulate)


# Work path
setwd('C:/Users/BIJINZHE/Desktop')
# load
data <- read.csv("data.csv", stringsAsFactors = FALSE)
set.seed(2025)
data$Time[data$Time <= 0] <- 1e-3

bootstrap_once_normal <- function(df, seed) {
  set.seed(seed)
  n <- nrow(df)
  idx <- sample.int(n, size = n, replace = TRUE)
  oob <- setdiff(seq_len(n), unique(idx))
  list(train = df[idx, , drop = FALSE],
       test  = df[oob, , drop = FALSE],
       idx   = idx,
       oob   = oob)
}
sp <- bootstrap_once_normal(data, 2025)
traindata <- sp$train
testdata  <- sp$test

tsk_train <- TaskSurv$new("task_train", backend = traindata, time = "Time", event = "Event")
tsk_test  <- TaskSurv$new("task_test",  backend = testdata,  time = "Time", event = "Event")
meas <- msr("surv.cindex")
inner_rsmp <- rsmp("cv", folds = 5)


## Learners
p <- length(tsk_train$feature_names)

# CoxPH
lrn_coxph <- lrn("surv.coxph",
                 ties    = "efron", 
                 robust  = FALSE,
                 iter.max = 100)

# LASSO-Cox
lrn_glmnet <- lrn("surv.glmnet", alpha = 1)
space_glmnet <- ParamSet$new(list(ParamDbl$new("lambda", lower = 1e-4, upper = 1, tags="logscale")))
at_glmnet <- AutoTuner$new(lrn_glmnet, inner_rsmp, meas, tnr("grid_search"), trm("none"), space_glmnet)

# RSF
lrn_rsf <- lrn("surv.ranger", num.trees = 1000, importance = "impurity")
space_rsf <- ParamSet$new(list(
  ParamInt$new("mtry", lower = 1L, upper = 4L),
  ParamInt$new("min.node.size", lower = 3, upper = 50),
  ParamInt$new("num.trees", lower = 500, upper = 1500),
  ParamDbl$new("sample.fraction", lower = 0.5, upper = 1.0)
))
at_rsf <- AutoTuner$new(lrn_rsf, inner_rsmp, meas, tnr("grid_search"), trm("none"), space_rsf)

# GBM 
lrn_gbm <- lrn(
  "surv.gbm",
  distribution   = "coxph",   
  bag.fraction   = 0.7,      
  keep.data      = FALSE,
  verbose        = FALSE)
space_gbm <- ParamSet$new(list(
  ParamInt$new("n.trees",           lower = 500,  upper = 1500),
  ParamInt$new("interaction.depth", lower = 1,    upper = 5),
  ParamDbl$new("shrinkage",         lower = 0.01, upper = 0.10),
  ParamInt$new("n.minobsinnode",    lower = 5,    upper = 30),
  ParamDbl$new("bag.fraction",      lower = 0.5,  upper = 1.0)
))
at_gbm <- AutoTuner$new(
  learner      = lrn_gbm,
  resampling   = inner_rsmp,     
  measure      = meas,           
  tuner        = tnr("grid_search"),
  terminator   = trm("none"),
  search_space = space_gbm)

# survivalSVM
lrn_svm <- lrn("surv.svm", kernel = "lin_kernel")
space_svm <- ParamSet$new(list(
  ParamDbl$new("gamma.mu", lower = 1e-3, upper = 1),
  ParamDbl$new("cost",     lower = 0.1,  upper = 100),
  ParamDbl$new("margin",   lower = 1e-3, upper = 0.5),
  ParamInt$new("maxiter",  lower = 50,   upper = 300)
))
at_svm <- AutoTuner$new(lrn_svm, inner_rsmp, meas, tnr("grid_search"), trm("none"), space_svm)

# XGBoost
lrn_xgb <- lrn("surv.xgboost", objective = "survival:cox", eval_metric = "cox-nloglik")
space_xgb <- ParamSet$new(list(
  ParamInt$new("nrounds", lower = 200, upper = 1000),
  ParamDbl$new("eta",     lower = 0.01, upper = 0.3),
  ParamInt$new("max_depth", lower = 2, upper = 8),
  ParamDbl$new("subsample", lower = 0.5, upper = 1.0),
  ParamDbl$new("colsample_bytree", lower = 0.5, upper = 1.0),
  ParamDbl$new("min_child_weight", lower = 0.1, upper = 10)
))
at_xgb <- AutoTuner$new(lrn_xgb, inner_rsmp, meas, tnr("grid_search"), trm("none"), space_xgb)

learners <- list(
  CoxPH       = lrn_coxph,
  Lasso_Cox   = at_glmnet,
  RSF         = at_rsf,
  GBM         = at_gbm,
  survivalSVM = at_svm,
  XGBoost     = at_xgb
)


# DeepSurv 
deepsurv_fit <- function(train_df, epochs, batch_size, learning_rate, layers, dropout, weight_decay, val_frac = 0.2, patience = 20L, min_delta = 0.0, seed = 2025L) {
  np    <- import("numpy", convert = FALSE)
  pd    <- import("pandas", convert = FALSE)
  torch <- import("torch", convert = FALSE)
  pycox <- import("pycox", convert = FALSE)
  tt    <- import("torchtuples", convert = FALSE)
  sk    <- import("sklearn.preprocessing", convert = FALSE)
  
  torch$manual_seed(as.integer(seed))
  np$random$seed(as.integer(seed))

  x_all    <- as.matrix(train_df[, setdiff(colnames(train_df), c("Time","Event")), drop=FALSE])
  y_time   <- as.numeric(train_df[["Time"]])
  y_event  <- as.integer(train_df[["Event"]])
  
  n <- nrow(x_all)
  n_val <- max(1L, as.integer(round(n * val_frac)))
  set.seed(seed)
  vidx <- sample.int(n, n_val)
  tidx <- setdiff(seq_len(n), vidx)
  
  x_tr <- x_all[tidx, , drop=FALSE]; y_time_tr <- y_time[tidx]; y_event_tr <- y_event[tidx]
  x_va <- x_all[vidx, , drop=FALSE]; y_time_va <- y_time[vidx]; y_event_va <- y_event[vidx]
  
  scaler   <- sk$StandardScaler()$fit(r_to_py(x_tr))
  x_tr_sc  <- scaler$transform(r_to_py(x_tr))
  x_va_sc  <- scaler$transform(r_to_py(x_va))
  
  in_features <- as.integer(ncol(x_all))
  net <- tt$practical$MLPVanilla(
    in_features = in_features,
    num_nodes   = r_to_py(as.integer(layers)),  
    out_features = as.integer(1L),
    batch_norm   = FALSE,
    dropout      = r_to_py(as.numeric(dropout))
  )
  
  model <- pycox$models$cox$CoxPH(net, tt$optim$Adam)

  model$optimizer <- tt$optim$Adam(model$net$parameters(),
                                   lr = as.numeric(learning_rate),
                                   weight_decay = as.numeric(weight_decay))

  callbacks <- reticulate::tuple(
    tt$callbacks$EarlyStopping(patience = as.integer(patience),
                               min_delta = as.numeric(min_delta),
                               restore_best = TRUE)
  )
  
  y_tr_tuple <- reticulate::tuple(r_to_py(y_time_tr), r_to_py(y_event_tr))
  y_va_tuple <- reticulate::tuple(r_to_py(y_time_va), r_to_py(y_event_va))
  
  hist <- model$fit(
    input = x_tr_sc,
    target = y_tr_tuple,
    batch_size = as.integer(batch_size),
    epochs = as.integer(epochs),     
    callbacks = callbacks,
    verbose = FALSE,
    val_data = reticulate::tuple(x_va_sc, y_va_tuple)
  )

  list(
    model     = model,
    scaler    = scaler,
    history   = hist,                               
    best_epoch = as.integer(model$epoch),           
    info = list(patience = patience, min_delta = min_delta, val_frac = val_frac)
  )
}
deepsurv_predict_risk <- function(bundle, new_df) {
  x <- as.matrix(new_df[, setdiff(colnames(new_df), c("Time","Event")), drop=FALSE])
  x_sc <- bundle$scaler$transform(r_to_py(x))
  as.numeric(bundle$model$predict(x_sc)$numpy())
}

## hyperparameter tuning
fit_and_predict_all <- function(at_list, train_task, train_df, test_df) {
  risks_train <- list(); risks_test <- list(); best_pars <- list()
  
  for (nm in names(at_list)) {
    at <- at_list[[nm]]$clone(deep = TRUE)
    at$train(train_task)
    pr_tr <- at$predict(train_task)
    tsk_te <- TaskSurv$new("tmp_test", backend = test_df, time = "Time", event = "Event")
    pr_te <- at$predict(tsk_te)
    risks_train[[nm]] <- as.numeric(pr_tr$response)
    risks_test[[nm]]  <- as.numeric(pr_te$response)
    best_pars[[nm]]   <- at$learner$param_set$values
  }
  
  grid_layers       <- list(c(32L,32L), c(64L,64L), c(128L,64L))
  grid_lr           <- c(1e-4, 5e-4, 1e-3)
  grid_wd           <- c(1e-5, 1e-4, 5e-4)
  grid_do           <- c(0.0, 0.1, 0.2, 0.4)
  grid_epochs       <- c(200L, 500L, 1000L)
  grid_batch_size   <- c(64L, 128L, 256L)
  
  grids <- list()
  for (L in grid_layers) for (lr in grid_lr) for (wd in grid_wd)
    for (do in grid_do) for (ep in grid_epochs) for (bs in grid_batch_size) {
      grids[[length(grids) + 1L]] <- list(layers=L, lr=lr, wd=wd, do=do, epochs=ep, bs=bs)
    }
  
  folds <- split(seq_len(nrow(train_df)), sample(rep(1:5, length.out=nrow(train_df))))
  cv_scores <- sapply(seq_along(grids), function(gi) {
    g <- grids[[gi]]
    mean(sapply(seq_along(folds), function(i){
      tr_idx <- setdiff(seq_len(nrow(train_df)), folds[[i]])
      va_idx <- folds[[i]]
      m <- deepsurv_fit(train_df[tr_idx,], g$epochs, g$bs, g$lr, g$layers, g$do, g$wd)
      r_va <- deepsurv_predict_risk(m, train_df[va_idx,])
      as.numeric(pec::cindex(Surv(train_df$Time[va_idx], train_df$Event[va_idx]) ~ r_va)[1])
    }))
  })
  best_g <- grids[[which.max(cv_scores)]]
  m_final <- deepsurv_fit(train_df, best_g$epochs, best_g$bs, best_g$lr, best_g$layers, best_g$do, best_g$wd)
  risks_train[["Deepsurv"]] <- deepsurv_predict_risk(m_final, train_df)
  risks_test [["Deepsurv"]] <- deepsurv_predict_risk(m_final, test_df)
  best_pars[["Deepsurv"]]   <- best_g
  
  list(
    trainplot_data = cbind(train_df[,c("Time","Event")], as.data.frame(risks_train)),
    testplot_data  = cbind(test_df[,c("Time","Event")],  as.data.frame(risks_test)),
    best_params    = best_pars,
    final_model    = m_final
  )
}
fit_out <- fit_and_predict_all(learners, tsk_train, traindata, testdata)
trainplot_data <- as.data.frame(fit_out$trainplot_data)
testplot_data  <- as.data.frame(fit_out$testplot_data)


## Colors 
model_colors <- c(
  "Lasso_Cox"   = "#E58606",
  "GBM"         = "#5D69B1",
  "RSF"         = "#52BCA3",
  "survivalSVM" = "#99C945",
  "XGBoost"     = "#2F8AC4",
  "CoxPH"       = "#FFD92F",
  "Deepsurv"    = "#E73F74"
)
MODEL_ORDER <- names(model_colors)

fit_risk_cox <- function(df, models){
  lapply(models, function(m){
    coxph(as.formula(paste0("Surv(Time,Event) ~ `", m, "`")), data=df, x=TRUE, y=TRUE)
  }) |> rlang::set_names(models)
}

## time-AUC
Score_AUC <- function(fit_list, df, times, B=1000, seed=1){
  set.seed(seed)
  riskRegression::Score(
    object     = fit_list,
    formula    = Surv(Time, Event) ~ 1,
    data       = df,
    metrics    = "AUC",
    times      = times,
    conf.int   = TRUE,
    B          = B,
    summary    = NULL)
}
plot_time_auc <- function(score_obj, file_pdf, title_lab, MODEL_ORDER=NULL, model_colors=NULL){
  auc_df <- as.data.frame(score_obj$AUC$score)
  if (!is.null(MODEL_ORDER)) auc_df$model <- factor(auc_df$model, levels = MODEL_ORDER)
  p <- ggplot(auc_df, aes(times, AUC, group = model, color = model, fill = model)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, linewidth = 0) +
    geom_line(linewidth = 1) +
    labs(title = title_lab, x = "Time (months)", y = "AUC") +
    coord_cartesian(ylim = c(0.5, 1)) +
    theme_classic(base_family="serif") +
    theme(plot.title = element_text(hjust = 0.5, size = 15),
          axis.text   = element_text(size = 12, face = "bold"),
          axis.title  = element_text(size = 12),
          legend.title= element_blank(),
          legend.text = element_text(size = 10, face = "bold"),
          legend.background = element_rect(fill = rgb(1,1,1,0.5), colour = "grey"),
          legend.position   = c(0.98, 0.02),
          legend.justification = c("right","bottom"),
          panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.8),
          axis.line    = element_blank())
  if (!is.null(model_colors)) {
    p <- p + scale_color_manual(values = model_colors, drop = FALSE) +
      scale_fill_manual(values  = model_colors, drop = FALSE)
  }
  pdf(file_pdf, 10.5, 7, family = "serif"); print(p); dev.off()
  invisible(auc_df)}

times_tr <- .ensure_times(trainplot_data, eval_times)
times_te <- .ensure_times(testplot_data,  eval_times)

train_AUCplot <- Score_AUC(fit_list_train, trainplot_data, times_tr, B=1000, seed=2025)
test_AUCplot  <- Score_AUC(fit_list_test,  testplot_data,  times_te, B=1000, seed=2025)

auc_tr_df <- plot_time_auc(train_AUCplot, "Train_AUC.pdf", "Training set", MODEL_ORDER, model_colors)
auc_te_df <- plot_time_auc(test_AUCplot,  "Test_AUC.pdf",  "Validation set", MODEL_ORDER, model_colors)

## C-index 
.time_cindex_once <- function(fit_list, df, times){
  ci_obj <- pec::cindex(object = fit_list, formula = Surv(Time, Event) ~ 1,
                        data = df, eval.times = times)
  out <- data.frame(times = times, as.data.frame(ci_obj$AppCindex))
  out_long <- out |>
    tidyr::pivot_longer(cols = -times, names_to = "model", values_to = "Cindex")
  out_long
}
time_cindex_boot <- function(fit_list, df, times, B=1000, seed=2025){
  set.seed(seed)
  n <- nrow(df)
  base <- .time_cindex_once(fit_list, df, times)
  base$model <- factor(base$model, levels = names(fit_list))
  acc <- vector("list", B)
  for (b in 1:B){
    idx <- sample.int(n, n, replace = TRUE)
    d_b <- df[idx, , drop = FALSE]
    acc[[b]] <- .time_cindex_once(fit_list, d_b, times)
  }
  boot_df <- dplyr::bind_rows(acc, .id = "bs")
  summ <- boot_df |>
    group_by(model, times) |>
    summarise(lower = quantile(Cindex, 0.025, na.rm=TRUE),
              upper = quantile(Cindex, 0.975, na.rm=TRUE),
              .groups="drop")
  out <- base |> left_join(summ, by = c("model","times"))
  out
}
plot_cindex_curve <- function(df_long, file_pdf, title_lab, MODEL_ORDER=NULL, model_colors=NULL){
  if (!is.null(MODEL_ORDER)) df_long$model <- factor(df_long$model, levels = MODEL_ORDER)
  p <- ggplot(df_long, aes(times, Cindex, group = model, color = model, fill = model)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, linewidth = 0) +
    geom_line(linewidth = 1) +
    labs(title = title_lab, x = "Time (months)", y = "C-index") +
    coord_cartesian(ylim = c(0.5, 1)) +
    theme_classic(base_family="serif") +
    theme(plot.title = element_text(hjust = 0.5, size = 15),
          axis.text   = element_text(size = 12, face = "bold"),
          axis.title  = element_text(size = 12),
          legend.title= element_blank(),
          legend.text = element_text(size = 11, face = "bold"),
          legend.background = element_rect(fill = rgb(1,1,1,0.5), colour = "grey"),
          legend.position   = c(0.98, 0.02),
          legend.justification = c("right","bottom"),
          panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.8),
          axis.line    = element_blank())
  if (!is.null(model_colors)) {
    p <- p + scale_color_manual(values = model_colors, drop = FALSE) +
      scale_fill_manual(values  = model_colors, drop = FALSE)
  }
  pdf(file_pdf, 7, 7, family = "serif"); print(p); dev.off()
}

cidx_tr <- time_cindex_boot(fit_list_train, trainplot_data, times_tr, B=1000, seed=2025)
cidx_te <- time_cindex_boot(fit_list_test,  testplot_data,  times_te, B=1000, seed=2025)

plot_cindex_curve(cidx_tr, "Train_C_index.pdf", "Training set", MODEL_ORDER, model_colors)
plot_cindex_curve(cidx_te, "Test_C_index.pdf",  "Validation set", MODEL_ORDER, model_colors)

global_cindex_boot <- function(fit_list, df, B=1000, seed=2025){
  set.seed(seed)
  n <- nrow(df)
  scores0 <- lapply(fit_list, function(f) as.numeric(predict(f, newdata = df, type = "risk")))
  est <- lapply(names(scores0), function(m){
    ci <- survcomp::concordance.index(x = scores0[[m]],
                                      surv.time = df$Time, surv.event = df$Event,
                                      method = "noether")
    data.frame(model = m, C_index = ci$c.index, stringsAsFactors = FALSE)
  }) |> bind_rows()

  boot_list <- vector("list", B)
  for (b in 1:B){
    idx <- sample.int(n, n, replace = TRUE)
    d_b <- df[idx, , drop = FALSE]
    scores_b <- lapply(fit_list, function(f) as.numeric(predict(f, newdata = d_b, type = "risk")))
    boot_list[[b]] <- lapply(names(scores_b), function(m){
      ci <- survcomp::concordance.index(x = scores_b[[m]],
                                        surv.time = d_b$Time, surv.event = d_b$Event,
                                        method = "noether")
      data.frame(model = m, cindex_b = ci$c.index)
    }) |> bind_rows()
  }
  boot_df <- bind_rows(boot_list, .id = "bs")
  ci_tbl <- boot_df |>
    group_by(model) |>
    summarise(lower = quantile(cindex_b, 0.025, na.rm=TRUE),
              upper = quantile(cindex_b, 0.975, na.rm=TRUE),
              .groups="drop")
  est |> left_join(ci_tbl, by = "model")
}

gc_tr <- global_cindex_boot(fit_list_train, trainplot_data, B=1000, seed=2025)
gc_te <- global_cindex_boot(fit_list_test,  testplot_data,  B=1000, seed=2025)


## t-Brier + IBS
Score_Brier_curve <- function(fit_list, df, times, B=1000, seed=2025, cens.model="km"){
  set.seed(seed)
  riskRegression::Score(
    object     = fit_list,
    formula    = Surv(Time, Event) ~ 1,
    data       = df,
    metrics    = "Brier",
    times      = times,
    conf.int   = TRUE,
    B          = B,
    summary    = NULL,
    cens.model = cens.model
  )
}
Score_IBS <- function(fit_list, df, times, B=1000, seed=2025, cens.model="km"){
  set.seed(seed + 1)
  riskRegression::Score(
    object     = fit_list,
    formula    = Surv(Time, Event) ~ 1,
    data       = df,
    metrics    = "Brier",
    times      = times,
    conf.int   = TRUE,
    B          = B,
    summary    = "ibs",
    cens.model = cens.model
  )
}
extract_brier_df <- function(score_obj){
  as.data.frame(plotBrier(score_obj))   # columns: model, times, Brier, lower, upper
}
extract_ibs_table <- function(score_ibs){
  x <- NULL
  try({ x <- score_ibs$Brier$score$Brier$IBS }, silent = TRUE)
  if (is.null(x)) stop("")
  out <- as.data.frame(x)
  nm <- names(out); names(out) <- sub("^ibs$", "IBS", nm, ignore.case = TRUE)
  if (!"model" %in% names(out)) names(out)[1] <- "model"
  out[, c("model","IBS","lower","upper")]
}
plot_brier <- function(brier_df, file_pdf, title_lab, MODEL_ORDER=NULL, model_colors=NULL, ylim=c(0,0.25)){
  if (!is.null(MODEL_ORDER)) brier_df$model <- factor(brier_df$model, levels = MODEL_ORDER)
  p <- ggplot(brier_df, aes(times, Brier, group = model, color = model, fill = model)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, linewidth = 0) +
    geom_line(linewidth=1) +
    labs(title = title_lab, x = "Time (months)", y = "Brier score") +
    coord_cartesian(ylim = ylim) +
    theme_classic(base_family = "serif") +
    theme(plot.title = element_text(hjust = 0.5, size = 15),
          axis.text   = element_text(size = 12, face = "bold"),
          axis.title  = element_text(size = 12),
          legend.title= element_blank(),
          legend.text = element_text(size = 11, face = "bold"),
          legend.background = element_rect(fill = rgb(1,1,1,0.5), colour = "grey"),
          legend.position   = c(0.98, 0.02),
          legend.justification = c("right","bottom"),
          panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.8),
          axis.line    = element_blank())
  if (!is.null(model_colors)) {
    p <- p + scale_color_manual(values = model_colors, drop = FALSE) +
      scale_fill_manual(values  = model_colors, drop = FALSE)
  }
  pdf(file_pdf, 7, 7, family = "serif"); print(p); dev.off()
}

brier_train_obj <- Score_Brier_curve(fit_list_train, trainplot_data, times_tr, B=1000, seed=2025)
brier_test_obj  <- Score_Brier_curve(fit_list_test,  testplot_data,  times_te, B=1000, seed=2025)
brier_train_df  <- extract_brier_df(brier_train_obj)
brier_test_df   <- extract_brier_df(brier_test_obj)

plot_brier(brier_train_df, "Train_Brier.pdf", "Training set",   MODEL_ORDER, model_colors)
plot_brier(brier_test_df,  "Test_Brier.pdf",  "Validation set", MODEL_ORDER, model_colors)

ibs_train_obj <- Score_IBS(fit_list_train, trainplot_data, times_tr, B=1000, seed=2025)
ibs_test_obj  <- Score_IBS(fit_list_test,  testplot_data,  times_te, B=1000, seed=2025)
ibs_train_tbl <- extract_ibs_table(ibs_train_obj)
ibs_test_tbl  <- extract_ibs_table(ibs_test_obj)

## Calibration
predrisk_at_time <- function(fit, newdata, t){
  as.numeric(predictRisk(fit, newdata = newdata, times = t))
}
calibration_deciles <- function(fit_list, newdata, times = c(12,24,36), g = 10){
  out <- list()
  for (t in times){
    for (nm in names(fit_list)){
      pr  <- predrisk_at_time(fit_list[[nm]], newdata, t)
      grp <- Hmisc::cut2(pr, g = g)
      dat <- data.frame(Time = newdata$Time, Event = newdata$Event,
                        p = pr, grp = grp, model = nm)
      bybin <- dat |>
        group_by(model, grp) |>
        summarise(prob_pred_mean = mean(p, na.rm=TRUE),
                  n = dplyr::n(),
                  .groups="drop") |>
        left_join(
          dat |>
            group_by(grp) |>
            summarise(prob_obs = {
              sf <- survfit(Surv(Time, Event) ~ 1, data = cur_data())
              sm <- summary(sf, times = t)
              1 - ifelse(length(sm$surv)==0, NA_real_, sm$surv[1])
            }, .groups="drop"),
          by = "grp"
        ) |>
        mutate(u = t)
      out[[length(out)+1]] <- bybin
    }
  }
  bind_rows(out)
}
plot_calibration_loess <- function(df_bins, u, file_pdf, MODEL_ORDER=NULL, model_colors=NULL){
  d <- df_bins %>% filter(u == !!u)
  if (!is.null(MODEL_ORDER)) d$model <- factor(d$model, levels = MODEL_ORDER)
  p <- ggplot(d, aes(x = prob_pred_mean, y = prob_obs, color = model)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dotdash") +
    geom_point(size = 2.2, alpha = 0.9) +
    geom_smooth(method = "loess", se = FALSE, linewidth = 1.1, span = 0.8) +
    scale_x_continuous(limits = c(0,1), breaks = seq(0,1,0.2)) +
    scale_y_continuous(limits = c(0,1), breaks = seq(0,1,0.2)) +
    labs(title = paste0("Validation set – ", u, " months"),
         x = "Mean predicted probability",
         y = "Observed failure fraction (KM)") +
    theme_bw(base_family = "serif") +
    theme(plot.title = element_text(hjust = 0.5, size = 15),
          axis.text   = element_text(size = 12, face = "bold"),
          axis.title  = element_text(size = 12),
          legend.position = c(0.88, 0.18),
          legend.text = element_text(size = 11, face = "bold"),
          panel.border = element_rect(color = "black", linewidth = 0.9),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank())
  if (!is.null(model_colors)) p <- p + scale_color_manual(values = model_colors, drop = FALSE)
  ggsave(file_pdf, p, width = 7, height = 7, device = "pdf")
}
u_vec <- c(12,24,36)
calib_bins_df <- calibration_deciles(fit_list_test, testplot_data, times = u_vec, g = 10)
write.csv(calib_bins_df, "Test_calibration_bins_deciles.csv", row.names = FALSE)
for (u in u_vec) {
  plot_calibration_loess(calib_bins_df, u, sprintf("Test_calibration_%02dm.pdf", u),
                         MODEL_ORDER, model_colors)
}

## DCA
dca_times <- c(12, 24, 36)
model_labels <- rlang::set_names(
  c("CoxPH", "Lasso-Cox", "RSF", "GBM", "Survival SVM", "XGBoost", "DeepSurv"),
  MODEL_ORDER
)

for (t in dca_times) {
  dca_obj <- dcurves::dca(
    formula = as.formula(paste0("Surv(Time, Event) ~ ", paste(MODEL_ORDER, collapse = " + "))),
    data    = testplot_data,
    time    = t,
    label   = model_labels
  )
  
  p_dca <- plot(dca_obj, smooth = TRUE) +
    labs(title = paste0("Decision Curve Analysis (Validation set - ", t, " months)")) +
    scale_color_manual(values = model_colors) +
    coord_cartesian(ylim = c(-0.05, 0.6)) + # Adjust Y-axis range if needed
    theme_bw(base_family = "serif") +
    theme(
      plot.title = element_text(hjust = 0.5, size = 15),
      axis.text   = element_text(size = 12, face = "bold"),
      axis.title  = element_text(size = 12),
      legend.title = element_blank(),
      legend.text  = element_text(size = 11, face = "bold"),
      legend.position = "right",
      panel.border = element_rect(color = "black", linewidth = 0.9),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    )
  
  file_name <- sprintf("Test_DCA_%02dm.pdf", t)
  ggsave(file_name, p_dca, width = 8, height = 6.5, device = "pdf")
}

## Point ROC 
plot_point_roc <- function(df, time, file_pdf){
  y <- ifelse(df$Event==1 & df$Time <= time, 1, 0)
  rocs <- lapply(intersect(MODEL_ORDER, colnames(df)), function(nm) pROC::roc(y, df[[nm]], quiet=TRUE))
  names(rocs) <- intersect(MODEL_ORDER, colnames(df))
  AUCs <- sapply(rocs, function(r) as.numeric(pROC::auc(r)))
  CIs  <- lapply(rocs, function(r) as.numeric(pROC::ci.auc(r)))
  legend_labels <- sapply(names(rocs), function(nm){
    ci <- CIs[[nm]]; sprintf("%s (AUC: %.3f, 95%%CI: %.3f–%.3f)", nm, AUCs[[nm]], ci[1], ci[3])
  })
  rocs_s <- lapply(rocs, function(r) pROC::smooth(r, method="density"))
  p <- pROC::ggroc(rocs_s, size=1, legacy.axes=TRUE) +
    theme_bw(base_family="serif") +
    labs(title = paste0("Validation set (", time, " months)"),
         x = "1 - Specificity", y = "Sensitivity") +
    geom_segment(aes(x=0,y=0,xend=1,yend=1), colour="grey50", linetype="dotdash") +
    scale_color_manual(values = model_colors[names(rocs)], breaks = names(rocs), labels = legend_labels) +
    theme(plot.title = element_text(hjust=0.5, size=15),
          axis.text = element_text(size=12, face="bold"),
          axis.title = element_text(size=12),
          legend.title = element_blank(),
          legend.text  = element_text(size=11, face="bold"),
          legend.position = c(0.71, 0.15),
          legend.background = element_rect(fill="white", color=NA),
          panel.border = element_rect(color="black", linewidth=0.8),
          panel.background = element_blank(),
          panel.grid = element_blank())
  pdf(file_pdf, 7, 7, family = "serif"); print(p); dev.off()
}
for (tm in u_vec) plot_point_roc(testplot_data, tm, paste0("Test_ROC_", tm, "m.pdf"))

## KM 
pick_font <- function() {
  cands <- c("C:/Windows/Fonts/msyh.ttc","C:/Windows/Fonts/simhei.ttf",
             "C:/Windows/Fonts/segoeui.ttf","C:/Windows/Fonts/arialuni.ttf")
  fp <- cands[file.exists(cands)][1]
  fam <- tools::file_path_sans_ext(basename(fp))
  sysfonts::font_add(family=fam, regular=fp)
  fam
}
font_family <- pick_font(); showtext_auto(); theme_set(theme_classic(base_family = font_family))
best_model <- "Deepsurv"

res.cut <- surv_cutpoint(trainplot_data[, c("Time","Event",best_model)], time = "Time", event = "Event", variables = best_model)
cutoff <- as.numeric(summary(res.cut)$cutpoint[1])
nm_colors <- c("high"="#E05133","low"="#4877B9"); nm_labels <- c("high"="High risk","low"="Low risk")

km_plot <- function(df, cutoff, file_pdf){
  df2 <- df %>% mutate(RiskGroup = factor(ifelse(.data[[best_model]] > cutoff, "high", "low"), levels=c("high","low")))
  fit <- survfit2(Surv(Time, Event) ~ RiskGroup, data = df2)
  x_breaks <- pretty(c(0, max(df2$Time, na.rm=TRUE)), n=8)
  p <- ggsurvfit(fit, linewidth=0.9) +
    add_censor_mark(size = 3, shape = 73) +
    add_quantile(y_value = 0.5, linetype = "dashed", color = "black", linewidth = 0.6) +
    add_risktable(risktable_height=0.25, risktable_stats=c("{n.risk} ({cum.censor})"),
                  stats_label=list(n.risk="Number at risk", cum.censor="number censored"),
                  size=4, theme = theme_risktable_default() +
                    theme(plot.title=element_text(face="bold"),
                          axis.text.y=element_text(size=12, family=font_family),
                          text=element_text(family=font_family))) +
    add_risktable_strata_symbol(symbol = "\u25CF", size = 12) +
    labs(x="Time (months)", y="OS (%)", title="", color=NULL, fill=NULL) +
    scale_x_continuous(breaks = x_breaks, expand = c(0.04, 0)) +
    scale_y_continuous(breaks = seq(0,1,0.25), labels = function(x) x*100, expand = c(0,0)) +
    scale_color_manual(values=nm_colors, breaks=names(nm_labels), labels=nm_labels) +
    scale_fill_manual(values=nm_colors,  breaks=names(nm_labels), labels=nm_labels) +
    guides(color = guide_legend(ncol=1)) +
    theme_classic(base_family=font_family) +
    theme(axis.text=element_text(size=14, color="black"),
          axis.title=element_text(size=14, color="black"),
          axis.ticks.length = grid::unit(2, "mm"),
          legend.text=element_text(size=13, color="black"),
          legend.background=element_blank(),
          legend.position=c(0.12, 0.12),
          panel.grid=element_blank())
  lr  <- survival::survdiff(Surv(Time, Event) ~ RiskGroup, data = df2, rho = 0)
  pvl <- pchisq(unname(lr$chisq), df = length(lr$n)-1, lower.tail = FALSE)
  p <- p + annotate("text", x = max(df2$Time, na.rm = TRUE) * 0.65, y = 0.95,
                    label = paste0("Log-rank p = ", formatC(pvl, format="f", digits=3)),
                    hjust = 0, size = 4, family = font_family)
  ggsave(file_pdf, p, device = cairo_pdf, width = 8, height = 7, dpi = 300)
}
km_plot(trainplot_data, cutoff, paste0("Train_", best_model, "_survival.pdf"))
km_plot(testplot_data,  cutoff, paste0("Test_" , best_model, "_survival.pdf"))

## Model Interpretability Analysis
library(survex)
library(survshap)

# Custom prediction function for the PyCox model to return survival probabilities
predict_survival_prob_pycox <- function(model_bundle, newdata) {
  np    <- import("numpy", convert = FALSE)
  x_new <- as.matrix(newdata[, setdiff(colnames(newdata), c("Time", "Event")), drop = FALSE])
  x_new_sc <- model_bundle$scaler$transform(r_to_py(x_new))
  
  # Predict survival probabilities and convert to a matrix
  surv_df <- model_bundle$model$predict_surv_df(x_new_sc)
  surv_matrix <- as.matrix(surv_df)
  colnames(surv_matrix) <- as.character(surv_df$index)
  return(surv_matrix)
}

# Extract the final trained DeepSurv model
final_deepsurv_model <- fit_out$final_model

# Create the explainer object
explainer <- survex::explain(
  model = final_deepsurv_model,
  data = testdata[, setdiff(colnames(testdata), c("Time", "Event"))],
  y = Surv(testdata$Time, testdata$Event),
  predict_survival_function = function(model, newdata, ...) predict_survival_prob_pycox(model, newdata),
  label = "DeepSurv"
)

## SurvSHAP(t)
shap_obj <- predict_parts(explainer, new_observation = testdata, type = "shap", B = 25)
p_shap <- plot(shap_obj) + labs(title = "Global Feature Importance (SurvSHAP)")
ggsave("Global_SHAP_Importance.pdf", p_shap, width = 8, height = 7)

# Time-dependent feature importance plot
p_shap_time <- plot(shap_obj, show_boxplots = FALSE, bar_width = 2) + 
  labs(title = "Time-dependent Feature Importance")
ggsave("Time_Dependent_SHAP.pdf", p_shap_time, width = 10, height = 7)

## PDP
pdp_obj <- model_profile(explainer, variables = names(model_colors))
p_pdp <- plot(pdp_obj) + labs(title = "Partial Dependence Profiles")
ggsave("Partial_Dependence_Plots.pdf", p_pdp, width = 12, height = 9)

## SurvLIME
target_times_lime <- c(12, 24, 36)
event_patients_df <- testdata[testdata$Event == 1, ]

for (t_target in target_times_lime) {
  time_diff <- abs(event_patients_df$Time - t_target)
  closest_idx <- which.min(time_diff)
  patient_to_explain <- event_patients_df[closest_idx, ]
  
  actual_time <- round(patient_to_explain$Time, 1)
  
  lime_obj <- predict_parts(
    explainer,
    new_observation = patient_to_explain,
    type = "survlime",
    B = 1000
  )
  
  plot_title <- sprintf(
    "Local Explanation (SurvLIME) for Patient who died at %s months (Target: %d months)",
    actual_time,
    t_target
  )
  p_lime <- plot(lime_obj) + labs(title = plot_title)
  
  file_name <- sprintf("Local_LIME_Patient_died_at_%02dm.pdf", t_target)
  ggsave(file_name, p_lime, width = 8, height = 7)
}

## Bootstrap + OOB
bootstrap_with_oob_normal <- function(data, learners, B, seed) {
  set.seed(seed)
  out_rows <- vector("list", B * (length(learners) + 1))
  row_ptr <- 1L
  
  for (b in seq_len(B)) {
    spb <- bootstrap_once_normal(data, seed + b)
    train_df <- spb$train
    test_df  <- spb$test
    
    tsk_tr <- TaskSurv$new(paste0("boot_tr_", b), backend = train_df, time = "Time", event = "Event")
    tsk_te <- TaskSurv$new(paste0("boot_te_", b), backend = test_df,  time = "Time", event = "Event")
    
    for (nm in names(learners)) {
      at <- learners[[nm]]$clone(deep = TRUE)
      at$train(tsk_tr)
      c_tr <- at$predict(tsk_tr)$score(msr("surv.cindex"))
      c_te <- at$predict(tsk_te)$score(msr("surv.cindex"))
      out_rows[[row_ptr]] <- data.table(iter=b, model=nm,
                                        cindex_train=as.numeric(c_tr),
                                        cindex_oob=as.numeric(c_te))
      row_ptr <- row_ptr + 1L
    }
    
    grid_layers       <- list(c(32L,32L), c(64L,64L), c(128L,64L))
    grid_lr           <- c(1e-4, 5e-4, 1e-3)
    grid_wd           <- c(1e-5, 1e-4, 5e-4)
    grid_do           <- c(0.0, 0.1, 0.2, 0.4)
    grid_epochs       <- c(200L, 500L, 1000L)
    grid_batch_size   <- c(64L, 128L, 256L)
    
    grids <- list()
    for (L in grid_layers) for (lr in grid_lr) for (wd in grid_wd)
      for (do in grid_do) for (ep in grid_epochs) for (bs in grid_batch_size) {
        grids[[length(grids) + 1L]] <- list(layers=L, lr=lr, wd=wd, do=do, epochs=ep, bs=bs)
      }
    
    folds <- split(seq_len(nrow(train_df)), sample(rep(1:5, length.out=nrow(train_df))))
    cv_scores <- sapply(seq_along(grids), function(gi) {
      g <- grids[[gi]]
      mean(sapply(seq_along(folds), function(i){
        tr_idx <- setdiff(seq_len(nrow(train_df)), folds[[i]])
        va_idx <- folds[[i]]
        m <- deepsurv_fit(train_df[tr_idx,], g$epochs, g$bs, g$lr, g$layers, g$do, g$wd)
        r_va <- deepsurv_predict_risk(m, train_df[va_idx,])
        as.numeric(pec::cindex(Surv(train_df$Time[va_idx], train_df$Event[va_idx]) ~ r_va)[1])
      }))
    })
    best_g <- grids[[which.max(cv_scores)]]
    m_final <- deepsurv_fit(train_df, best_g$epochs, best_g$bs, best_g$lr, best_g$layers, best_g$do, best_g$wd)
    r_tr <- deepsurv_predict_risk(m_final, train_df)
    r_te <- deepsurv_predict_risk(m_final, test_df)
    c_tr <- as.numeric(pec::cindex(Surv(train_df$Time, train_df$Event) ~ r_tr)[1])
    c_te <- as.numeric(pec::cindex(Surv(test_df$Time,  test_df$Event)  ~ r_te)[1])
    out_rows[[row_ptr]] <- data.table(iter=b, model="Deepsurv",
                                      cindex_train=c_tr, cindex_oob=c_te)
    row_ptr <- row_ptr + 1L
  }
  
  res <- data.table::rbindlist(out_rows)
  summary_tbl <- res[, .(
    mean_oob = mean(cindex_oob, na.rm = TRUE),
    sd_oob   = sd(cindex_oob,   na.rm = TRUE),
    q2.5     = quantile(cindex_oob, 0.025, na.rm = TRUE),
    q97.5    = quantile(cindex_oob, 0.975, na.rm = TRUE)
  ), by = model][order(-mean_oob)]
  list(detail = res, summary = summary_tbl)
}

boot1000 <- bootstrap_with_oob_normal(
  data = data,
  learners = learners,
  B = 1000,
  seed = 2025
)
print(boot1000$summary)
