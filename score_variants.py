import numpy as np
import pandas as pd
from pathlib import Path

from task_template import rank01, mean_rank

df = pd.read_csv("/home/atml_team052/TML-Task2-Stolen_Model_Detection/submission_features.csv").sort_values("id").reset_index(drop=True)
n = len(df)

def write(name, raw):
    score = rank01(raw, higher_is_more_stolen=True)
    out = pd.DataFrame({"id": df["id"].astype(int), "score": score})
    path = f"submission_{name}.csv"
    out.to_csv(path, index=False)
    print(path, out["score"].min(), out["score"].max())
    print(out.sort_values("score", ascending=False).head(20)["id"].tolist())

weight = mean_rank(df, ["w_cos_all", "w_cos_conv", "w_cos_bn", "w_cos_early", "w_cos_mid", "w_cos_late", "w_sign"])
bn = mean_rank(df, ["w_cos_bn"])
cka = mean_rank(df, ["cka_mean", "cka_early", "cka_mid", "cka_late", "cka_layer1", "cka_layer2", "cka_layer3", "cka_layer4", "cka_avgpool"])
early_mid_cka = mean_rank(df, ["cka_early", "cka_mid", "cka_layer1", "cka_layer2", "cka_layer3"])
late_cka = mean_rank(df, ["cka_late", "cka_layer4", "cka_avgpool"])

clean_dark = mean_rank(df, ["test_logit_cos", "test_neg_js", "test_top5_overlap", "main_neg_js", "nonmain_neg_js"])
mistake = mean_rank(df, [
    "test_same_wrong", "main_same_wrong", "nonmain_same_wrong",
    "confwrong_top1_agree", "vconfwrong_top1_agree", "confwrong_logit_cos", "confwrong_neg_js",
])
leakage = mean_rank(df, [
    "lowmargin_logit_cos", "lowmargin_neg_js", "lowmargin_top5_overlap",
    "highentropy_logit_cos", "highentropy_neg_js", "leakage_logit_cos", "leakage_neg_js",
])
ood = mean_rank(df, ["ood_logit_cos", "ood_neg_js", "ood_top5_overlap", "ood_conf_corr"])
transform = mean_rank(df, ["transform_delta_cos", "transform_stability_agree", "mixmatch_cos", "mixmatch_neg_js", "mixmatch_top1"])
mem = mean_rank(df, ["mem_loss_gap_similarity", "mem_conf_gap_similarity", "mem_acc_gap_similarity", "mem_agree_gap_similarity"])
fgsm = mean_rank(df, ["fgsm_logit_cos", "fgsm_top1_agree", "fgsm_neg_js"])
jac = mean_rank(df, ["jacobian_cos", "jacobian_sign"])

direct = 0.50 * weight + 0.35 * bn + 0.15 * cka
finetune = 0.35 * early_mid_cka + 0.25 * bn + 0.25 * mem + 0.15 * transform
distilled = 0.28 * leakage + 0.22 * ood + 0.22 * mistake + 0.18 * clean_dark + 0.10 * late_cka
boundary = 0.35 * leakage + 0.25 * fgsm + 0.20 * jac + 0.20 * mistake
dataset = 0.60 * mem + 0.25 * early_mid_cka + 0.15 * transform

# 1. Previous direct-heavy style
write("direct_heavy", np.maximum.reduce([
    direct,
    finetune,
    0.95 * distilled,
    0.95 * boundary,
    dataset,
]))

# 2. Current non-direct boost, but less extreme
write("balanced", np.maximum.reduce([
    direct,
    finetune,
    1.03 * distilled,
    1.03 * boundary,
    1.02 * dataset,
]))

# 3. Strong behavior version
write("behavior_heavy", np.maximum.reduce([
    0.90 * direct,
    1.00 * finetune,
    1.15 * distilled,
    1.15 * boundary,
    1.05 * dataset,
]))

# 4. Dataset/memorization version
write("dataset_heavy", np.maximum.reduce([
    0.95 * direct,
    1.05 * finetune,
    0.95 * distilled,
    0.95 * boundary,
    1.20 * dataset,
]))

# 5. CKA + behavior, less weight-based
write("cka_behavior", np.maximum.reduce([
    0.75 * direct,
    1.05 * finetune,
    1.10 * distilled,
    1.10 * boundary,
    1.10 * (0.45 * cka + 0.25 * leakage + 0.20 * transform + 0.10 * mistake),
]))

# 6. Multi-signal support version
specialist_max = np.maximum.reduce([direct, finetune, distilled, boundary, dataset])
specialist_mean = (direct + finetune + distilled + boundary + dataset) / 5.0
support = (
    (weight >= 0.8).astype(float)
  + (cka >= 0.8).astype(float)
  + (leakage >= 0.8).astype(float)
  + (transform >= 0.8).astype(float)
  + (fgsm >= 0.8).astype(float)
  + (jac >= 0.8).astype(float)
  + (mistake >= 0.8).astype(float)
  + (mem >= 0.8).astype(float)
) / 8.0
write("support_weighted", 0.70 * specialist_max + 0.20 * specialist_mean + 0.10 * support)

# 7. Remove OOD influence
distilled_no_ood = 0.35 * leakage + 0.30 * mistake + 0.20 * clean_dark + 0.15 * late_cka
write("no_ood", np.maximum.reduce([
    direct,
    finetune,
    1.08 * distilled_no_ood,
    1.05 * boundary,
    dataset,
]))

# 8. Remove clean_dark influence, because clean agreement can be false-positive-prone
distilled_no_clean = 0.35 * leakage + 0.25 * ood + 0.30 * mistake + 0.10 * late_cka
write("no_clean_dark", np.maximum.reduce([
    direct,
    finetune,
    1.08 * distilled_no_clean,
    1.05 * boundary,
    dataset,
]))
