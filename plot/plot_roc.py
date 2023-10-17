import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load the data from .npy file
bert_base_chinese_fpr = np.load('./result/bert-base-chinese/bert-base-chinese_fpr.npy')
bert_base_chinese_tpr = np.load('./result/bert-base-chinese/bert-base-chinese_tpr.npy')

bert_base_uncased_fpr = np.load('./result/bert-base-uncased/bert-base-uncased_fpr.npy')
bert_base_uncased_tpr = np.load('./result/bert-base-uncased/bert-base-uncased_tpr.npy')

pcnn_att_fpr = np.load('./result/pcnn+att/pcnn+att_fpr.npy')
pcnn_att_tpr = np.load('./result/pcnn+att/pcnn+att_tpr.npy')

pcnn_one_fpr = np.load('./result/pcnn+one/pcnn+one_fpr.npy')
pcnn_one_tpr = np.load('./result/pcnn+one/pcnn+one_tpr.npy')

cnn_fpr = np.load('./result/cnn/cnn_fpr.npy')
cnn_tpr = np.load('./result/cnn/cnn_tpr.npy')

# Compute ROC curve and ROC area
bert_base_uncased_roc_auc = auc(bert_base_uncased_fpr, bert_base_uncased_tpr)
bert_base_chinese_roc_auc = auc(bert_base_chinese_fpr, bert_base_chinese_tpr)
pcnn_att_roc_auc = auc(pcnn_att_fpr, pcnn_att_tpr)
pcnn_one_roc_auc = auc(pcnn_one_fpr, pcnn_one_tpr)
cnn_roc_auc = auc(cnn_fpr, cnn_tpr)

# Plot ROC curve
plt.figure()
lw = 2
plt.figure(figsize=(8, 8), dpi=300)

plt.plot(cnn_fpr, cnn_tpr, linestyle='-', linewidth=2, color='#808080', label='CNN (area = %0.2f)' % cnn_roc_auc)
plt.plot(pcnn_att_fpr, pcnn_att_tpr, linestyle='-', linewidth=2, color='#0000FF', label='PCNN-ATT (area = %0.2f)' % pcnn_att_roc_auc)
plt.plot(pcnn_one_fpr, pcnn_one_tpr, linestyle='-', linewidth=2, color='#FFA500', label='PCNN-ONE (area = %0.2f)' % pcnn_one_roc_auc)
plt.plot(bert_base_uncased_fpr, bert_base_uncased_tpr, linestyle='-', linewidth=2, color='#000000', label='BERT (area = %0.2f)' % bert_base_uncased_roc_auc)
plt.plot(bert_base_chinese_fpr, bert_base_chinese_tpr, linestyle='-', linewidth=2, color='#FF0000', label='Ont4RE (area = %0.2f)' % bert_base_chinese_roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('AUROC Curve', fontsize= 18)
plt.legend(fontsize=12, loc="lower right")
plt.grid(True)
plt.savefig('./plot/pr_roc.jpg')
plt.show()
