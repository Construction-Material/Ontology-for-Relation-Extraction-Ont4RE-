import numpy as np
import matplotlib.pyplot as plt

# 加载数据
pcnn_att_prec = np.load('./result/pcnn+att/pcnn+att_p.npy')
pcnn_att_rec = np.load('./result/pcnn+att/pcnn+att_r.npy')

bert_base_chinese_prec = np.load('./result/bert-base-chinese/bert-base-chinese_p.npy')
bert_base_chinese_rec = np.load('result/bert-base-chinese/bert-base-chinese_r.npy')

pcnn_one_prec = np.load('./result/pcnn+one/pcnn+one_p.npy')
pcnn_one_rec = np.load('./result/pcnn+one/pcnn+one_r.npy')

cnn_prec = np.load('./result/cnn/cnn_p.npy')
cnn_rec = np.load('./result/cnn/cnn_r.npy')

bert_base_uncased_prec = np.load('./result/bert-base-uncased/bert-base-uncased_p.npy')
bert_base_uncased_rec = np.load('./result/bert-base-uncased/bert-base-uncased_r.npy')

# 画出PR曲线
plt.figure(figsize=(8, 8), dpi=300)

plt.plot(cnn_rec, cnn_prec, linestyle='-', linewidth=2, color='#808080', label='CNN')
plt.plot(pcnn_att_rec, pcnn_att_prec, linestyle='-', linewidth=2, color='#0000FF', label='PCNN-ATT')
plt.plot(pcnn_one_rec, pcnn_one_prec, linestyle='-', linewidth=2, color='#FFA500', label='PCNN-ONE')
plt.plot(bert_base_uncased_rec, bert_base_uncased_prec, linestyle='-', linewidth=2, color='#000000', label='BERT')
plt.plot(bert_base_chinese_rec, bert_base_chinese_prec, linestyle='-', linewidth=2, color='#FF0000', label='Ont4RE')

# 绘制F1参考线
recall = np.linspace(0.001, 1, 1000)
epsilon = 1e-7

for f1_score in np.arange(0.1, 1.0, 0.1):
    denominator = (2 * recall - f1_score)
    precision = np.where(denominator > 0, (recall * f1_score) / (2 * recall - f1_score), 1 - epsilon)
    plt.plot(recall, precision, linestyle='-', linewidth=1.5, color='#CCCCCC')
    plt.text(1.01, precision[-1], f'F1={f1_score:.1f}', fontsize=12, va='bottom')

# 调整x轴和y轴刻度的大小和粗细
plt.tick_params(axis='x', labelsize=14, width=2, length=6)
plt.tick_params(axis='y', labelsize=14, width=2, length=6)

# 设置标题和标签
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('AUPRC Curve', fontsize=18)
plt.xlabel('Recall', fontsize=16)
plt.ylabel('Precision', fontsize=16)
plt.legend(fontsize=12, loc='lower left')
plt.grid(False)

plt.savefig('./plot/pr_curve.jpg')
plt.show()