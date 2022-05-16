import os

import matplotlib.pyplot as plt
import numpy as np

save_dir = '/media/ntu/volume1/home/s121md302_06/workspace/code/mmselfsup/work_dirs/histogram'  # noqa
save_file = os.path.join(save_dir, 'histogram_v1.png')

palette = ['#176D9C', '#C38820', '#158A6A', '#BA611B', '#C182B5']

plt.rcParams['axes.linewidth'] = 10
plt.rcParams.update({'font.size': 120})
fig = plt.figure(figsize=(100, 30))
ax = fig.add_subplot(111)

N = 6
xTickMarks = [
    'PathMNIST\n0.2%', 'PathMNIST\n0.5%', 'BloodMNIST\n0.2%',
    'BloodMNIST\n0.5%', 'OrganAMNIST\n0.2%', 'OrganAMNIST\n0.5%'
]
easy_learn_means = [94.0, 96.2, 88.6, 92.3, 84.8, 88.9]
easy_learn_std = [0.0, 0.0, 2.1, 0.1, 0.9, 0.3]
hard_learn_means = [58.7, 62.5, 57.5, 63.8, 66.8, 72.6]
hard_learn_std = [2.8, 0.4, 0.2, 1.1, 1.9, 1.3]
easy_contrast_means = [91.4, 94.4, 76.7, 83.3, 80.6, 83.5]
easy_contrast_std = [0.1, 1.1, 1.5, 2.5, 1.1, 0.9]
hard_contrast_means = [94.8, 95.9, 90.3, 92.7, 85.8, 86.6]
hard_contrast_std = [0.8, 0.7, 0.1, 0.1, 0.3, 0.3]

ind = np.arange(N)  # the x locations for the groups
width = 0.15  # the width of the bars
error_kw = dict(elinewidth=20, ecolor='black', capsize=20)
rects1 = ax.bar(
    ind,
    easy_learn_means,
    width,
    color=palette[0],
    yerr=easy_learn_std,
    error_kw=error_kw)

rects2 = ax.bar(
    ind + width,
    hard_learn_means,
    width,
    color=palette[1],
    yerr=hard_learn_std,
    error_kw=error_kw)

rects3 = ax.bar(
    ind + 2 * width,
    easy_contrast_means,
    width,
    color=palette[2],
    yerr=easy_contrast_std,
    error_kw=error_kw)

rects4 = ax.bar(
    ind + 3 * width,
    hard_contrast_means,
    width,
    color=palette[3],
    yerr=hard_contrast_std,
    error_kw=error_kw)

# axes and labels
ax.set_xlim(-width, len(ind) + width)
ax.set_ylim(50, 100)
ax.set_ylabel('AUC (%)')
# ax.set_title('Scores by group and gender')
ax.set_xticks(ind + width)
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames,
         #  rotation=45,
         )
plt.tight_layout(pad=0.5)

ax.legend(
    (rects1[0], rects2[0], rects3[0], rects4[0]),
    ('Easy-to-learn', 'Hard-to-learn', 'Easy-to-contrast', 'Hard-to-contrast'),
    loc='lower right',
    ncol=2)

plt.savefig(save_file)
