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
    'PathMNIST\n0.1%', 'PathMNIST\n1%', 'BloodMNIST\n0.5%', 'BloodMNIST\n1%',
    'OrganAMNIST\n0.5%', 'OrganAMNIST\n0.7%'
]
easy_learn_means = [91.2, 97.4, 92.3, 94.7, 27, 32]
easy_learn_std = [0.6, 0.4, 0.1, 0.1, 2, 5]
hard_learn_means = [59.2, 60.7, 63.8, 74.3, 25, 34]
hard_learn_std = [0.3, 2.8, 1.1, 1.1, 3, 4]
easy_contrast_means = [85.1, 94.1, 83.3, 93.6, 83.5, 85.8]
easy_contrast_std = [0.5, 0.3, 2.5, 0.7, 0.9, 1.3]
hard_contrast_means = [94.1, 96.2, 92.7, 94.0, 86.6, 89.1]
hard_contrast_std = [1.0, 0.3, 0.1, 0.3, 0.3, 0.3]

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
    loc='lower right')

plt.savefig(save_file)
