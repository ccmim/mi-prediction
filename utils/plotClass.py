import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import scipy as sc
from sklearn import metrics


class plot_results(object):

    def __init__(self, dir_gt, dir_preds):

        self.dir_gt = dir_gt
        self.dir_preds = dir_preds
        # Path for plots
        self.dir_plots = self.dir_preds.rsplit('/',1)[0] + '/'

        # load gt
        self.all_gt = pd.read_csv(self.dir_gt)

        self.all_gt = self.all_gt[['ID', 'LVEDV_automatic', 'LVM_automatic']] # For automatic

        # if 'automatic' in self.all_gt.columns[1]:
        #     self.all_gt = self.all_gt[['ID', 'LVEDV_automatic', 'LVM_automatic']] # For automatic
        # else:
        #     self.all_gt = self.all_gt[['ID', 'LVEDV', 'LVM']] # For manual

        # load preds
        self.all_pred = pd.read_csv(self.dir_preds)
        print('Number of subjects: ' + str(len(self.all_pred)))

        # Taking only the ones for testing
        self.all_gt = self.all_gt[self.all_gt.ID.isin(self.all_pred.ID.values)]

        # organising gt and preds
        self.gt = self.all_gt.sort_values(by=['ID']).set_index('ID')
        self.pred = self.all_pred.sort_values(by=['ID']).set_index('ID')

    def compute_mae(self):

        for col in range(len(self.pred.columns)):
            err = np.absolute((np.array(self.gt.values[:,col], dtype=np.float64) - np.array(self.pred.values[:,col], dtype=np.float64)))
            m_err = np.mean(err)
            std_err = np.std(err)
            print('Mean error of ' + self.pred.columns[col]  + ' ' +
                  str(round(m_err, 2)) + ' +/- ' + str(round(std_err,2)))


    def bland_altman_plot(self, predicted, truth, filename):

        diff = predicted - truth
        avg =  np.mean([predicted, truth], axis=0)

        fig, ax = plt.subplots()
        im = ax.scatter(avg, diff, s=20, alpha=0.6)
        # Computing mean and Stds
        if filename.split('/')[-1].split('_')[-3] == 'LVEDV':
            ax.axhline(y=np.mean(diff), linewidth=2, color='r', ls='--', label='Bias: ' + str(round(np.mean(diff),2)) + ' ml')
        if filename.split('/')[-1].split('_')[-3] == 'LVM':
            ax.axhline(y=np.mean(diff), linewidth=2, color='r', ls='--', label='Bias: ' + str(round(np.mean(diff),2)) + ' gr')

        LoA1 = round(np.mean(diff) + 1.96*np.std(diff),2)
        LoA2 = round(np.mean(diff)- 1.96*np.std(diff),2)

        ax.axhline(y=LoA1, linewidth=2, color='b', ls='-.', label='1.96SD: ' + str(LoA1))
        ax.axhline(y=LoA2, linewidth=2, color='b', ls='-.', label='-1.96SD: ' + str(LoA2))
        # fig.colorbar(im, ax=ax)
        ax.set_xlabel('Mean', fontsize=15)
        ax.set_ylabel('Difference [Predicted - GT]', fontsize=15)
        ax.set_title("Bland-Altman Plot - " + filename.split('/')[-1].split('_')[-3], fontsize=15)
        ax.grid()
        ax.legend()

        # Plot a horizontal line at 0
        ax.axhline(0, ls=":", c=".2")

        fig.savefig(filename, bbox_inches='tight')

        plt.clf()

        return ax

    def plot_BA(self):
        for col in range(len(self.pred.columns)):
            self.bland_altman_plot(self.pred.values[:,col],
                                   self.gt.values[:,col],
                                   self.dir_plots + self.pred.columns[col] + "_BA_plot.pdf")
            # Saving plot in PNG
            self.bland_altman_plot(self.pred.values[:,col],
                                   self.gt.values[:,col],
                                   self.dir_plots + self.pred.columns[col] + "_BA_plot.png")


    def plot_corr(self):

        for col in range(len(self.pred.columns)):

            if 'LVEDV' in self.pred.columns[col]:
                max_axis_val = 300 # This is for the max X and Y axis
            else: # for LVM
                max_axis_val = 200 # This is for the max X and Y axis

            # Creating linear equation
            coef = np.polyfit(self.pred.values[:,col],self.gt.values[:,col],1)
            poly1d_fn = np.poly1d(coef)
            print(poly1d_fn)
            new_x_axis = np.concatenate((np.arange(np.min(self.pred.values[:,col])),
                                         self.pred.values[:,col],
                                         np.arange(np.max(self.pred.values[:,col]), max_axis_val)))
            if poly1d_fn[0] > 0:
                plt.plot(new_x_axis, poly1d_fn(new_x_axis), '--r', label='y = ' + str(round(poly1d_fn[1],1)) + ' x +' + str(round(poly1d_fn[0],1)))
            else:
                plt.plot(new_x_axis, poly1d_fn(new_x_axis), '--r', label='y = ' + str(round(poly1d_fn[1],1)) + ' x ' + str(round(poly1d_fn[0],1)))
            plt.rcParams["font.size"] = "15"
            plt.plot(self.gt.values[:,col], self.pred.values[:,col], 'o', alpha=0.6)
            plt.title(self.pred.columns[col] + ' (r=' + str(self.p_coeffs[self.pred.columns[col]]) +')', fontsize=15)
            # Diagonal line
            l = mlines.Line2D([0, max_axis_val], [0, max_axis_val], color='green', linewidth=0.5)
            ax = plt.gca()
            ax.add_line(l)
            ax.legend()
            plt.xlabel('Ground Truth', fontsize=15)
            plt.ylabel('Predicted', fontsize=15)
            plt.axis('square')
            plt.xlim(0, max_axis_val)
            plt.ylim(0, max_axis_val)
            plt.grid()
            plt.savefig(self.dir_plots + self.pred.columns[col] + "_corr.pdf", bbox_inches='tight')
            plt.savefig(self.dir_plots + self.pred.columns[col] + "_corr.png", bbox_inches='tight')
            plt.clf()

    def pearson_coff(self):
        # Pearson coefficient
        self.p_coeffs = {}
        for col in range(len(self.pred.columns)):
            coeff, p_val = sc.stats.pearsonr(np.array(self.pred.values[:,col], dtype=np.float64),
                                             np.array(self.gt.values[:,col], dtype=np.float64))
            self.p_coeffs[self.pred.columns[col]] = round(coeff,2)
            print('Pearson coeff for ' + self.pred.columns[col]  + ' ' + str(round(coeff,2)))

    def plot_dist(self):
        for col in range(len(self.pred.columns)):
            mutual_info = metrics.normalized_mutual_info_score(self.gt.values[:,col], self.pred.values[:,col])
            print('Mutual information ' + self.pred.columns[col] + ': ' +  str(round(mutual_info, 2)))
            ax = sns.distplot(self.pred.values[:,col], color="orange", hist=False, label='Predicted')
            ax = sns.distplot(self.gt.values[:,col], color="blue", hist=False, label='GT')
            ax.set_ylabel('Normalised probability', fontsize=15)
            if self.pred.columns[col] == 'LVEDV':
                ax.set_xlabel('ml', fontsize=15)
            if self.pred.columns[col] == 'LVM':
                ax.set_xlabel('gr', fontsize=15)
            ax.grid(True)
            plt.savefig(self.dir_plots + self.pred.columns[col] + "_dis.pdf", bbox_inches='tight')
            plt.savefig(self.dir_plots + self.pred.columns[col] + "_dis.png", bbox_inches='tight')
            plt.clf()
