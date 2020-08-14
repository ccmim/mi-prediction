
from utils.rescalingClass import scalingClass
from utils.plotClass import plot_results


if __name__ == "__main__":

    # For mcVAE
    res_fol = './results/2020-05-10_19-44-26_automatic_600Epochs_reducedList/results_regressor/2020-05-12_00-28-28_1384_subjets/'
    dir_data = './results/2020-05-10_19-44-26_automatic_600Epochs_reducedList/results_regressor/2020-05-12_00-28-28_1384_subjets/test_set.csv'
    # # Linear regression
    # res_fol = './results/2020-05-13_17-42-01_automatic_1800Epochs_reducedList'
    # dir_data = './results/2020-05-13_17-42-01_automatic_1800Epochs_reducedList/test_set.csv'
    # # For metadata
    # res_fol = './results_only_mtdt/2020-05-24_19-01-07_best'
    # dir_data = './results_only_mtdt/2020-05-24_19-01-07_best/test_set.csv'
    # # For fundus
    # res_fol = './results_only_fundus/2020-05-03_02-19-42_best'
    # dir_data = './results_only_fundus/2020-05-03_02-19-42_best/test_set.csv'
    # predictions
    dir_preds_rescaled = res_fol + '/preds.csv'
    # dir_preds = res_fol + '/preds.csv'

    # # Scaled data
    # dir_scaled_data = './input_data/ids/LVEDV_LVM_manual_IDs_scaled.csv'
    # # Predictions rescaled
    # dir_preds_rescaled = res_fol + '/preds_rescaled.csv'
    # # Scaling fator
    # scaling_factor = 1.0
    # # Save scaled original data
    # save_scaled_ids = False

    ### Scaling/rescaling the predictions ###
    # print('\nRescaling predictions ...')
    # object_scaling = scalingClass(dir_data, dir_scaled_data, scaling_factor, save_scaled_ids)
    # object_scaling.rescaled_preds(dir_preds)
    # del object_scaling

    ### Plotting ###
    print('\nComputing MSE & plotting ...')
    show_results = plot_results(dir_data, dir_preds_rescaled)
    show_results.compute_mae()
    show_results.pearson_coff()
    show_results.plot_corr()
    show_results.plot_BA()
    show_results.plot_dist()
