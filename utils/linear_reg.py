import pandas as pd
import numpy as np
from sklearn import linear_model, svm
import pdb

def linear_reg(laten_vars, labels_4_linear_reg, img_names_4_linear_reg, file_name, args):

    ## Saving latent variables and computing linear regression
    results_lin_reg = pd.DataFrame(data=np.array(laten_vars), columns=np.arange(len(laten_vars[0])))
    results_lin_reg.insert(loc=0, column='ID', value=img_names_4_linear_reg)
    results_lin_reg.insert(loc=1, column='LVEDV', value=np.array(labels_4_linear_reg)[:,0])
    results_lin_reg.insert(loc=2, column='LVM', value=np.array(labels_4_linear_reg)[:,1])
    results_lin_reg.to_csv(args.dir_results + args.dir_test_ids + file_name + '_latent.csv', index=False)

    test_set = pd.read_csv(args.dir_results + args.dir_test_ids + 'test_set.csv', sep=',')
    train_set = pd.read_csv(args.dir_results + args.dir_test_ids + 'train_set.csv', sep=',')

    test_results_lin_reg = results_lin_reg[results_lin_reg.ID.isin(test_set.ID.values)]
    train_results_lin_reg = results_lin_reg[results_lin_reg.ID.isin(train_set.ID.values)]

    lm = linear_model.LinearRegression()

    labels = train_results_lin_reg[['LVEDV', 'LVM']]

    model = lm.fit(train_results_lin_reg.iloc[:,3:], labels)
    predictions = lm.predict(test_results_lin_reg.iloc[:,3:])

    # pdb.set_trace()

    preds = pd.DataFrame({'ID': test_results_lin_reg.ID.values, 'LVEDV': predictions[:,0], 'LVM': predictions[:,1]})
    preds.to_csv(args.dir_results + args.dir_test_ids + 'preds.csv', index=False)
