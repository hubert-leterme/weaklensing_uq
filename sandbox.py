# Calculate mean and standard deviation for each statistical series
means_predinterv_ks = [np.mean(predinterv) for predinterv in list_of_predinterv_ks_test]
stds_predinterv_ks = [np.std(predinterv) for predinterv in list_of_predinterv_ks_test]

means_predinterv_wiener = [np.mean(predinterv) for predinterv in list_of_predinterv_wiener_test]
stds_predinterv_wiener = [np.std(predinterv) for predinterv in list_of_predinterv_wiener_test]

means_predinterv_mcalens = [np.mean(predinterv) for predinterv in list_of_predinterv_mcalens_test]
stds_predinterv_mcalens = [np.std(predinterv) for predinterv in list_of_predinterv_mcalens_test]