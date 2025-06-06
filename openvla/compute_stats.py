from scipy import stats

# Example lists of scores
list1 = [-0.26573, -0.14225, -0.13898, -0.03035]
list2 = [12.7, 24.6, 30.3, 32.1]

# Compute the Pearson correlation coefficient and the p-value
corr_coefficient, p_value = stats.pearsonr(list1, list2)

print("Correlation coefficient:", corr_coefficient)
print("P-value:", p_value)