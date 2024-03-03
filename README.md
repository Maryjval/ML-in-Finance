## Financial Analysis Using Machine Learning

This repository contains a dataset comprising various financial metrics for a firm. While the dataset offers insights into the financial health and performance of these firm, our primary focus is on predicting investment grade status.

### Questions Addressed by the Dataset:

- **Predicting Investment Grade Status:** Utilizing machine learning algorithms, we aim to predict whether the firm is classified as investment grade based on its financial metrics.

- Financial Health Analysis: While our primary objective is predicting investment grade, the dataset enables comprehensive analysis of the financial health of the firm involved. Metrics such as debt ratios, liquidity ratios, and profitability ratios can provide valuable insights.

- Trend Analysis: We can investigate trends in sales/revenues, gross margin, EBITDA, net income, debt levels, and other metrics over time or across different firms to identify patterns and potential areas for improvement.

- Comparative Analysis: Comparing the financial performance of different entities within the dataset allows us to identify relative strengths, weaknesses, and performance benchmarks.

- Risk Assessment: By analyzing metrics such as leverage ratios, interest coverage ratios, and liquidity ratios, we can assess the risk levels associated with the firm and its financial operations.

- Correlation Analysis: Exploring relationships between different financial metrics can uncover patterns and dependencies that provide insights into business operations and financial performance.

### Focus on Predictive Modeling:

While the dataset offers opportunities for various analyses, our primary objective is building predictive models to determine investment grade status. We will employ machine learning techniques such as logistic regression, ridge regression, and possibly neural networks to achieve this goal.

## Conclusion

Upon analyzing different machine learning approaches for predicting investment grade status, the following observations were made:

- Linear regression with Ridge regularization and Logistic regression with Ridge regularization achieved the highest accuracy scores, around 76.76% and 76.47% respectively.
- Linear regression with Lasso regularization and Logistic regression with Lasso regularization also performed well, with accuracy scores around 75.29% and 76.18% respectively, albeit slightly lower than the Ridge regularization models.
- The neural network approach exhibited significantly lower accuracy, approximately 22.35%, suggesting it might not be the most suitable model without further optimization.

## Recommendations

Based on these findings, it is recommended to use Linear regression or Logistic regression with Ridge regularization for predicting investment grade status due to their high accuracy and reliability. Further optimization of the neural network model could potentially improve its performance.

