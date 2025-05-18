# Fraud-Detection

The project involves developing and evaluating multiple machine learning models to detect fraudulent transactions. The dataset includes features such as transaction type, amount, and old/new balances, which are used to train the models for fraud detection.

## Data columns
- **Step:** Time step when the transaction was recorded.
- **Type:** Type of transaction (CASH_OUT, PAYMENT, TRANSFER, etc.).
- **Amount:** Transaction amount.
- **OldbalanceOrg:** Initial balance of the sender before the transaction.
- **NewbalanceOrig:** Balance of the sender after the transaction.
- **OldbalanceDest:** Initial balance of the recipient before the transaction.
- **NewbalanceDest:** Balance of the recipient after the transaction.
- **isFraud:** Target variable (1 for fraudulent, 0 for non-fraudulent).
