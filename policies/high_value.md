# High Value Transactions

Sample policy written for this project.

## FP-03: High value transfer above 200,000
Any TRANSFER above 200,000 needs a second check even if the model approves it. If the model approves, the transaction can go through but must appear in the next day's high value review list. If the model blocks it, a senior analyst must sign off before the funds are released.

## FP-04: Large CASH_OUT withdrawal from an account
A large CASH_OUT above 100,000 must be checked against the customer's KYC profile and usual withdrawal pattern. If the amount is far above anything the customer has done before, hold the cash-out and confirm with the customer through a call-back before paying out.
