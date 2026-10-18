# Routine Activity and False Alarms

Sample policy written for this project.

## FP-07: Routine low risk transactions
PAYMENT, DEBIT and CASH_IN transactions, and small transfers where the sender keeps most of their balance, are normal everyday activity. If the model approves them, no manual action is needed and the customer must not be contacted. Unnecessary calls hurt customer experience.

## FP-08: Releasing a hold after the customer confirms
If the customer confirms on a call-back that they made the transaction themselves, release the hold within 1 hour. Mark the case as a false alarm so it can be used when the model is retrained.
