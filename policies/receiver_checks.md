# Receiver Account Checks

Sample policy written for this project.

## FP-05: Receiver balance does not show the incoming money
If the amount left the sender but the receiver's balance did not go up by the same amount, the records do not add up. The analyst must log the mismatch and ask the receiving bank or merchant to confirm that the money arrived. Treat the transaction as suspicious until the mismatch is explained.

## FP-06: Money sent to an empty or new receiver account
A receiver account that had a zero balance before a large incoming payment is often newly opened for fraud. Check how old the receiver account is and whether it has received money from other flagged accounts. If it is less than 30 days old, escalate to fraud operations.
