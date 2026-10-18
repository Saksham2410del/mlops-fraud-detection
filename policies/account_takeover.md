# Account Takeover

Sample policy written for this project. Loosely based on common retail banking practice.

## FP-01: Full balance drained by a transfer or cash-out
A TRANSFER or CASH_OUT that moves the entire balance out of the sender's account, leaving it at zero, is the most common sign of account takeover. The transaction must be held. The analyst must call the customer on their registered mobile number within 2 hours to confirm it. If the customer cannot be reached, keep the hold for 24 hours and block further outgoing transfers from the account until the customer is verified.

## FP-02: Money moved out and then cashed out (mule pattern)
Fraudsters usually transfer the stolen balance to a second account and cash it out from there within minutes. When a large TRANSFER is blocked, the receiving account must also be flagged as a possible mule account and reported to the fraud operations team. Do not release the funds until both accounts have been reviewed.
