# Load frequency stats if not already passed
with open("frequency_stats.pkl", "rb") as f:
    freq_stats = pickle.load(f)

# Extract values
bunit     = original_df.at[0, 'BUnit']
cpty      = original_df.at[0, 'Cpty']
buy_curr  = original_df.at[0, 'BuyCurr']
sell_curr = original_df.at[0, 'SellCurr']

# Check frequencies
freq_combo_1 = freq_stats.get((bunit, cpty), 0)
freq_combo_2 = freq_stats.get((bunit, cpty, buy_curr), 0)

reason_notes = []

if freq_combo_1 < 3:  # Adjust threshold as needed
    reason_notes.append(f"This business unit and counterparty pair ({bunit}, {cpty}) is rarely seen in past transactions.")

if freq_combo_2 < 2:
    reason_notes.append(f"The combination of business unit, counterparty, and buy currency ({bunit}, {cpty}, {buy_curr}) is very uncommon.")

# Model-predicted categorical values
predicted_cat_values = reconstructed_df.loc[:, ['BuyCurr', 'SellCurr', 'Cpty']]  # extend as needed
actual_cat_values = features.loc[:, ['BuyCurr', 'SellCurr', 'Cpty']]

for col in predicted_cat_values.columns:
    pred_val = predicted_cat_values[col].values[0]
    actual_val = actual_cat_values[col].values[0]
    if pred_val != actual_val:
        reason_notes.append(
            f"The transaction lists {col} as {actual_val}, but typical patterns suggest {pred_val} is more expected."
        )

# Return `reason_notes` with other anomaly reasoning
