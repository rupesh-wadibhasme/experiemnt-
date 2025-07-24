import pickle

# Step 1: Load frequency stats (call this once globally or pass into inference)
with open('combo_frequency_stats.pkl', 'rb') as f:
    freq_stats = pickle.load(f)

# Step 2: Extract the actual combination from input
combo_keys = ['BUnit', 'Cpty', 'BuyCurr', 'SellCurr']
actual_combo = tuple(features.loc[0, k] for k in combo_keys)

# Step 3: Identify least frequent combinations from training (bottom 2)
sorted_combos = sorted(freq_stats.items(), key=lambda x: x[1])
least_freq_combos = set([k for k, _ in sorted_combos[:2]])

# Step 4: Check if current combo is among least frequent
if actual_combo in least_freq_combos:
    reason_bits.append(
        f"The combination of Business Unit '{actual_combo[0]}', Counterparty '{actual_combo[1]}', "
        f"Buy Currency '{actual_combo[2]}' and Sell Currency '{actual_combo[3]}' is one of the least frequently seen in past transactions."
    )

    # Step 5: Compare with model-predicted combination
    predicted_combo = tuple(reconstructed_df.loc[0, k] for k in combo_keys)

    # Step 6: Identify differences
    differing_parts = [
        f"{key}: expected '{actual}', predicted '{predicted}'"
        for key, actual, predicted in zip(combo_keys, actual_combo, predicted_combo)
        if actual != predicted
    ]

    if differing_parts:
        diffs_str = ', '.join(differing_parts)
        reason_bits.append(
            f"Based on similar historical data, the model expects a different setup for: {diffs_str}."
        )
