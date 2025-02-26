import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for Matplotlib

# Load the CSV file
data = pd.read_csv("bank.csv")

# Preprocessing and Bucketing Functions
def bucket_age(age):
    if age < 30:
        return '<30'
    elif age <= 40:
        return '30-40'
    elif age <= 50:
        return '40-50'
    elif age <= 70:
        return '50-70'
    else:
        return '>70'

def bucket_balance(balance):
    if balance < 0:
        return '<0'
    elif balance <= 1000:
        return '0-1000'
    elif balance <= 2000:
        return '1001-2000'
    elif balance <= 3450:
        return '2001-3450'
    else:
        return '>3450'

def bucket_duration(duration):
    if duration <= 120:
        return '1-120'
    elif duration <= 300:
        return '121-300'
    elif duration <= 640:
        return '301-640'
    else:
        return '>640'

def bucket_campaign(campaign):
    if campaign == 1:
        return '1'
    elif campaign <= 3:
        return '2-3'
    elif campaign <= 6:
        return '4-6'
    else:
        return '>6'

def bucket_pdays(pdays):
    if pdays == -1:
        return '-1'
    elif pdays <= 90:
        return '1-90'
    elif pdays <= 180:
        return '91-180'
    elif pdays <= 620:
        return '181-620'
    else:
        return '>620'

def bucket_previous(previous):
    if previous == 0:
        return '0'
    elif previous <= 3:
        return '1-3'
    elif previous <= 6:
        return '4-6'
    else:
        return '>6'

def bucket_day(day):
    if 1 <= day <= 7:
        return "Week 1"
    elif 8 <= day <= 14:
        return "Week 2"
    elif 15 <= day <= 21:
        return "Week 3"
    else:
        return "Week 4"

def bucket_month(month):
    if month in ["jan", "feb", "mar"]:
        return "Q1"
    elif month in ["apr", "may", "jun"]:
        return "Q2"
    elif month in ["jul", "aug", "sep"]:
        return "Q3"
    else:
        return "Q4"

def bucket_job(job):
    if job in ["admin.", "management"]:
        return "White-collar"
    elif job in ["blue-collar", "services", "housemaid", "technician"]:
        return "Blue-collar"
    elif job in ["entrepreneur", "self-employed"]:
        return "Self-employed/Business"
    elif job in ["student", "retired", "unemployed"]:
        return "Not in Workforce"
    else:
        return "Unknown"

def preprocess_row(row):
    row['age'] = bucket_age(row['age'])
    row['balance'] = bucket_balance(row['balance'])
    row['duration'] = bucket_duration(row['duration'])
    row['campaign'] = bucket_campaign(row['campaign'])
    row['pdays'] = bucket_pdays(row['pdays'])
    row['previous'] = bucket_previous(row['previous'])
    row['job'] = bucket_job(row['job'])
    row['day'] = bucket_day(row['day'])
    row['month'] = bucket_month(row['month'])
    return row

# Apply preprocessing
data_processed = data.apply(preprocess_row, axis=1)

# Calculate entropy
def calculate_entropy(data, target_column):
    counts = data[target_column].value_counts()
    total = len(data)
    return -sum((count / total) * np.log2(count / total) for count in counts if count > 0)

# χ2 Test Implementation
def chi_square_test(data, feature, target_column):
    contingency_table = {}
    for value in data[feature].unique():
        subset = data[data[feature] == value]
        counts = subset[target_column].value_counts()
        contingency_table[value] = counts.to_dict()

    row_totals = {k: sum(v.values()) for k, v in contingency_table.items()}
    column_totals = {}
    total = sum(row_totals.values())

    for row in contingency_table.values():
        for label, count in row.items():
            column_totals[label] = column_totals.get(label, 0) + count

    chi_square_stat = 0
    for row_value, row in contingency_table.items():
        for label, observed in row.items():
            expected = (row_totals[row_value] * column_totals[label]) / total
            if expected > 0:
                chi_square_stat += ((observed - expected) ** 2) / expected

    degrees_of_freedom = (len(row_totals) - 1) * (len(column_totals) - 1)
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(chi_square_stat, degrees_of_freedom)
    return chi_square_stat, p_value

# Build the decision tree
def build_tree(ratio):
    data_sampled = data_processed.sample(frac=ratio, random_state=42)

    def recursive_tree(data, target_column='y', alpha=0.05):
        if len(data[target_column].unique()) == 1 or len(data) == 0:
            return data[target_column].mode()[0] if len(data) > 0 else None

        current_entropy = calculate_entropy(data, target_column)
        best_feature = None
        best_gain = 0

        for feature in data.columns:
            if feature == target_column:
                continue
            split_entropy = sum(
                (len(subset) / len(data)) * calculate_entropy(subset, target_column)
                for _, subset in data.groupby(feature)
            )
            info_gain = current_entropy - split_entropy
            if info_gain > best_gain:
                best_gain = info_gain
                best_feature = feature

        if not best_feature:
            return data[target_column].mode()[0]

        _, p_value = chi_square_test(data, best_feature, target_column)
        if p_value > alpha:
            return data[target_column].mode()[0]

        tree = {best_feature: {}}
        for value, subset in data.groupby(best_feature):
            tree[best_feature][value] = recursive_tree(subset)
        return tree

    return recursive_tree(data_sampled)

global_decision_tree = build_tree(0.65)


# Tree error with cross-validation
def tree_error(k):
    # Split data into k folds
    folds = np.array_split(data_processed, k)
    errors = []

    # Iterate through each fold
    for i in range(k):
        # Separate training and validation data
        validation_data = folds[i]
        train_data = pd.concat(folds[:i] + folds[i + 1:])

        # Build the decision tree using the training data
        decision_tree = build_tree(1)  # Build tree on the entire training set

        # Make predictions on the validation set
        validation_data['prediction'] = validation_data.apply(
            lambda row: validate_tree(decision_tree, row), axis=1
        )

        # Calculate error rate for the current fold
        error_rate = (validation_data['prediction'] != validation_data['y']).mean()
        errors.append(error_rate)

    # Report the average error rate
    average_error = np.mean(errors)
    print(f"Average Cross-Validation Error Rate (k={k}): {average_error:.2%}")
    return average_error


# Validate tree for a single input
def validate_tree(tree, row):
    if not isinstance(tree, dict):
        return tree

    feature = next(iter(tree))
    value = row[feature]

    if value in tree[feature]:
        return validate_tree(tree[feature][value], row)
    else:
        return None

def preprocess_input(row_input):
    row = {
        'age': bucket_age(row_input[0]),
        'job': bucket_job(row_input[1]),
        'marital': row_input[2],
        'education': row_input[3],
        'default': row_input[4],
        'balance': bucket_balance(row_input[5]),
        'housing': row_input[6],
        'loan': row_input[7],
        'contact': row_input[8],
        'day': bucket_day(row_input[9]),
        'month': bucket_month(row_input[10]),
        'duration': bucket_duration(row_input[11]),
        'campaign': bucket_campaign(row_input[12]),
        'pdays': bucket_pdays(row_input[13]),
        'previous': bucket_previous(row_input[14]),
        'poutcome': row_input[15]
    }
    return row


# Predict deposit
def will_open_deposit(row_input):
    # Convert the row_input list into a pandas Series with column names
    input_row = pd.Series(row_input, index=data_processed.columns[:-1])  # Exclude the target column 'y'

    # Use the validate_tree function to predict
    prediction = validate_tree(global_decision_tree, input_row)
    preprocessed_row = preprocess_input(row_input)  # Convert raw input to bucketed
    return 1 if validate_tree(global_decision_tree, preprocessed_row) == 'yes' else 0

# Matplotlib Tree Visualization
def print_tree(tree, depth=0, max_graph_depth=None, label='', layer=1):
    # Stop printing nodes if max depth is reached
    if max_graph_depth is not None and depth >= max_graph_depth:
        print(f"{layer}. " + "    " * depth + f"|-- {label}: ... (depth limit reached)")
        return

    # Check if we are at a leaf node
    if not isinstance(tree, dict):
        print(f"{layer}. " + "    " * depth + f"|-- {label}: Prediction -> {tree}")
        return

    # Get the splitting feature
    feature = next(iter(tree))

    # Print the current decision node
    if depth == 0:
        print(f"[Root: {feature}]")
    else:
        print(f"{layer}. " + "    " * depth + f"|-- {label}: [Split on {feature}]")

    # Recursively print the child nodes
    for i, (value, subtree) in enumerate(tree[feature].items(), start=1):
        print_tree(subtree, depth + 1, max_graph_depth=max_graph_depth, label=value, layer=layer + 1)

