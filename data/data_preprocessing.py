import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    X = []
    y = []
    # Map conditions to labels
    label_dict = {'Normal Condition': 0, 'Misalignment': 1, 'Unbalance': 2, 'Looseness': 3}

    print(f"Loading data from: {data_dir}")

    # Walk through each category directory
    for category in sorted(os.listdir(data_dir)):
        category_path = os.path.join(data_dir, category)
        if os.path.isdir(category_path):
            print(f"Processing category: '{category}'")

            # Extract condition from category name
            condition = None
            for cond in label_dict.keys():
                if cond in category:
                    condition = cond
                    break
            if condition is None:
                print(f"Could not determine condition for category '{category}'")
                continue

            label = label_dict[condition]

            # Navigate into the subdirectory with the same name
            subcategory_path = os.path.join(category_path, category)
            if os.path.isdir(subcategory_path):
                # List all .npy files in the subcategory directory
                npy_files = [f for f in os.listdir(subcategory_path) if f.endswith('.npy')]
                print(f"Found {len(npy_files)} .npy files in '{subcategory_path}'")

                for file_name in npy_files:
                    file_path = os.path.join(subcategory_path, file_name)
                    try:
                        data = np.load(file_path)
                        X.append(data)
                        y.append(label)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
            else:
                print(f"Subdirectory '{category}' not found in '{category_path}'")
                continue

    X = np.array(X)
    y = np.array(y)
    print(f"Total samples loaded: {len(X)}")
    return X, y, label_dict

def main():
    data_dir = '/home/dapgrad/tenzinl2/fault_analysis/Mechanical faults in rotating machinery dataset (normal, unbalance, misalignment, looseness)'

    X, y, label_dict = load_data(data_dir)

    if len(X) == 0:
        print("No data loaded. Please check the data directory and files.")
        return

    # Split the dataset into training and test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Save the datasets
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

    # Save the label dictionary for future reference
    np.save('label_dict.npy', label_dict)

    print("Data has been successfully split and saved.")
    print("Label mapping:")
    for condition, label_id in label_dict.items():
        print(f"  {label_id}: {condition}")

if __name__ == '__main__':
    main()

