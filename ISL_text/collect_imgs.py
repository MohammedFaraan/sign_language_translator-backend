import os
import cv2
import shutil

# Directory to store images
DATA_IMG_DIR = os.path.join("data", "images")
os.makedirs(DATA_IMG_DIR, exist_ok=True)

def create_label():
    # Creates a new label directory to store images.
    label_name = input("Enter new label name: ").strip()
    label_dir = os.path.join(DATA_IMG_DIR, label_name)
    os.makedirs(label_dir, exist_ok=True)
    print(f'New label "{label_name}" created successfully.')
    return label_name, label_dir

def update_label():
    # Allows users to add images to an existing label without deleting previous ones.
    existing_labels = [d for d in os.listdir(DATA_IMG_DIR) if os.path.isdir(os.path.join(DATA_IMG_DIR, d))]
    if not existing_labels:
        print("No existing labels found! Please create a new label first.")
        return None, None
    print(f"Available labels: {existing_labels}")
    label_name = input("Enter the label to update: ").strip()
    label_dir = os.path.join(DATA_IMG_DIR, label_name)
    if not os.path.exists(label_dir):
        print("Invalid label name! Please try again.")
        return None, None
    print(f'Adding images to label "{label_name}"...')
    return label_name, label_dir

def overwrite_label():
    # Overwrites an existing label by deleting all its images before adding new ones.
    existing_labels = [d for d in os.listdir(DATA_IMG_DIR) if os.path.isdir(os.path.join(DATA_IMG_DIR, d))]
    if not existing_labels:
        print("No existing labels found! Please create a new label first.")
        return None, None
    print(f"Available labels: {existing_labels}")
    label_name = input("Enter the label to overwrite: ").strip()
    label_dir = os.path.join(DATA_IMG_DIR, label_name)
    if not os.path.exists(label_dir):
        print("Invalid label name! Please try again.")
        return None, None
    for file in os.listdir(label_dir):
        file_path = os.path.join(label_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print(f'Label "{label_name}" has been reset successfully.')
    return label_name, label_dir

def delete_all_labels():
    # Deletes all label directories and resets the dataset.
    confirm = input("Are you sure you want to delete ALL labels? Type 'yes' to confirm: ").strip().lower()
    if confirm == 'yes':
        shutil.rmtree(DATA_IMG_DIR)
        os.makedirs(DATA_IMG_DIR, exist_ok=True)
        print("All labels have been deleted successfully.")
    else:
        print("Deletion cancelled.")

def delete_one_label():
    # Deletes a specific label directory.
    existing_labels = [d for d in os.listdir(DATA_IMG_DIR) if os.path.isdir(os.path.join(DATA_IMG_DIR, d))]
    if not existing_labels:
        print("No existing labels found! Nothing to delete.")
        return
    print(f"Available labels: {existing_labels}")
    label_name = input("Enter the label to delete: ").strip()
    label_dir = os.path.join(DATA_IMG_DIR, label_name)
    if not os.path.exists(label_dir):
        print("Invalid label name! Please try again.")
        return
    confirm = input(f"Are you sure you want to delete label '{label_name}'? Type 'yes' to confirm: ").strip().lower()
    if confirm == 'yes':
        shutil.rmtree(label_dir)
        print(f'Label "{label_name}" has been deleted successfully.')
    else:
        print("Deletion cancelled.")

def capture_images(label_name, label_dir):
    # Capture images for the given label using the webcam.
    try:
        dataset_size = int(input("Enter the number of images to capture: "))
    except ValueError:
        print("Invalid number entered. Returning to main menu.")
        return

    cap = cv2.VideoCapture(0)
    print("Press 'E' to start capturing images.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting capture mode.")
            cap.release()
            cv2.destroyAllWindows()
            return
        cv2.putText(frame, 'Press E to Start', (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('e'):
            break

    # Determine the starting index for image filenames
    existing_images = [int(f.split('.')[0]) for f in os.listdir(label_dir)
                       if f.endswith('.jpg') and f.split('.')[0].isdigit()]
    start_index = max(existing_images) + 1 if existing_images else 0
    counter = start_index

    print(f"Capturing {dataset_size} images for label '{label_name}'...")
    for i in range(dataset_size):
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame, skipping image.")
            continue
        # Optionally resize the image (to match the first script's behavior)
        frame_resized = cv2.resize(frame, (128, 128))
        img_path = os.path.join(label_dir, f"{counter}.jpg")
        cv2.imwrite(img_path, frame_resized)
        print(f"Captured image {i+1}/{dataset_size} for label '{label_name}'...")
        counter += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished capturing images for label '{label_name}'.")

def main_menu():
    while True:
        print("\n=== Image Collection Menu ===")
        print("1. Create a new label")
        print("2. Add images to an existing label")
        print("3. Overwrite an existing label")
        print("4. Delete all labels")
        print("5. Delete a specific label")
        print("6. Exit")
        choice = input("Enter your choice: ").strip()
        if choice == '1':
            label_name, label_dir = create_label()
        elif choice == '2':
            label_name, label_dir = update_label()
        elif choice == '3':
            label_name, label_dir = overwrite_label()
        elif choice == '4':
            delete_all_labels()
            continue
        elif choice == '5':
            delete_one_label()
            continue
        elif choice == '6':
            print("Exiting program. Have a great day!")
            break
        else:
            print("Invalid choice! Please enter a valid option.")
            continue

        if not label_name or not label_dir:
            continue

        capture_images(label_name, label_dir)

print("Welcome to the Image Collection Program!")
main_menu()
