import cv2
import os

def capture_faces(name):
    # Ensure data directory exists
    if not os.path.exists("data"):
        os.mkdir("data")
    
    # Directory to save images
    save_dir = f"data/{name}"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    # Start video capture
    cap = cv2.VideoCapture(0)

    # Check if the video is opened successfully
    if not cap.isOpened():
        print("Could not open webcam!")
        return

    img_count = 0

    while True:
        # Read frame from webcam
        ret, frame = cap.read()

        if not ret:
            break

        # Display the frame
        cv2.imshow('Press "c" to capture, "q" to quit', frame)

        key = cv2.waitKey(1) & 0xFF

        # Capture the frame when 'c' key is pressed
        if key == ord("c"):
            # Ensure unique filename
            img_path = os.path.join(save_dir, f"{name}_{img_count}.png")
            while os.path.exists(img_path):
                img_count += 1
                img_path = os.path.join(save_dir, f"{name}_{img_count}.png")
            
            cv2.imwrite(img_path, frame)
            print(f"Saved: {img_path}")
            img_count += 1

        # Exit when 'q' key is pressed
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    name = input("Enter the name for the dataset: ")
    capture_faces(name)
