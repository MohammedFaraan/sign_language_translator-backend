import cv2
import os
import time

# Create directory for videos
data_root = os.path.join("data", "videos")
os.makedirs(data_root, exist_ok=True)

cap = cv2.VideoCapture(0)

try:
    while True:
        label = input("Enter sign label to record video (or type 'q' to quit): ").strip()
        if label.lower() == 'q':
            break

        # Setup label directory
        label_dir = os.path.join(data_root, label)
        os.makedirs(label_dir, exist_ok=True)

        print(f"\nReady to record sign '{label}'.")
        print("Press 's' to start recording, 'e' to stop, 'q' to quit.")

        # Wait for 's' to start
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.putText(frame, f"Sign: {label} | Press 's' to start", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Video Collection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                break
            elif key == ord('q'):
                raise KeyboardInterrupt

        # Recording parameters
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        timestamp = int(time.time())
        orig_filename = os.path.join(label_dir, f"{label}_{timestamp}.avi")
        frame_height, frame_width = frame.shape[:2]
        fps = 20.0

        # Prepare memory buffer for frames
        frames = []
        out = cv2.VideoWriter(orig_filename, fourcc, fps, (frame_width, frame_height))

        print(f"Recording... Press 'e' to end recording of '{label}'.")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            out.write(frame)
            # store a copy in memory
            frames.append(frame.copy())
            cv2.putText(frame, f"Recording {label}... Press 'e' to stop", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Video Collection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('e'):
                break
            elif key == ord('q'):
                out.release()
                raise KeyboardInterrupt

        # Finalize original video
        out.release()
        print(f"Saved original video at: {orig_filename}")
        cv2.destroyWindow("Video Collection")

        # Give OS time to flush file
        time.sleep(0.5)

        # Create duplicates from memory buffer
        print("Creating 10 duplicates from memory buffer...")
        for i in range(1, 11):
            dup_filename = os.path.join(label_dir, f"{label}_{timestamp}_copy{i}.avi")
            writer = cv2.VideoWriter(dup_filename, fourcc, fps, (frame_width, frame_height))
            for frm in frames:
                writer.write(frm)
            writer.release()
            print(f"Created duplicate: {dup_filename}")

except KeyboardInterrupt:
    print("Exiting video collection.")
finally:
    cap.release()
    cv2.destroyAllWindows()
