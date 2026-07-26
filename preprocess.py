import cv2
import os
import glob
import numpy as np

def extract_frames(video_path, output_folder, num_frames=10):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return
        
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    count = 0
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            frame = cv2.resize(frame, (224, 224))
            cv2.imwrite(os.path.join(output_folder, f"frame_{count}.jpg"), frame)
            count += 1
            
    cap.release()

def process_all_videos(sample_limit=50):
    real_videos = glob.glob('deepfake_dataset/**/Celeb-real/*.mp4', recursive=True)[:sample_limit]
    fake_videos = glob.glob('deepfake_dataset/**/Celeb-synthesis/*.mp4', recursive=True)[:sample_limit]

    print(f"Processing {len(real_videos)} real videos...")
    for vid in real_videos:
        vid_name = os.path.basename(vid).split('.')[0]
        extract_frames(vid, f'dataset/real/{vid_name}')

    print(f"Processing {len(fake_videos)} fake videos...")
    for vid in fake_videos:
        vid_name = os.path.basename(vid).split('.')[0]
        extract_frames(vid, f'dataset/fake/{vid_name}')

    print("Preprocessing completed!")

if __name__ == "__main__":
    process_all_videos()
