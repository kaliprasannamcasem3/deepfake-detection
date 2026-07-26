import cv2
import numpy as np
import tensorflow as tf
import gradio as gr

# Load pre-trained model
MODEL_PATH = 'deepfake_detector_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

def classify_video(video_path, trained_model, num_frames=15):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return "Error", 0.0
        
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    predictions = []
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            frame_resized = cv2.resize(frame, (224, 224))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img_array = np.expand_dims(frame_rgb, axis=0) / 255.0
            
            pred = trained_model.predict(img_array, verbose=0)
            predictions.append(pred[0][0])
            
    cap.release()
    avg_score = np.mean(predictions)
    verdict = "REAL" if avg_score > 0.5 else "FAKE"
    return verdict, avg_score

def predict_deepfake(video_file):
    if video_file is None:
        return "Please upload a video file."
    
    verdict, score = classify_video(video_file, model)
    confidence = score if verdict == "REAL" else (1 - score)
    
    return f"Verdict: {verdict}\nConfidence: {confidence * 100:.2f}%\n(Raw Score: {score:.4f})"

interface = gr.Interface(
    fn=predict_deepfake,
    inputs=gr.Video(label="Upload a Video (.mp4)"),
    outputs=gr.Textbox(label="Detection Results"),
    title="🕵️ Deepfake Video Detector",
    description="Upload a video to test our CNN model."
)

if __name__ == "__main__":
    interface.launch(share=True)
