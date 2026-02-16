"""Quick test of signal extraction functions."""

import cv2
import numpy as np
import librosa
import subprocess
import os

video_path = "videos/Everything_Wrong_With_Tekken_8"

print("=" * 70)
print("Testing Scene Cuts Extraction")
print("=" * 70)

try:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Could not open video")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"✓ Video opened: {fps} FPS, {total_frames} frames")

        # Read a few frames
        cut_count = 0
        prev_hist = None
        frame_idx = 0
        sample_rate = 2

        while frame_idx < 100:  # Test first 100 frames
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()

                if prev_hist is not None:
                    distance = cv2.compareHist(
                        prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA
                    )
                    if distance > 0.4:
                        cut_count += 1
                    if frame_idx < 20:
                        print(
                            f"  Frame {frame_idx}: histogram distance = {distance:.4f}"
                        )

                prev_hist = hist

            frame_idx += 1

        cap.release()
        print(f"✓ Scene cuts detected in first 100 frames: {cut_count}")

except Exception as e:
    print(f"✗ ERROR: {e}")

print("\n" + "=" * 70)
print("Testing Motion Intensity Extraction")
print("=" * 70)

try:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Could not open video")
    else:
        motion_deltas = []
        prev_frame = None
        frame_idx = 0
        sample_rate = 5

        while frame_idx < 100:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                frame = cv2.resize(frame, (320, 180))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_frame is not None:
                    mad = cv2.absdiff(prev_frame, gray).mean()
                    motion_deltas.append(mad)
                    if frame_idx < 30:
                        print(f"  Frame {frame_idx}: MAD = {mad:.2f}")

                prev_frame = gray

            frame_idx += 1

        cap.release()
        if motion_deltas:
            avg_motion = np.mean(motion_deltas)
            normalized = min(100, (avg_motion / 50) * 100)
            print(f"✓ Average motion intensity: {normalized:.2f}/100")
        else:
            print("✗ No motion deltas computed")

except Exception as e:
    print(f"✗ ERROR: {e}")

print("\n" + "=" * 70)
print("Testing Color Variance Extraction")
print("=" * 70)

try:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Could not open video")
    else:
        color_variances = []
        frame_idx = 0
        sample_rate = 5

        while frame_idx < 100:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                frame = cv2.resize(frame, (320, 180))
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

                h_var = np.var(hsv[:, :, 0])
                s_var = np.var(hsv[:, :, 1])
                v_var = np.var(hsv[:, :, 2])

                avg_variance = (h_var + s_var + v_var) / 3
                color_variances.append(avg_variance)
                if frame_idx < 30:
                    print(f"  Frame {frame_idx}: avg color var = {avg_variance:.2f}")

            frame_idx += 1

        cap.release()
        if color_variances:
            avg_color_var = np.mean(color_variances)
            normalized = min(100, (avg_color_var / 2000) * 100)
            print(f"✓ Average color variance: {normalized:.2f}/100")
        else:
            print("✗ No color variances computed")

except Exception as e:
    print(f"✗ ERROR: {e}")

print("\n" + "=" * 70)
print("Testing Audio Extraction")
print("=" * 70)

try:
    audio_path = "/tmp/test_audio.wav"
    result = subprocess.run(
        ["ffmpeg", "-i", video_path, "-q:a", "9", "-y", audio_path],
        capture_output=True,
        timeout=30,
    )

    print(f"ffmpeg return code: {result.returncode}")
    print(f"Audio file exists: {os.path.exists(audio_path)}")
    print(
        f"Audio file size: {os.path.getsize(audio_path) if os.path.exists(audio_path) else 'N/A'}"
    )

    if result.returncode == 0 and os.path.exists(audio_path):
        y, sr = librosa.load(audio_path, sr=None)
        print(f"✓ Audio loaded: {sr} Hz, {len(y)} samples")

        S = librosa.feature.melspectrogram(y=y, sr=sr)
        rms = librosa.feature.rms(S=S)[0]
        print(f"✓ RMS computed: {len(rms)} frames")
        print(
            f"  RMS stats: min={rms.min():.4f}, max={rms.max():.4f}, mean={rms.mean():.4f}"
        )

        # Clean up
        os.remove(audio_path)
    else:
        print("✗ ffmpeg failed or audio not created")
        if result.stderr:
            print(f"stderr: {result.stderr[:500]}")

except Exception as e:
    print(f"✗ ERROR: {e}")
