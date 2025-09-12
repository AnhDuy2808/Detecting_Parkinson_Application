import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import os
from tqdm import tqdm
import random

def split_audio(file_path, output_dir, segment_duration=5):
    """
    Split audio file into segments of exactly 5 seconds.
    Returns a list of created segment file paths.
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        duration = len(y) / sr
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        segments = []
        start_time = 0
        
        while start_time < duration:
            # Set fixed segment duration to 5 seconds
            end_time = min(start_time + segment_duration, duration)
            
            # Ensure the last segment is at least 3 seconds
            if end_time - start_time < 3:
                break
                
            # Extract audio segment
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = y[start_sample:end_sample]
            
            # Save audio segment
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_file = os.path.join(output_dir, f"{base_name}_segment_{start_time:.1f}_{end_time:.1f}.wav")
            sf.write(output_file, segment_audio, sr)
            
            segments.append(output_file)
            
            start_time = end_time
            
        return segments
        
    except Exception as e:
        print(f"Error splitting file {file_path}: {e}")
        return []

def main():
    # Read dataset file
    csv_file = "./data/dataset.csv"
    df = pd.read_csv(csv_file)
    
    # Filter for HC and PD groups
    df_filtered = df[df['group'].isin(['HC', 'PD'])]
    
    # Create directory for audio segments
    output_dir = "segmented_audio"
    os.makedirs(output_dir, exist_ok=True)
    
    # List to store segment information
    segment_info = []
    
    print("🔍 Starting audio splitting...")
    
    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
        # Assume audio file name matches record with .wav extension
        file_name = f"{row['record']}.wav"
        file_path = os.path.join("audio_files", file_name)  # Adjust path as needed
        
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue
        
        # Split audio into segments
        segments = split_audio(file_path, output_dir)
        
        # Add segment info to list
        for segment_file in segments:
            segment_info.append({
                'id': os.path.basename(segment_file),
                'label': 0 if row['group'] == 'HC' else 1  # HC=0, PD=1
            })
    
    # Save segment info to CSV
    df_segments = pd.DataFrame(segment_info)
    df_segments.to_csv("./data/segment_info.csv", index=False)
    
    print("✅ Saved segment info to segment_info.csv")
    
    # Print statistics
    print(f"\n📊 Statistics:")
    print(f"Total segments: {len(segment_info)}")
    print(f"HC segments (0): {len(df_segments[df_segments['label'] == 0])}")
    print(f"PD segments (1): {len(df_segments[df_segments['label'] == 1])}")
    print(f"Segment directory: {output_dir}/")

if __name__ == "__main__":
    main()