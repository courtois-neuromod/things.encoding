import subprocess

class VideoSegmenteur:

    def __init__(self, video_path, stimuli_dir):
        self.video_path = video_path
        self.stimuli_dir = stimuli_dir

    def create_segment(self, start, duration, segments_idx):
        """ Pour dupliquer frames, utiliser ces paramètres
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-i", str(self.video_path),
            "-vf", "fps=29.97",  # duplique la frame fixe sur toute la durée
            "-fps_mode", "cfr",  # force un framerate constant
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(self.stimuli_dir / f"stimulus_{segments_idx:04d}.mp4"),
        """
        subprocess.run([
            "ffmpeg",
            "-i", self.video_path,
            "-ss", str(start),
            "-t", str(duration),
            "-c:v", "libx264",
            "-force_key_frames", "expr:gte(t,0)",
            "-y",
            str(self.stimuli_dir / f"stimulus_{segments_idx:04d}.mp4"),
        ], capture_output=True)

    def cut_run(self, timestamps, n_frames):
        segments_idx = 0
        for i in range(n_frames - 1):
            frame_start = float(timestamps[i])
            frame_end = float(timestamps[i + 1])
            frame_duration = frame_end - frame_start
            if frame_duration < 0.1: continue
            if frame_duration <= 1.50:
                self.create_segment(frame_start, frame_duration, segments_idx)
                print(f"Segment {segments_idx:04d} | {frame_start:.3f}s → {frame_end:.3f}s | durée={frame_duration:.3f}s")
                segments_idx += 1
            else:
                n_sub = round(frame_duration / 1.49)
                for j in range(n_sub):
                    self.create_segment(frame_start, frame_duration, segments_idx)
                    print(f"Segment {segments_idx:04d} | {frame_start:.3f}s → {frame_start + frame_duration:.3f}s | durée={(frame_end - frame_start)/n_sub:.3f}s")
                    segments_idx += 1

