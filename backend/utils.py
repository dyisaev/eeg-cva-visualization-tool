import subprocess

def get_video_duration(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)
def get_video_fps(filename):
    result=subprocess.run(["ffprobe","-v", "error", "-select_streams", "v", "-of", "default=noprint_wrappers=1:nokey=1",
                           "-show_entries", "stream=r_frame_rate", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return eval(result.stdout)
