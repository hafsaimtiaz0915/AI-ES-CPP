from flask import Flask, render_template, request, redirect, url_for, send_file
import subprocess
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    transcript = ""
    summary = ""
    
    if request.method == 'POST':
        video = request.files['video']
        if video:
            video_path = os.path.join(UPLOAD_FOLDER, 'video.mp4')
            video.save(video_path)

            # Run test.py (assumes it reads "video.mp4")
            subprocess.run(['python', 'test.py'], check=True)

            # Read the output
            try:
                with open('transcript_summary.txt', 'r', encoding='utf-8') as f:
                    content = f.read()
                    parts = content.split("SUMMARY:\n\n")
                    transcript = parts[0].replace("TRANSCRIPT:\n\n", "").strip()
                    summary = parts[1].strip() if len(parts) > 1 else ""
            except FileNotFoundError:
                transcript = "Error: Output file not found."

    return render_template('index.html', transcript=transcript, summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
