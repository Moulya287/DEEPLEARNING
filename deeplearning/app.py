from flask import Flask, render_template, request, send_from_directory, after_this_request
import os
import uuid
import shutil

app = Flask(__name__)

# Folder configuration
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

@app.route("/")
def home():
    """Home page - upload form"""
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    """Handle video upload and processing"""
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "No file selected", 400
    
    # Generate unique filename to avoid conflicts
    unique_id = str(uuid.uuid4())[:8]
    original_filename = file.filename
    filename = f"{unique_id}_{original_filename}"
    
    # Save uploaded file
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)
    
    # Temporary output path for processed video
    temp_output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"temp_{unique_id}.mp4")
    
    print(f"Processing video: {filename}")
    
    # Run detection script
    try:
        result = os.system(f'python detect.py "{input_path}" "{temp_output_path}"')
        if result != 0:
            print("Error in detection script")
    except Exception as e:
        print(f"Error: {e}")
    
    # Clean up uploaded file after processing
    @after_this_request
    def cleanup(response):
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
                print(f"Cleaned up: {input_path}")
        except Exception as e:
            print(f"Cleanup error: {e}")
        return response
    
    # Render result page
    return render_template("result.html")

@app.route("/video")
def video():
    """Serve the processed video"""
    video_path = os.path.join(app.config['OUTPUT_FOLDER'], "final_output.mp4")
    
    # Check if video exists
    if not os.path.exists(video_path):
        return "Video not found. Please process a video first.", 404
    
    # Check if download is requested
    download = request.args.get('download', False)
    
    if download:
        return send_from_directory(
            app.config['OUTPUT_FOLDER'], 
            "final_output.mp4", 
            as_attachment=True,
            download_name="detected_output.mp4"
        )
    
    return send_from_directory(
        app.config['OUTPUT_FOLDER'], 
        "final_output.mp4",
        mimetype='video/mp4'
    )

@app.route("/status")
def status():
    """Check if video is ready"""
    video_path = os.path.join(app.config['OUTPUT_FOLDER'], "final_output.mp4")
    if os.path.exists(video_path):
        return {"ready": True, "size": os.path.getsize(video_path)}
    return {"ready": False}

@app.route("/clear")
def clear():
    """Clear output video"""
    try:
        video_path = os.path.join(app.config['OUTPUT_FOLDER'], "final_output.mp4")
        if os.path.exists(video_path):
            os.remove(video_path)
        return {"success": True, "message": "Video cleared"}
    except Exception as e:
        return {"success": False, "message": str(e)}

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 SMART HIGHWAY AI MONITORING SYSTEM")
    print("="*50)
    print(f"📍 Upload folder: {UPLOAD_FOLDER}")
    print(f"📍 Output folder: {OUTPUT_FOLDER}")
    print("\n🌐 Starting web server...")
    print("👉 Open http://127.0.0.1:5000 in your browser")
    print("="*50 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)