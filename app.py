from flask import Flask, request, render_template, redirect, url_for, send_file
import os
from werkzeug.utils import secure_filename
import firebase_admin
from firebase_admin import credentials, storage
from deepface import DeepFace
import pandas as pd
from PIL import Image
import tempfile
import gc
import shutil
from io import BytesIO
import zipfile

app = Flask(__name__)

# Path to your service account key JSON file
cred = credentials.Certificate("atithi-photobooth-firebase-adminsdk-x4dib-4f7ff74e61.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'atithi-photobooth.appspot.com'})

bucket = storage.bucket()

def upload_image_bytes_to_firebase(image_bytes, filename, folder_name):
    try:
        # Ensure folder creation by setting the path within the file name
        blob = bucket.blob(f'{folder_name}/{filename}')
        blob.upload_from_string(image_bytes, content_type='image/jpeg')
        url = blob.public_url
        print(f"Uploaded matched face URL: {url}")  # Debugging: Print URL to console
        return url
    except Exception as e:
        print(f"Error uploading to Firebase: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_db', methods=['POST'])
def upload_db():
    if 'file' not in request.files:
        return redirect(request.url)
    files = request.files.getlist('file')
    for file in files:
        if file.filename == '':
            continue
        filename = secure_filename(file.filename)
        blob = bucket.blob(f'db/{filename}')
        blob.upload_from_file(file, content_type=file.content_type)
    return "Congrats all images have been uploaded Mr Photographer"

@app.route('/process', methods=['GET', 'POST'])
def process():
    if request.method == 'POST':
        if 'file' not in request.files or 'user_name' not in request.form:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        filename = secure_filename(file.filename)

        # Get user name from form
        user_name = request.form.get('user_name')
        if not user_name:
            return "Please provide a name."

        print(f"User name: {user_name}")  # Debugging: Print user name

        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Save the reference image in the temporary directory
            reference_image_path = os.path.join(temp_dir, filename)
            file.save(reference_image_path)
            reference_image = Image.open(reference_image_path)
            reference_image.close()
            
            # Retrieve database images from Firebase
            blobs = bucket.list_blobs(prefix='db/')
            db_face_paths = []
            
            for blob in blobs:
                db_img_bytes = blob.download_as_bytes()
                db_img = Image.open(BytesIO(db_img_bytes))
                
                # Convert to RGB if image has an alpha channel
                if db_img.mode == 'RGBA':
                    db_img = db_img.convert('RGB')
                    
                db_img = db_img.resize((224, 224))
                db_img_path = os.path.join(temp_dir, blob.name.split('/')[-1])
                db_img.save(db_img_path)
                db_face_paths.append(db_img_path)
            
            # Process in smaller batches to avoid memory issues
            dfs = []
            batch_size = 5  # Adjust the batch size based on your memory constraints
            for i in range(0, len(db_face_paths), batch_size):
                batch_paths = db_face_paths[i:i+batch_size]
                df = DeepFace.find(img_path=reference_image_path, db_path=temp_dir, enforce_detection=False)
                if isinstance(df, pd.DataFrame):
                    print(f"Batch {i//batch_size + 1}: {df.shape[0]} faces matched")  # Debugging: Print number of faces matched
                    dfs.append(df)
                elif isinstance(df, list):
                    print(f"Batch {i//batch_size + 1}: {len(df)} faces matched")  # Debugging: Print number of faces matched
                    dfs.extend(df)
                gc.collect()  # Explicitly release memory after each batch
            
            matched_files = []
            for df in dfs:
                if isinstance(df, pd.DataFrame):
                    for index, row in df.iterrows():
                        face_path = row['identity']
                        face_img = Image.open(face_path)
                        if face_img.mode == 'RGBA':
                            face_img = face_img.convert('RGB')
                        img_byte_arr = BytesIO()
                        face_img.save(img_byte_arr, format='JPEG')
                        img_byte_arr = img_byte_arr.getvalue()
                        
                        # Upload matched face to Firebase
                        filename = f'matched_face_{index}.jpg'
                        print(f"Uploading {filename} to {user_name} directory")  # Debugging: Print upload info
                        url = upload_image_bytes_to_firebase(img_byte_arr, filename, user_name)
                        if url:
                            print(f"Matched face uploaded to Firebase: {url}")  # Debugging: Confirm upload
                            matched_files.append(url)
                elif isinstance(df, dict):
                    face_path = df['identity']
                    face_img = Image.open(face_path)
                    if face_img.mode == 'RGBA':
                        face_img = face_img.convert('RGB')
                    img_byte_arr = BytesIO()
                    face_img.save(img_byte_arr, format='JPEG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    # Upload matched face to Firebase
                    filename = f'matched_face_{index}.jpg'
                    print(f"Uploading {filename} to {user_name} directory")  # Debugging: Print upload info
                    url = upload_image_bytes_to_firebase(img_byte_arr, filename, user_name)
                    if url:
                        print(f"Matched face uploaded to Firebase: {url}")  # Debugging: Confirm upload
                        matched_files.append(url)
            
            if not matched_files:
                print("No matched files were uploaded to Firebase.")  # Debugging: Check if no files were uploaded

            # Return the matched faces to the user
            return render_template('matched_faces.html', images=matched_files, user_name=user_name)
        except Exception as e:
            print(f"Error: {e}")  # Debugging: Print error to console
            return str(e)
        finally:
            # Clean up temporary files
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)  # Remove the temporary directory and its contents

    return render_template('process.html')

@app.route('/download_matched_faces/<user_name>')
def download_matched_faces(user_name):
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, 'matched_faces.zip')

    try:
        # Retrieve matched faces from Firebase
        blobs = bucket.list_blobs(prefix=f'{user_name}/')
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for blob in blobs:
                blob_bytes = blob.download_as_bytes()
                blob_name = blob.name.split('/')[-1]
                blob_path = os.path.join(temp_dir, blob_name)
                with open(blob_path, 'wb') as img_file:
                    img_file.write(blob_bytes)
                zipf.write(blob_path, arcname=blob_name)

        return send_file(zip_path, as_attachment=True)
    except Exception as e:
        print(f"Error: {e}")  # Debugging: Print error to console
        return str(e)
    finally:
        # Clean up temporary files only after send_file completes
        def cleanup():
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        request.environ['werkzeug.server.shutdown'] = cleanup
        
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
